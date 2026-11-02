"""
Extract *spatial-crop* feature views of EPIC-KITCHENS videos with the
OpenTAD EPIC-Verb-finetuned VideoMAE-L backbone, for the spatial-crop NeCo SSL
stage (see libs/datasets/ssl2d.py).

For every snippet (16 frames, stride 8 on a 30 fps timeline) we run VideoMAE
*once* on the full frame and keep its dense token grid (8 x 14 x 14 x 1024).
Each of the K spatial crops is then obtained by mean-pooling the tokens inside
its box (and over the 8 temporal tokens) -- i.e. no per-crop re-extraction.

Output layout (one folder per video):

    <output>/<video_id>/meta.json    {"K":.., "T":.., "grid":[14,14]}
    <output>/<video_id>/boxes.npy    (K,4) float32, normalized [x0,y0,x1,y1]
    <output>/<video_id>/view_000.npy (T,1024) fp16   (crop 0 is the full frame)

The 30 fps sampling matches the existing OpenTAD pooled features (3.75 feat/s),
so the SSL encoder sees the same temporal density as the downstream supervised
features.
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from videomae_backbone import load_videomae  # noqa: E402

# ImageNet normalization == mmaction's ActionDataPreprocessor on 0-255 inputs
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MEAN_T = torch.from_numpy(MEAN).view(3, 1, 1)
STD_T = torch.from_numpy(STD).view(3, 1, 1)
# fused normalization: (x/255 - mean)/std == x*scale + bias
SCALE_T = torch.from_numpy(((1.0 / 255.0) / STD).astype(np.float32)).view(3, 1, 1)
BIAS_T = torch.from_numpy((-MEAN / STD).astype(np.float32)).view(3, 1, 1)


def sample_crop_boxes(K, scale, rng):
    """K boxes, index 0 = full frame, rest random rectangles in `scale` area."""
    boxes = [[0.0, 0.0, 1.0, 1.0]]
    for _ in range(K - 1):
        area = float(rng.uniform(scale[0], scale[1]))
        ar = float(rng.uniform(3.0 / 4.0, 4.0 / 3.0))
        w = min(1.0, math.sqrt(area * ar))
        h = min(1.0, math.sqrt(area / ar))
        x0 = float(rng.uniform(0.0, 1.0 - w))
        y0 = float(rng.uniform(0.0, 1.0 - h))
        boxes.append([x0, y0, x0 + w, y0 + h])
    return np.asarray(boxes, dtype=np.float32)


def boxes_to_masks(boxes, grid):
    """(K,4) normalized boxes -> (K, grid*grid) bool token masks (cell center)."""
    masks = np.zeros((len(boxes), grid * grid), dtype=bool)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        for r in range(grid):
            cy = (r + 0.5) / grid
            for c in range(grid):
                cx = (c + 0.5) / grid
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    masks[k, r * grid + c] = True
        if not masks[k].any():                      # degenerate tiny box
            cc = min(grid - 1, int(round(((x0 + x1) / 2) * grid)))
            rr = min(grid - 1, int(round(((y0 + y1) / 2) * grid)))
            masks[k, rr * grid + cc] = True
    return masks


def load_video_list(json_file, split):
    db = json.load(open(json_file))['database']
    vids = []
    for key, value in db.items():
        if value['subset'].lower() != split:
            continue
        vids.append({'id': key,
                     'duration': value.get('duration', None),
                     'fps': value.get('fps', None)})
    return vids


def find_video(video_root, vid):
    p = vid.split('_')[0]
    return os.path.join(video_root, p, 'videos', vid + '.MP4')


@torch.no_grad()
def process_video(model, video_path, duration, boxes, masks, args, device):
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    if args.decode_threads:
        stream.thread_count = args.decode_threads
    fps = float(args.fps_override) if args.fps_override else float(stream.average_rate)
    ratio = fps / args.target_fps

    # native frame indices that map onto the 30 fps timeline
    if duration:
        n_target = int(duration * args.target_fps) + 1
        target_set = set(int(round(j * ratio)) for j in range(n_target))
    else:
        target_set = None

    grid = model.grid
    buf = collections.deque(maxlen=args.clip_len)
    pending = []
    views = []                                       # list of (K, D) arrays
    kept = 0
    n_clips = 0
    t_model = 0.0
    t_reformat = 0.0
    t_torch = 0.0
    t_decode = 0.0

    def flush():
        nonlocal t_model
        if not pending:
            return
        # pending: list of (T=16, 3, H, W) -> (B, 3, T, H, W)
        x = torch.stack(pending, dim=0).permute(0, 2, 1, 3, 4).to(device)
        _t = time.time()
        toks = model.forward_tokens(x)                # (B,8,14,14,D)
        t_model += time.time() - _t
        B, T, H, W, D = toks.shape
        toks = toks.reshape(B, T, H * W, D)
        pooled = []
        for k in range(len(boxes)):
            m = torch.from_numpy(masks[k]).to(device)
            pooled.append(toks[:, :, m, :].mean(dim=(1, 2)))  # (B,D)
        pooled = torch.stack(pooled, dim=1)           # (B,K,D)
        views.append(pooled.float().cpu().numpy())
        pending.clear()

    for i, frame in enumerate(container.decode(video=0)):
        _td = time.time()
        if target_set is not None:
            if i not in target_set:
                t_decode += time.time() - _td
                continue
        else:
            # no duration available: keep frames on the target-fps grid
            if i < int(round(kept * ratio)):
                t_decode += time.time() - _td
                continue
        t_decode += time.time() - _td

        _tr = time.time()
        f = frame.reformat(width=args.img_size, height=args.img_size, format='rgb24')
        arr = f.to_ndarray()
        t_reformat += time.time() - _tr

        _tt = time.time()
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        t.mul_(SCALE_T).add_(BIAS_T)
        t_torch += time.time() - _tt
        buf.append(t)
        kept += 1
        if kept >= args.clip_len and kept % args.stride == 0:
            pending.append(torch.stack(list(buf), dim=0))
            n_clips += 1
            if len(pending) >= args.batch_size:
                flush()
            if args.max_clips and n_clips >= args.max_clips:
                break
    flush()
    container.close()

    if not views:
        return None
    feats = np.concatenate(views, axis=0)            # (T, K, D)
    print('    [%s] clips=%d model=%.1fs decode=%.1fs reformat=%.1fs torch=%.1fs' % (
        os.path.basename(video_path), n_clips, t_model, t_decode, t_reformat, t_torch), flush=True)
    return feats


def save_video(out_dir, vid, feats, boxes, grid):
    d = os.path.join(out_dir, vid)
    os.makedirs(d, exist_ok=True)
    T, K, D = feats.shape
    for k in range(K):
        np.save(os.path.join(d, 'view_{:03d}.npy'.format(k)),
                feats[:, k, :].astype(np.float16))
    np.save(os.path.join(d, 'boxes.npy'), boxes.astype(np.float32))
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump({'K': int(K), 'T': int(T), 'grid': [int(grid), int(grid)]}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-root', required=True)
    ap.add_argument('--json-file', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='training')
    ap.add_argument('--num-crops', type=int, default=16)
    ap.add_argument('--crop-scale', type=float, nargs=2, default=[0.25, 1.0])
    ap.add_argument('--target-fps', type=float, default=30.0)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--decode-threads', type=int, default=0,
                    help='PyAV decode threads per process (0 = auto; set so '
                         'NPROC * threads ~= allocated CPUs)')
    ap.add_argument('--fps-override', type=float, default=None)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-clips', type=int, default=0,
                    help='debug: stop each video after N clips (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, flush=True)
    model = load_videomae(args.ckpt, device=device)
    grid = model.grid

    vids = load_video_list(args.json_file, args.split)
    vids.sort(key=lambda v: v['id'])
    if args.num_shards > 1:
        vids = [v for i, v in enumerate(vids) if i % args.num_shards == args.shard_index]
    if args.limit:
        vids = vids[:args.limit]
    print('videos to process:', len(vids), flush=True)

    os.makedirs(args.output, exist_ok=True)
    for n, v in enumerate(vids, 1):
        vid = v['id']
        meta = os.path.join(args.output, vid, 'meta.json')
        if os.path.exists(meta):
            print('[%d/%d] skip %s (done)' % (n, len(vids), vid), flush=True)
            continue
        path = find_video(args.video_root, vid)
        if not os.path.exists(path):
            print('[%d/%d] MISSING %s' % (n, len(vids), vid), flush=True)
            continue
        rng = np.random.default_rng(args.seed + (abs(hash(vid)) % (2 ** 31)))
        boxes = sample_crop_boxes(args.num_crops, args.crop_scale, rng)
        masks = boxes_to_masks(boxes, grid)
        try:
            t0 = time.time()
            feats = process_video(model, path, v['duration'], boxes, masks, args, device)
            if feats is None:
                print('[%d/%d] EMPTY %s' % (n, len(vids), vid), flush=True)
                continue
            save_video(args.output, vid, feats, boxes, grid)
            print('[%d/%d] %s -> %s in %.1fs' % (
                n, len(vids), vid, feats.shape, time.time() - t0), flush=True)
        except Exception as e:  # noqa: BLE001
            print('[%d/%d] FAIL %s: %r' % (n, len(vids), vid, e), flush=True)


if __name__ == '__main__':
    main()
