"""
TAL counterpart of the NeCo / VOC neighbour-label test (neco_diag/neighbor_label_agreement.py):
do an encoder's per-timestep nearest neighbours share the query timestep's verb label?

For every encoder spec, encodes all validation videos (backbone + neck, eval mode),
takes the level-0 FPN features (stride 1, one vector per input timestep) and reports,
over foreground timesteps (covered by >= 1 GT verb segment):

  cross-video : top-k cosine neighbours drawn from OTHER videos
                -> semantic consistency, analogue of VOC "cross image"
  within-video: top-k neighbours from the SAME video, excluding |dt| <= --excl steps
                -> are different actions in the same scene told apart (what TAL needs)

agreement = fraction of the k neighbours whose label set intersects the query's
            (background neighbours never agree); chance = the same fraction over all
            candidates; lift = (agreement - chance) / (1 - chance), as in the VOC test.

Encoder specs (name=spec):
  input            the frozen clip features themselves (no encoder)
  random           ActionFormer encoder at its seeded random init
  <path.pth.tar>   encoder (backbone + neck) from a checkpoint's EMA / teacher weights

usage:
  python tools/knn_label_agreement.py configs/epic_slowfast_verb.yaml \
      input=input random=random sup=ckpt/.../epoch_021.pth.tar --out figs/knn_slowfast.json
"""
import argparse
import json
import os
import random as pyrandom
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (libs/)

import numpy as np
import torch
import torch.nn.functional as F

from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed, load_init_encoder


def label_matrix(item, T, num_classes):
    """(T, C) bool: timestep t (grid coords) lies inside a segment of class c."""
    Y = torch.zeros(T, num_classes, dtype=torch.bool)
    if item['segments'] is None:
        return Y
    t = torch.arange(T, dtype=torch.float32)
    for (s, e), c in zip(item['segments'].tolist(), item['labels'].tolist()):
        Y[(t >= s) & (t <= e), c] = True
    return Y


@torch.no_grad()
def encode_all(cfg, spec, dataset, device):
    if spec == 'input':
        model = None
    else:
        fix_random_seed(cfg['init_rand_seed'], include_cuda=False)  # CPU seed = init; CUDA determinism breaks local attention
        model = make_meta_arch(cfg['model_name'], **cfg['model'])
        if spec != 'random':
            model = load_init_encoder(model, spec, device, prefer_ema=True)
        model = model.to(device).eval()
    feats, labels, vids = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        T = item['feats'].shape[-1]
        if model is None:
            f = item['feats'].t()
        else:
            x, m = model.preprocessing([item])
            fb, mb = model.backbone(x, m)
            fpn, _ = model.neck(fb, mb)
            f = fpn[0][0, :, :T].t().cpu()          # level 0: (T, D)
        feats.append(F.normalize(f.float(), dim=1))
        labels.append(label_matrix(item, T, cfg['dataset']['num_classes']))
        vids.append(torch.full((T,), i, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels), torch.cat(vids)


@torch.no_grad()
def knn_stats(X, Y, V, k, excl, n_query, seed, device, part=None):
    """part: optional (num_videos,) participant/kitchen index per video -> also report how
    often cross-video neighbours come from the same kitchen (scene-identity probe)."""
    g = torch.Generator().manual_seed(seed)
    fg = torch.nonzero(Y.any(1)).squeeze(1)
    q = fg[torch.randperm(len(fg), generator=g)[:n_query]]
    pos = torch.zeros(len(V), dtype=torch.long)             # timestep index within video
    for v in V.unique():
        idx = torch.nonzero(V == v).squeeze(1)
        pos[idx] = torch.arange(len(idx))
    Xd, Yd, Vd, Pd = X.to(device), Y.to(device).float(), V.to(device), pos.to(device)
    Kd = part.to(device)[Vd] if part is not None else None   # kitchen per timestep
    out = {}
    for mode in ('cross', 'within'):
        agree, chance, n = 0.0, 0.0, 0
        kagree, kchance = 0.0, 0.0
        for chunk in q.split(1024):
            chunk = chunk.to(device)
            sim = Xd[chunk] @ Xd.t()                          # (b, N)
            same_vid = Vd[chunk][:, None] == Vd[None, :]
            if mode == 'cross':
                valid = ~same_vid
            else:
                far = (Pd[chunk][:, None] - Pd[None, :]).abs() > excl
                valid = same_vid & far
            ok = valid.sum(1) >= k
            if ok.sum() == 0:
                continue
            chunk, sim, valid = chunk[ok], sim[ok], valid[ok]
            sim = sim.masked_fill(~valid, -1e4)
            nn_idx = sim.topk(k, dim=1).indices                # (b, k)
            share = (Yd[chunk] @ Yd.t()) > 0                    # (b, N) label sets intersect
            agree += share.gather(1, nn_idx).float().mean(1).sum().item()
            chance += ((share & valid).sum(1).float() / valid.sum(1).float()).sum().item()
            if Kd is not None and mode == 'cross':
                samek = Kd[chunk][:, None] == Kd[None, :]
                kagree += samek.gather(1, nn_idx).float().mean(1).sum().item()
                kchance += ((samek & valid).sum(1).float() / valid.sum(1).float()).sum().item()
            n += len(chunk)
        a, c = agree / n, chance / n
        out[mode] = {'knn_same_label': a, 'chance': c, 'lift': (a - c) / (1 - c), 'n_query': n}
        if Kd is not None and mode == 'cross':
            ka, kc = kagree / n, kchance / n
            out['cross_kitchen'] = {'knn_same_kitchen': ka, 'chance': kc, 'lift': (ka - kc) / (1 - kc)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('encoders', nargs='+', help='name=spec (input | random | ckpt path)')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--excl', type=int, default=16, help='within-video exclusion window (steps)')
    ap.add_argument('--n-query', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
    pyrandom.seed(args.seed); np.random.seed(args.seed)
    results = {'config': args.config, 'k': args.k, 'excl': args.excl}
    # EPIC video ids are <participant>_<n>; participants = kitchens
    pids = [dataset.data_list[i]['id'].split('_')[0] for i in range(len(dataset))]
    uniq = sorted(set(pids))
    part = torch.tensor([uniq.index(p) for p in pids])
    results['n_kitchens'] = len(uniq)
    for e in args.encoders:
        name, spec = e.split('=', 1)
        X, Y, V = encode_all(cfg, spec, dataset, device)
        r = knn_stats(X, Y, V, args.k, args.excl, args.n_query, args.seed, device, part)
        r['spec'] = spec
        r['fg_fraction'] = float(Y.any(1).float().mean())
        results[name] = r
        print('%-14s cross: knn %.3f chance %.3f lift %.3f | within: knn %.3f chance %.3f lift %.3f'
              % (name, r['cross']['knn_same_label'], r['cross']['chance'], r['cross']['lift'],
                 r['within']['knn_same_label'], r['within']['chance'], r['within']['lift']), flush=True)
        kk = r['cross_kitchen']
        print('%-14s cross-video neighbours from same kitchen: %.3f (chance %.3f, lift %.3f)'
              % (name, kk['knn_same_kitchen'], kk['chance'], kk['lift']), flush=True)
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
