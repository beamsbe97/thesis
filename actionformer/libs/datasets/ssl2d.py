"""
Self-supervised dataset for ActionFormer -- *spatial-crop* variant (add-on).

Sibling of `libs/datasets/ssl.py`. Instead of sampling two temporal crops of a
single (spatially pooled) feature sequence, this dataset samples two *spatial*
crops of the video and feeds each crop's spatially pooled feature sequence to
the encoder as a separate view.

Motivation (NeCo, Pariza et al.): NeCo builds positive pairs from two spatial
crops (a global and a local one) of the same image and enforces patch-neighbor
consistency on the spatially aligned region. ActionFormer's encoder is 1D over
time, so -- as agreed -- we collapse each spatial crop to a (D, T) sequence by
spatial pooling and keep the existing NeCo loss over time. The "spatial" signal
is which region of the frame each view summarises.

Feature layout (produced by `tools/extract_videomae_spatial.py`):

    <feat_folder>/<video_id>/meta.json    # {"K": int, "T": int, "grid": [H, W]}
    <feat_folder>/<video_id>/boxes.npy    # (K, 4) float32, normalized [x0,y0,x1,y1]
    <feat_folder>/<video_id>/view_000.npy # (T, D) fp16  (one file per crop)

Storing one file per crop view means a training item only reads the two crops
it actually uses (instead of the whole (K, T, D) stack), which keeps the I/O
per epoch at ~2x the existing pooled-feature load.

No action labels / segments are required, so unlabeled videos can be used.
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import register_dataset
from .ssl import sample_temporal_crops


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter(box_a, box_b):
    w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return w * h


def select_spatial_pair(boxes, min_overlap=0.2, num_trials=50):
    """
    Pick two crop indices from a precomputed crop pool such that their spatial
    intersection covers at least `min_overlap` of the *smaller* crop's area.

    Mirrors `sample_temporal_crops` in ssl.py, but selecting from a fixed pool
    (the full spatial token grid is too large to store, so crops are
    pre-extracted once -- see the module docstring).

    Returns (i, j); falls back to the highest-overlap pair if no trial
    satisfies the constraint.
    """
    K = len(boxes)
    if K < 2:
        return 0, 0
    areas = [_area(b) for b in boxes]
    best, best_ratio = (0, 1), -1.0
    for _ in range(num_trials):
        i, j = random.sample(range(K), 2)
        ratio = _inter(boxes[i], boxes[j]) / max(1e-8, min(areas[i], areas[j]))
        if ratio > best_ratio:
            best_ratio, best = ratio, (i, j)
        if ratio >= min_overlap:
            return i, j
    return best


def select_spatial_pair_deterministic(boxes):
    """
    Validation-time pairing: the single highest-overlap pair, so the val SSL
    loss is comparable across runs (mirrors `rate_range=(1,1)` on val).
    """
    K = len(boxes)
    if K < 2:
        return 0, 0
    areas = [_area(b) for b in boxes]
    best, best_ratio = (0, 1), -1.0
    for i in range(K):
        for j in range(i + 1, K):
            ratio = _inter(boxes[i], boxes[j]) / max(1e-8, min(areas[i], areas[j]))
            if ratio > best_ratio:
                best_ratio, best = ratio, (i, j)
    return best


@register_dataset("sslv2")
class SelfSupSpatialDataset(Dataset):
    """
    Self-supervised dataset over precomputed *spatial-crop* feature views.

    Each item yields two spatial-crop views of the same video, both expressed
    in the same (capped) temporal window. The two views are the spatially
    pooled feature sequences of crops `i` and `j` (see `select_spatial_pair`).
    """

    def __init__(
        self,
        is_training,       # if in training mode
        split,             # split, a tuple/list allowing concat of subsets
        feat_folder,       # folder for the per-video spatial-crop view folders
        json_file,         # json file for annotations (used to enumerate videos)
        feat_stride,       # temporal stride of the feats (metadata only)
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,       # cap the "full-length" context window
        file_prefix,
        file_ext,          # ignored (kept for interface parity)
        input_dim,
        num_classes,
        crop_scale,        # (lo, hi) spatial area fraction -- metadata only,
                           #   crops are already extracted; kept for logging
        min_overlap,       # min spatial overlap (= fraction of the smaller crop)
        min_crop_len,      # ignored (temporal-crop param, kept for parity)
        force_upsampling,  # ignored (kept for interface parity)
        rate_range=(1.0, 1.0),  # ignored (kept for interface parity)
        **kwargs
    ):
        self.split = split
        self.is_training = is_training
        self.feat_folder = os.path.expanduser(feat_folder)
        assert os.path.exists(self.feat_folder) and os.path.exists(json_file)

        self.file_prefix = file_prefix if file_prefix is not None else ''
        self.file_ext = file_ext
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len

        self.crop_scale = list(crop_scale)
        self.min_overlap = min_overlap

        # load the video list (with per-video meta: T, K)
        self.data_list = self._load_json_db(json_file)

    # ------------------------------------------------------------------ #
    def _load_json_db(self, json_file):
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split:
                continue
            vid_dir = os.path.join(self.feat_folder, self.file_prefix + key)
            meta_file = os.path.join(vid_dir, 'meta.json')
            boxes_file = os.path.join(vid_dir, 'boxes.npy')
            if not (os.path.exists(meta_file) and os.path.exists(boxes_file)):
                continue
            with open(meta_file, 'r') as fid:
                meta = json.load(fid)
            dict_db += ({'id': key,
                         'T': int(meta['T']),
                         'K': int(meta['K']),
                         'grid': meta.get('grid')}, )
        assert len(dict_db) > 0, "no spatial-crop videos found for the given split"
        return dict_db

    def __len__(self):
        return len(self.data_list)

    def _load_view(self, video_id, crop_idx):
        # (T, D) fp16 -> (D, T) float32
        path = os.path.join(
            self.feat_folder, self.file_prefix + video_id,
            'view_{:03d}.npy'.format(crop_idx))
        feats = np.load(path).astype(np.float32)
        feats = feats[::self.downsample_rate, :]
        return torch.from_numpy(np.ascontiguousarray(feats.transpose()))

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        vid = video_item['id']
        T = video_item['T']

        boxes = np.load(os.path.join(
            self.feat_folder, self.file_prefix + vid, 'boxes.npy'))

        # temporal window: identical for both views, capped at max_seq_len
        st, ed = 0, T
        if T > self.max_seq_len:
            st = random.randint(0, T - self.max_seq_len)
            ed = st + self.max_seq_len
        win = ed - st

        if self.is_training:
            i, j = select_spatial_pair(boxes, self.min_overlap)
        else:
            i, j = select_spatial_pair_deterministic(boxes)

        v_i = self._load_view(vid, i)[:, st:ed]
        v_j = self._load_view(vid, j)[:, st:ed]

        # (2, D, T): view 0 = student, view 1 = EMA teacher
        view_feats = torch.stack([v_i, v_j], dim=0)

        inter = _inter(boxes[i], boxes[j])
        inter_ratio = inter / max(1e-8, min(_area(boxes[i]), _area(boxes[j])))

        return {
            'video_id': vid,
            'feats': v_i,                       # parity: view-0 sequence
            'view_feats': view_feats,           # (2, D, T)
            'feats_lens': win,
            # temporal ROI = the whole (capped) window for both views
            'crop_box': ((0, win), (0, win)),
            'crop_rate': (1.0, 1.0),
            # spatial bookkeeping (logging / analysis)
            'spatial_idx': (i, j),
            'spatial_box': (tuple(boxes[i].tolist()), tuple(boxes[j].tolist())),
            'spatial_inter': float(inter_ratio),
        }


@register_dataset("sslv3")
class SelfSupSpatialTemporalDataset(SelfSupSpatialDataset):
    """
    Combined **spatial + temporal** crops: each view is a precomputed
    spatial-crop view (same feature layout as `sslv2`) further restricted to a
    random temporal crop, and the two views are aligned on their temporal
    intersection by the existing 1D ROI Align.

    This reuses `build_views` unchanged: it selects `view_feats[view_idx]` (the
    spatial crop) and slices it with `crop_box[view_idx]` (the temporal crop).
    """

    def __init__(
        self,
        is_training,                      # positional (make_dataset passes it)
        split,                            # positional (make_dataset passes it)
        temporal_crop_scale=(0.4, 1.0),   # temporal crop lengths, rel. to window
        temporal_min_overlap=0.2,         # min temporal overlap (of smaller crop)
        temporal_min_crop_len=256,        # min temporal crop length (grid units)
        **kwargs
    ):
        super().__init__(is_training, split, **kwargs)
        self.temporal_crop_scale = list(temporal_crop_scale)
        self.temporal_min_overlap = temporal_min_overlap
        self.temporal_min_crop_len = temporal_min_crop_len

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        T = item['feats_lens']
        (s1, e1), (s2, e2) = sample_temporal_crops(
            T, self.temporal_crop_scale, self.temporal_min_overlap,
            self.temporal_min_crop_len)
        # view 0 (spatial crop i) also gets temporal crop (s1, e1), etc.
        item['crop_box'] = ((s1, e1), (s2, e2))
        item['crop_rate'] = (1.0, 1.0)
        return item
