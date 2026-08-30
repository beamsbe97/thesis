"""
Self-supervised dataset for ActionFormer (add-on).

Samples *two temporal crops* (views) of every video, together with the
crop windows expressed in the *original feature-grid coordinate system*.
The model later slices the two views, encodes them independently, and
aligns them on a shared temporal ROI with a 1D ROI Align.

No action labels / segments are required, so unlabeled videos can be used.
"""
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import register_dataset


def sample_temporal_crops(
    T,
    crop_scale=(0.4, 1.0),
    min_overlap=0.2,
    min_crop_len=128,
    num_trials=50,
):
    """
    Sample two temporal crop windows (start, end) in [0, T) (feature grid).

    Both crops have random lengths in [max(lo_ratio*T, min_crop_len),
    hi_ratio*T] and are guaranteed to intersect by at least
    min_overlap * min(len1, len2). The intersection defines the shared ROI
    used for temporal ROI Align.

    Returns ((s1, e1), (s2, e2)).
    """
    lo_ratio, hi_ratio = crop_scale
    lo_ratio = min(float(lo_ratio), float(hi_ratio))
    mc = max(4, int(min(min_crop_len, T)))

    def rand_len():
        rng_lo = min(max(lo_ratio * T, mc, 8), T)
        rng_hi = min(hi_ratio * T, T)
        rng_hi = max(rng_hi, rng_lo)
        return int(round(random.uniform(rng_lo, rng_hi)))

    for _ in range(num_trials):
        l1 = rand_len()
        l2 = rand_len()
        s1 = random.randint(0, T - l1)
        inter = int(round(min_overlap * min(l1, l2)))
        lo2 = max(0, s1 + inter - l2)
        hi2 = min(s1 + l1 - inter, T - l2)
        if hi2 < lo2:
            continue
        s2 = random.randint(lo2, hi2)
        return ((s1, s1 + l1), (s2, s2 + l2))

    # fallback: guarantee the largest possible intersection by nesting the
    # shorter crop inside the longer one (degenerate but valid corner case)
    if l2 <= l1:
        lo2 = s1
        hi2 = s1 + l1 - l2
    else:
        lo2 = s1 + l1 - l2
        hi2 = s1
    lo2 = max(0, lo2)
    hi2 = min(hi2, T - l2)
    if lo2 > hi2:
        s2 = max(0, min(s1, T - l2))
    else:
        s2 = random.randint(lo2, hi2)
    return ((s1, s1 + l1), (s2, s2 + l2))


@register_dataset("sslv1")
class SelfSupDataset(Dataset):
    """
    A generic dataset for self-supervised temporal-view training.

    Mirrors the feature loading logic of the supervised ActionFormer datasets
    (supports both .npy and .npz[feats]) but never needs annotations.
    """

    def __init__(
        self,
        is_training,       # if in training mode
        split,             # split, a tuple/list allowing concat of subsets
        feat_folder,       # folder for features
        json_file,         # json file for annotations (used to enumerate videos)
        feat_stride,       # temporal stride of the feats (metadata only)
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,       # cap the "full-length" context window
        file_prefix,
        file_ext,
        input_dim,
        num_classes,
        crop_scale,        # (lo, hi) crop lengths, relative to context length
        min_overlap,       # min overlap (= fraction of the smaller crop)
        min_crop_len,      # min crop length (in feature grid units)
        force_upsampling,  # ignored (kept for interface parity)
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
        self.min_crop_len = min_crop_len

        # load the video list
        self.data_list = self._load_json_db(json_file)

    def _load_json_db(self, json_file):
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # might be missing on Youtube videos (used for object-level stories)
        label_dict = {}
        for key, value in json_db.items():
            for act in value.get('annotations', []):
                label_dict[act['label']] = act['label_id']

        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split:
                continue
            feat_file = os.path.join(
                self.feat_folder, self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                fps = 1.0
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8
            dict_db += ({'id': key,
                         'fps': fps,
                         'duration': duration}, )
        assert len(dict_db) > 0, "no videos found for the given split"
        return dict_db

    def __len__(self):
        return len(self.data_list)

    def _load_feats(self, video_id):
        # T x C
        filename = os.path.join(
            self.feat_folder, self.file_prefix + video_id + self.file_ext)
        if self.file_ext.endswith('.npz'):
            with np.load(filename) as data:
                feats = data['feats'].astype(np.float32)
        else:
            feats = np.load(filename).astype(np.float32)
        feats = feats[::self.downsample_rate, :]
        # T x C -> C x T
        return torch.from_numpy(np.ascontiguousarray(feats.transpose()))

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        feats = self._load_feats(video_item['id'])
        T = feats.size(1)

        # cap the full-length context window to the model's max seq length
        if T > self.max_seq_len:
            st = random.randint(0, T - self.max_seq_len)
            feats = feats[:, st:st + self.max_seq_len]
            T = self.max_seq_len

        (s1, e1), (s2, e2) = sample_temporal_crops(
            T, self.crop_scale, self.min_overlap, self.min_crop_len)

        return {'video_id': video_item['id'],
                'feats': feats,
                'feats_lens': T,
                'crop_box': ((s1, e1), (s2, e2))}