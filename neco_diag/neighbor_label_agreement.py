#!/usr/bin/env python
"""
Option 5 - does the SSL objective's patch neighbourhood encode semantics?

NeCo is trained so that a patch and its k nearest neighbours (in feature
space) are consistently ordered.  If that objective is "too far" from the
downstream segmentation task, the neighbourhoods it shapes will not respect
ground-truth object boundaries.

For DINOv2 and NeCo we measure, on PASCAL VOC val patches (14x14 px each at
448 input, 32x32 grid):

  within-image kNN agreement   - fraction of a patch's k nearest neighbours
                                 (cosine, same image) that carry the same GT
                                 patch label (majority label of the 14x14
                                 block, blocks with >=50% one class only)
  cross-image kNN agreement    - same, neighbours drawn from the whole pool
  neighbour-vote accuracy      - majority label among the k neighbours equals
                                 the patch label
  chance                       - sum_c p_c^2 with p the class frequency
                                 (per image for within, global for cross)
  k-means NMI / ARI            - k=21 clustering of tokens vs GT labels

Also reports patch purity of the GT blocks themselves as a sanity check.

Example
-------
python neighbor_label_agreement.py --backbone neco \
    --ckpt ../ckpt/neco_dinov2_vits14_student.ckpt \
    --data-root ../data/voc/VOCSegmentation \
    --out ../results/neighbor_agreement_neco.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.models.vit_v2 import vit_small as vit_small_v2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_backbone(patch_size=14):
    return vit_small_v2(patch_size=patch_size, num_register_tokens=0)


def load_backbone(model, ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    if any(k.startswith('model.') for k in sd):
        sd = {k[len('model.'):]: v for k, v in sd.items()
              if k.startswith('model.')}
    sd = {k: v for k, v in sd.items()
          if 'head' not in k and not k.startswith('projection')}
    msg = model.load_state_dict(sd, strict=False)
    bad = [k for k in msg.missing_keys if 'head' not in k]
    print(f'[load] missing(non-head)={len(bad)} unexpected='
          f'{len(msg.unexpected_keys)}')
    if bad:
        print('  sample missing:', bad[:6])
    return model


@torch.no_grad()
def patch_tokens(model, img, size=448):
    """PIL image -> (tokens [N,384] L2-normalised, patch labels [N], purity)."""
    dev = next(model.parameters()).device
    x = TF.normalize(
        TF.to_tensor(TF.resize(img, (size, size),
                               interpolation=InterpolationMode.BILINEAR)),
        MEAN, STD).unsqueeze(0).to(dev)
    res = size // model.patch_size
    tok = model.forward_backbone(x)[0, 1:]
    tok = tok.reshape(res, res, model.embed_dim)
    tok = F.normalize(tok, dim=-1)
    return tok.reshape(-1, model.embed_dim)


@torch.no_grad()
def patch_labels(model, mask, size=448, thresh=0.5):
    res = size // model.patch_size
    p = model.patch_size
    m = TF.resize(mask, (size, size), interpolation=InterpolationMode.NEAREST)
    m = torch.from_numpy(np.array(m)).long()
    valid_px = (m >= 0) & (m < 21)
    oh = F.one_hot(m.clamp(0, 20), 21).float()          # H,W,21
    oh = oh * valid_px.unsqueeze(-1).float()
    counts = oh.reshape(res, p, res, p, 21).sum(dim=(1, 3))  # res,res,21
    tot = counts.sum(-1)
    lab = counts.argmax(-1)
    purity = counts.max(-1).values / tot.clamp(min=1)
    ok = (tot > 0) & (purity >= thresh)
    return lab.reshape(-1), ok.reshape(-1), purity.reshape(-1)


def knn_stats(query, db, qlab, dlab, k=10, exclude_self=False, chunk=2048):
    """Mean same-label fraction in top-k, vote accuracy, and chance."""
    same, vote_ok, n = 0.0, 0.0, 0
    q = F.normalize(query, dim=-1)
    d = F.normalize(db, dim=-1)
    for s in range(0, q.shape[0], chunk):
        qq = q[s:s + chunk]
        sim = qq @ d.T
        if exclude_self:
            sim[torch.arange(qq.shape[0]), s + torch.arange(qq.shape[0])] = -1e4
        idx = sim.topk(k, dim=-1).indices
        nl = dlab[idx]                                   # B,k
        ql = qlab[s:s + chunk]
        good = (nl == ql.unsqueeze(-1)) & (ql.unsqueeze(-1) >= 0)
        same += float(good.sum())
        # majority vote
        votes = torch.zeros(ql.shape[0], 21, dtype=torch.float,
                            device=ql.device)
        votes.scatter_add_(1, nl.clamp(min=0), torch.ones_like(nl, dtype=torch.float))
        vote = votes.argmax(1)
        vote_ok += float(((vote == ql) & (ql >= 0)).sum())
        n += int((ql >= 0).sum())
    return {'knn_same_label': same / max(n * k, 1),
            'vote_acc': vote_ok / max(n, 1)}


def chance_from_labels(labels):
    labels = labels[labels >= 0]
    if labels.numel() == 0:
        return float('nan')
    p = torch.bincount(labels, minlength=21).float()
    p = p / p.sum()
    return float((p ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', choices=['dinov2', 'neco'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-images', type=int, default=400)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')

    with open(os.path.join(args.data_root, 'sets', f'{args.split}.txt')) as f:
        names = [x.strip() for x in f if x.strip()]
    rng = np.random.RandomState(0)
    names = [names[i] for i in rng.permutation(len(names))[:args.n_images]]

    model = load_backbone(build_backbone(), args.ckpt).to(device).eval()

    all_tok, all_lab = [], []
    within_same, within_vote, within_chance = [], [], []
    purity_all, coverage = [], []
    for i, n in enumerate(names):
        img = Image.open(os.path.join(args.data_root, 'images', n + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(args.data_root, 'SegmentationClass',
                                       n + '.png'))
        tok = patch_tokens(model, img, args.size)
        lab, ok, pur = patch_labels(model, mask, args.size)
        sel = ok
        purity_all.append(float(pur[sel].mean()) if sel.any() else float('nan'))
        coverage.append(float(sel.float().mean()))
        if sel.sum() < args.k + 1:
            continue
        t = tok[sel]
        l = lab[sel].to(device)
        st = knn_stats(t, t, l, l, k=args.k, exclude_self=True)
        within_same.append(st['knn_same_label'])
        within_vote.append(st['vote_acc'])
        within_chance.append(chance_from_labels(l))
        all_tok.append(t.cpu())
        all_lab.append(l.cpu())
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(names)} images', flush=True)

    tok = torch.cat(all_tok)
    lab = torch.cat(all_lab)
    print('pool', tok.shape)

    # cross-image kNN: subsample queries and database for tractability
    g = torch.Generator().manual_seed(0)
    qidx = torch.randperm(tok.shape[0], generator=g)[:30000]
    didx = torch.randperm(tok.shape[0], generator=g)[:60000]
    cross = knn_stats(tok[qidx].to(device), tok[didx].to(device),
                      lab[qidx].to(device), lab[didx].to(device), k=args.k)

    # k-means on a subsample
    cidx = torch.randperm(tok.shape[0], generator=g)[:50000]
    feats = tok[cidx].numpy()
    labels = lab[cidx].numpy()
    km = KMeans(n_clusters=21, n_init=10, random_state=0).fit(feats)
    nmi = normalized_mutual_info_score(labels, km.labels_)
    ari = adjusted_rand_score(labels, km.labels_)

    result = {
        'backbone': args.backbone,
        'ckpt': args.ckpt,
        'n_images': len(names),
        'k': args.k,
        'patch_size': 14,
        'grid': args.size // 14,
        'patch_label_purity': float(np.nanmean(purity_all)),
        'valid_patch_fraction': float(np.mean(coverage)),
        'within_image': {
            'knn_same_label': float(np.mean(within_same)),
            'vote_acc': float(np.mean(within_vote)),
            'chance': float(np.mean(within_chance)),
            'chance_std': float(np.std(within_chance)),
            'lift': float((np.mean(within_same) - np.mean(within_chance))
                          / (1 - np.mean(within_chance))),
        },
        'cross_image': {
            'knn_same_label': cross['knn_same_label'],
            'vote_acc': cross['vote_acc'],
            'chance': chance_from_labels(lab),
            'lift': float((cross['knn_same_label'] - chance_from_labels(lab))
                          / (1 - chance_from_labels(lab))),
        },
        'kmeans': {'nmi': float(nmi), 'ari': float(ari), 'n': int(len(cidx))},
    }
    json.dump(result, open(args.out, 'w'), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
