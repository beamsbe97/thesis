#!/usr/bin/env python
"""
Option 4 - gradient alignment between the NeCo SSL objective and the
downstream semantic-segmentation objective.

For each batch of PASCAL VOC images we compute, on the *same* ViT-S/14
backbone parameters:

  g_ssl : grad of the NeCo patch-neighbour-consistency loss (the actual
          `src.neco.NeCo.shared_step`, teacher + student, multi-crop)
  g_ce  : grad of the linear-probe cross-entropy on 100x100 interpolated
          patch features, using the frozen head trained in option 2

and report the cosine similarity and the norm ratio ||g_ssl|| / ||g_ce||,
globally and per parameter group.  A cosine near 0 means the SSL objective
pulls the backbone in a direction essentially unrelated to the downstream
one (the "too far from downstream" hypothesis); a large positive cosine
means the opposite.

The same script is run with the DINOv2 baseline weights and with the NeCo
weights; both use the *same* NeCo loss (same objective, different features),
so the difference isolates what NeCo's post-training did to the geometry.

A CE-vs-CE cosine between two disjoint batches is also reported as an
empirical noise floor for gradient cosine estimates.

Example
-------
python grad_alignment.py --backbone neco \
    --student-ckpt ../ckpt/neco_dinov2_vits14_student.ckpt \
    --teacher-ckpt ../ckpt/neco_dinov2_vits14_teacher.ckpt \
    --head ../results/probe_neco_voc_head.pt \
    --data-root ../data/voc/VOCSegmentation \
    --out ../results/grad_align_neco.json
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.neco import NeCo
from src.neco_transforms import NeCoTransforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def read_names(root, split):
    split_file = 'trainaug' if split == 'train' else split
    with open(os.path.join(root, 'sets', f'{split_file}.txt')) as f:
        return [x.strip() for x in f if x.strip()]


def load_image(path):
    return Image.open(path).convert('RGB')


class SegBatch:
    """Whole-image 448x448 views + masks, deterministic (val-style) resize."""

    def __init__(self, root, names, size=448):
        self.root = root
        self.names = names
        self.size = size

    def get(self, idxs):
        imgs, masks = [], []
        for i in idxs:
            n = self.names[i]
            img = load_image(os.path.join(self.root, 'images', n + '.jpg'))
            mask = Image.open(os.path.join(
                self.root, 'SegmentationClassAug', n + '.png'))
            img = TF.resize(img, (self.size, self.size),
                            interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.size, self.size),
                             interpolation=InterpolationMode.NEAREST)
            imgs.append(TF.normalize(TF.to_tensor(img), MEAN, STD))
            masks.append(torch.from_numpy(np.array(mask)).long())
        return torch.stack(imgs), torch.stack(masks)


def to_neco_batch(imgs_pil, transform):
    """list[PIL] -> (list[Tensor] one per crop, bboxes dict).

    Matches the official data module: the dataset yields a list of crops and
    default collate stacks crop j of every image, so `inputs[j]` is (B,3,H,W)
    and NeCo's shared_step can slice by crop id.
    """
    crops, bboxes = zip(*[transform(im) for im in imgs_pil])
    inputs = [torch.stack([c[j] for c in crops])
              for j in range(len(crops[0]))]
    gc = torch.stack([b['gc'] for b in bboxes])
    al = torch.stack([b['all'] for b in bboxes])
    return inputs, {'gc': gc, 'all': al}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_neco(args):
    model = NeCo(
        gpus=1, num_samples=10582, batch_size=args.batch_size, max_epochs=25,
        lr_heads=1e-3, lr_backbone=1e-5, final_lr=1e-6,
        weight_decay_end=0.04, weight_decay=0.04,
        projection_hidden_dim=2048, projection_feat_dim=256,
        n_layers_projection_head=3, crops_for_assign=[0, 1],
        nmb_crops=[2, 4], num_classes=21, val_iters=8,
        num_clusters_kmeans=[21], use_teacher=True, loss_mask='all',
        queue_length=0, momentum_teacher=0.9995, momentum_teacher_end=1.,
        exclude_norm_bias=True, optimizer='adamw', num_nodes=1,
        patch_size=14, roi_align_kernel_size=7, val_downsample_masks=True,
        arch='vit-small', student_steepness=100, teacher_steepness=100,
        arch_version='v2', grad_norm_clipping=None, num_register_tokens=0,
        sort_net='bitonic', trainable_blocks=None,
        is_queue_usable=False, sim_dist='cosine')
    return model


def load_weights(model, args):
    """Load plain backbone (+ SSL projection head) state dicts.

    The released NeCo checkpoints are plain dicts with `projection_head.*`
    keys (no Lightning wrapper); DINOv2's pretrain file is a plain dict too.
    Both modules get their own weights; the teacher keeps its own head.
    """
    def clean(path):
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        if any(k.startswith('model.') for k in sd):
            sd = {k[len('model.'):]: v for k, v in sd.items()
                  if k.startswith('model.')}
        return sd

    sd_s = clean(args.student_ckpt)
    sd_t = clean(args.teacher_ckpt or args.student_ckpt)
    m1 = model.model.load_state_dict(sd_s, strict=False)
    m2 = model.teacher.load_state_dict(sd_t, strict=False)
    print(f'[load] student missing={len(m1.missing_keys)} '
          f'unexpected={len(m1.unexpected_keys)} | teacher missing='
          f'{len(m2.missing_keys)} unexpected={len(m2.unexpected_keys)}')
    if m1.missing_keys:
        print('  student missing sample:', m1.missing_keys[:6])
    return model


def ce_loss(model, head, imgs, masks, ignore=255, out_res=100):
    tokens = model.model.forward_backbone(imgs)
    b = imgs.size(0)
    res = imgs.size(-1) // model.patch_size
    f = tokens[:, 1:].reshape(b, res, res, model.model.embed_dim)
    f = f.permute(0, 3, 1, 2)
    f = F.interpolate(f, size=(out_res, out_res), mode='bilinear',
                      align_corners=False)
    logits = head(f)
    masks = F.interpolate(masks.unsqueeze(1).float(), size=(out_res, out_res),
                          mode='nearest').squeeze(1).long()
    return F.cross_entropy(logits, masks, ignore_index=ignore)


def collect(model):
    out = {}
    for n, p in model.model.named_parameters():
        if 'head' in n or p.grad is None:
            continue
        out[n] = p.grad.detach().float().cpu()
    return out


def group_of(name):
    if name.startswith('blocks.'):
        return 'blocks.' + name.split('.')[1]
    return name.split('.')[0]


def compare(g1, g2):
    names = [n for n in g1 if n in g2 and g1[n].numel() == g2[n].numel()]
    if not names:
        return {}
    a = torch.cat([g1[n].view(-1) for n in names])
    b = torch.cat([g2[n].view(-1) for n in names])
    cos = float(F.cosine_similarity(a, b, dim=0))
    ratio = float(a.norm() / b.norm().clamp(min=1e-12))
    groups = {}
    for n in names:
        v = F.cosine_similarity(g1[n].view(-1), g2[n].view(-1), dim=0)
        groups.setdefault(group_of(n), []).append(float(v))
    return {'cosine': cos, 'norm_ratio': ratio,
            'n_params': int(a.numel()),
            'groups': {k: float(np.mean(v)) for k, v in groups.items()}}


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', choices=['dinov2', 'neco'], required=True)
    p.add_argument('--student-ckpt', required=True)
    p.add_argument('--teacher-ckpt', default=None)
    p.add_argument('--head', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=400)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    print('device', device)
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))

    model = build_neco(args)
    model = load_weights(model, args)
    model.to(device)
    model.eval()
    model.teacher.requires_grad_(False)
    # on_train_epoch_start() builds the diffsort sorters
    model.on_train_epoch_start()

    head = nn.Conv2d(model.model.embed_dim, 21, 1).to(device)
    head.load_state_dict(torch.load(args.head, map_location=device))
    head.eval()
    for q in head.parameters():
        q.requires_grad_(False)

    root = args.data_root
    names = read_names(root, args.split)
    random.shuffle(names)
    ds = SegBatch(root, names)
    ssl_tf = NeCoTransforms(
        size_crops=[518, 98], nmb_crops=[2, 4],
        min_scale_crops=[0.25, 0.05], max_scale_crops=[1., 0.25],
        jitter_strength=1.0, min_intersection=0.01, blur_strength=1.0)

    rows = []
    ce_grads = []          # for the CE-vs-CE noise floor
    t0 = time.time()
    for it in range(args.iters):
        idxs = [(it * args.batch_size + j) % len(names)
                for j in range(args.batch_size)]

        # ---- CE gradient (downstream) ----
        imgs, masks = ds.get(idxs)
        imgs, masks = imgs.to(device), masks.to(device)
        model.model.zero_grad(set_to_none=True)
        ce = ce_loss(model, head, imgs, masks)
        ce.backward()
        g_ce = collect(model)

        # ---- NeCo SSL gradient ----
        model.model.zero_grad(set_to_none=True)
        pils = [load_image(os.path.join(root, 'images', names[i] + '.jpg'))
                for i in idxs]
        inputs, bboxes = to_neco_batch(pils, ssl_tf)
        inputs = [x.to(device) for x in inputs]
        bboxes = {k: v.to(device) for k, v in bboxes.items()}
        ssl = model.shared_step((inputs, bboxes))
        ssl.backward()
        g_ssl = collect(model)

        row = compare(g_ssl, g_ce)
        row.update({'iter': it, 'ssl_loss': float(ssl), 'ce_loss': float(ce)})
        if len(ce_grads) >= 1:
            row['ce_vs_ce_prev'] = compare(g_ce, ce_grads[-1])['cosine']
        ce_grads.append(g_ce)
        rows.append(row)
        print(f"iter {it:02d} | ssl {float(ssl):.4f} | ce {float(ce):.4f} "
              f"| cos {row['cosine']:+.4f} | ||ssl||/||ce|| "
              f"{row['norm_ratio']:.4f} | {time.time() - t0:.1f}s", flush=True)

    def agg(key):
        v = [r[key] for r in rows if key in r]
        return {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                'min': float(np.min(v)), 'max': float(np.max(v))}

    groups = {}
    for r in rows:
        for k, v in r['groups'].items():
            groups.setdefault(k, []).append(v)

    result = {
        'backbone': args.backbone,
        'n_iters': args.iters,
        'batch_size': args.batch_size,
        'cosine': agg('cosine'),
        'norm_ratio': agg('norm_ratio'),
        'ce_vs_ce_noise': agg('ce_vs_ce_prev') if any(
            'ce_vs_ce_prev' in r for r in rows) else None,
        'group_cosine': {k: float(np.mean(v)) for k, v in groups.items()},
        'per_iter': rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, 'w'), indent=2)
    print(json.dumps({k: result[k] for k in
                      ['cosine', 'norm_ratio', 'ce_vs_ce_noise']}, indent=2))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
