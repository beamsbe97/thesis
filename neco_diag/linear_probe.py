#!/usr/bin/env python
"""
Option 2 — linear segmentation probe, NeCo paper setting.

Frozen DINOv2 ViT-S/14 backbone (baseline) vs. the NeCo post-trained ViT-S/14,
with a 1x1 convolutional linear head trained on PASCAL VOC 2012 (trainaug) or
ADE20k, following the protocol in `experiments/linear_segmentation/` of
github.com/vpariza/NeCo (448x448 crops, SGD lr 0.01, 25 epochs, step decay at
epoch 20, CE loss, mIoU on 100x100 downsampled masks).

Also supports:
  --eval-only --head ...   evaluate a head trained with one backbone on another
                           backbone (the "frozen trained head" transfer test,
                           the direct analogue of the ActionFormer frozen-head
                           splice), reporting both mIoU and the *initial* CE
                           loss before any adaptation.

Outputs a JSON with per-epoch train loss / val mIoU and the final mIoU.

Example
-------
python linear_probe.py --backbone dinov2 --data voc \
    --ckpt ../ckpt/dinov2_vits14_pretrain.pth --out ../results/probe_dinov2_voc.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

# NeCo repo modules
from src.models.vit_v2 import vit_small as vit_small_v2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class SegDataset(Dataset):
    """VOC or ADE20k image/mask pairs, joint crop+flip for training."""

    def __init__(self, root, split, dataset, size=448, train=False, limit=0):
        self.size = size
        self.train = train
        if dataset == 'voc':
            split_file = 'trainaug' if split == 'train' else split
            names_file = os.path.join(root, 'sets', f'{split_file}.txt')
            with open(names_file) as f:
                names = [x.strip() for x in f if x.strip()]
            img_dir = os.path.join(root, 'images')
            mask_dir = os.path.join(
                root, 'SegmentationClassAug' if split.startswith('train')
                else 'SegmentationClass')
            self.images = [os.path.join(img_dir, n + '.jpg') for n in names]
            self.masks = [os.path.join(mask_dir, n + '.png') for n in names]
        elif dataset == 'ade20k':
            sub = 'training' if split == 'train' else 'validation'
            img_dir = os.path.join(root, 'images', sub)
            mask_dir = os.path.join(root, 'annotations', sub)
            names = sorted(os.listdir(img_dir))
            self.images = [os.path.join(img_dir, n) for n in names]
            self.masks = [os.path.join(mask_dir, n.replace('.jpg', '.png'))
                          for n in names]
        else:
            raise ValueError(dataset)
        if limit:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert('RGB')
        mask = Image.open(self.masks[i])
        if self.train:
            top, left, h, w = RandomResizedCrop.get_params(
                img, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3))
            img = TF.resized_crop(img, top, left, h, w,
                                  (self.size, self.size), InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, top, left, h, w,
                                   (self.size, self.size), InterpolationMode.NEAREST)
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            img = TF.resize(img, (self.size, self.size),
                            interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.size, self.size),
                             interpolation=InterpolationMode.NEAREST)
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_backbone(arch='vit-small', patch_size=14, num_register_tokens=0):
    return vit_small_v2(patch_size=patch_size,
                        num_register_tokens=num_register_tokens)


def load_backbone(model, ckpt, kind):
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # NeCo PyTorch-Lightning checkpoint: keep the student backbone
    if any(k.startswith('model.') for k in sd):
        sd = {k[len('model.'):]: v for k, v in sd.items()
              if k.startswith('model.')}
    # drop any SSL projection head; the linear head is built separately
    sd = {k: v for k, v in sd.items()
          if 'head' not in k and not k.startswith('projection')}
    msg = model.load_state_dict(sd, strict=False)
    print(f'[{kind}] load {ckpt}: missing={len(msg.missing_keys)} '
          f'unexpected={len(msg.unexpected_keys)}')
    if len(msg.missing_keys) > 8:
        print('  missing sample:', msg.missing_keys[:8])
    return model


@torch.no_grad()
def extract_tokens(model, imgs, size=100):
    """(B,3,H,W) -> (B,384,size,size) frozen patch features."""
    tokens = model.forward_backbone(imgs)
    b = imgs.size(0)
    res = int(round((imgs.size(-1) / model.patch_size)))
    tokens = tokens[:, 1:].reshape(b, res, res, model.embed_dim)
    tokens = tokens.permute(0, 3, 1, 2)
    return F.interpolate(tokens, size=(size, size), mode='bilinear',
                         align_corners=False)


def downsample_mask(masks, size=100):
    """(B,H,W) long -> (B,size,size) long, nearest (paper protocol)."""
    return F.interpolate(masks.unsqueeze(1).float(), size=(size, size),
                         mode='nearest').squeeze(1).long()


def confusion(pred, gt, n_classes, ignore):
    m = gt != ignore
    pred = pred[m].view(-1)
    gt = gt[m].view(-1)
    idx = gt * n_classes + pred
    conf = torch.bincount(idx, minlength=n_classes ** 2)
    return conf.reshape(n_classes, n_classes).float()


def miou_from_conf(conf):
    tp = torch.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    present = (tp + fn) > 0
    return float(iou[present].mean()), iou


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', choices=['dinov2', 'neco'], required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--arch', default='vit-small')
    p.add_argument('--patch-size', type=int, default=14)
    p.add_argument('--num-register-tokens', type=int, default=0)
    p.add_argument('--data', choices=['voc', 'ade20k'], required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--decay-at', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--size', type=int, default=448)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--head', default=None, help='head checkpoint for --eval-only')
    p.add_argument('--max-train', type=int, default=0, help='debug: cap train images')
    p.add_argument('--max-val', type=int, default=0, help='debug: cap val images')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=400)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')

    num_classes = 21 if args.data == 'voc' else 151
    ignore = 255 if args.data == 'voc' else 0

    model = build_backbone(args.arch, args.patch_size,
                           args.num_register_tokens)
    model = load_backbone(model, args.ckpt, args.backbone)
    model.eval().to(device)
    for q in model.parameters():
        q.requires_grad_(False)
    embed_dim = model.embed_dim
    head = nn.Conv2d(embed_dim, num_classes, 1).to(device)

    val_ds = SegDataset(args.data_root, 'val', args.data, args.size, train=False,
                        limit=args.max_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    def evaluate():
        head.eval()
        conf = torch.zeros(num_classes, num_classes, device=device)
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                masks = downsample_mask(masks, 100)
                feats = extract_tokens(model, imgs, 100)
                logits = head(feats)
                loss = F.cross_entropy(logits, masks,
                                       ignore_index=ignore) * imgs.size(0)
                loss_sum += float(loss)
                n += imgs.size(0)
                pred = logits.argmax(1)
                conf += confusion(pred, masks, num_classes, ignore)
        head.train()
        m, per_class = miou_from_conf(conf)
        return m, loss_sum / n

    if args.eval_only:
        head.load_state_dict(torch.load(args.head, map_location=device))
        m, loss = evaluate()
        print(f'eval: mIoU {m:.4f} | CE {loss:.4f}')
        json.dump({'backbone': args.backbone, 'data': args.data,
                   'eval_only': True, 'mIoU': m, 'ce_loss': loss},
                  open(args.out, 'w'), indent=2)
        return

    train_ds = SegDataset(args.data_root, 'train', args.data, args.size, train=True,
                          limit=args.max_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    print(f'train {len(train_ds)} images, {len(train_loader)} iters/epoch; '
          f'val {len(val_ds)} images')

    opt = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.decay_at,
                                            gamma=0.1)

    history = []
    best = (-1, None)
    for ep in range(args.epochs):
        t0 = time.time()
        head.train()
        run_loss, nb = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device,
                                                                     non_blocking=True)
            masks = downsample_mask(masks, 100)
            with torch.no_grad():
                feats = extract_tokens(model, imgs, 100)
            logits = head(feats)
            loss = F.cross_entropy(logits, masks, ignore_index=ignore)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss)
            nb += 1
        sched.step()
        m, val_loss = evaluate()
        history.append({'epoch': ep, 'train_loss': run_loss / nb,
                        'val_loss': val_loss, 'miou': m,
                        'lr': opt.param_groups[0]['lr'],
                        'minutes': (time.time() - t0) / 60})
        print(f'epoch {ep:02d} | train {run_loss / nb:.4f} | val {val_loss:.4f} '
              f'| mIoU {m:.4f} | {history[-1]["minutes"]:.1f} min', flush=True)
        if m > best[0]:
            best = (m, ep)
            torch.save(head.state_dict(), args.out.replace('.json', '_head.pt'))

    result = {'backbone': args.backbone, 'data': args.data,
              'epochs': args.epochs, 'batch_size': args.batch_size,
              'lr': args.lr, 'seed': args.seed,
              'mIoU': best[0], 'best_epoch': best[1],
              'final_mIoU': history[-1]['miou'], 'history': history,
              'ckpt': args.ckpt}
    json.dump(result, open(args.out, 'w'), indent=2)
    print(f"done: best mIoU {best[0]:.4f} @ epoch {best[1]}")


if __name__ == '__main__':
    main()
