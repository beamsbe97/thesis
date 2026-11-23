#!/usr/bin/env python
"""Collect the NeCo diagnostic JSONs into the report figure + table."""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'dinov2': 'DINOv2', 'neco': 'NeCo'}
C_DINO = '#4c72b0'
C_NECO = '#dd8452'


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/zfsstore/user/s4561341/neco_diag/results')
    ap.add_argument('--out', default='/zfsstore/user/s4561341/neco_diag/figs/neco_diagnostics.png')
    args = ap.parse_args()
    D = args.dir

    probe = {}
    for bb in ['dinov2', 'neco']:
        p = os.path.join(D, f'probe_{bb}_voc.json')
        if os.path.exists(p):
            probe[bb] = load(p)
    grad = {}
    for bb in ['dinov2', 'neco']:
        p = os.path.join(D, f'grad_align_{bb}.json')
        if os.path.exists(p):
            grad[bb] = load(p)
    knn = {}
    for bb in ['dinov2', 'neco']:
        p = os.path.join(D, f'neighbor_agreement_{bb}.json')
        if os.path.exists(p):
            knn[bb] = load(p)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle('NeCo vs. DINOv2 on PASCAL VOC 2012 (ViT-S/14)', fontsize=13)

    # ---- (a) linear probe curves ----
    ax = axes[0, 0]
    for bb, c in [('dinov2', C_DINO), ('neco', C_NECO)]:
        if bb not in probe:
            continue
        h = probe[bb]['history']
        ep = [r['epoch'] for r in h]
        mi = [100 * r['miou'] for r in h]
        ax.plot(ep, mi, color=c, lw=1.8, label=f'{NAMES[bb]} (best {100 * probe[bb]["mIoU"]:.1f})')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val mIoU (%)')
    ax.set_title('(a) Frozen-backbone linear probe (1x1 conv)')
    ax.grid(alpha=.3)
    ax.legend(frameon=False, loc='lower right')
    if 'dinov2' in probe and 'neco' in probe:
        h0 = {b: probe[b]['history'][0] for b in ['dinov2', 'neco']}
        ax.text(.03, .04,
                'epoch-0 CE loss (train / val)\n'
                f"DINOv2 {h0['dinov2']['train_loss']:.3f} / {h0['dinov2']['val_loss']:.3f}\n"
                f"NeCo   {h0['neco']['train_loss']:.3f} / {h0['neco']['val_loss']:.3f}",
                transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(fc='white', ec='0.8', alpha=.9, boxstyle='round,pad=0.3'))

    # ---- (b) probe mIoU + trained-head transfer ----
    ax = axes[0, 1]
    labels, vals, colors = [], [], []
    for bb, c in [('dinov2', C_DINO), ('neco', C_NECO)]:
        if bb in probe:
            labels.append(f'{NAMES[bb]}\nown head')
            vals.append(100 * probe[bb]['mIoU'])
            colors.append(c)
    for tr, ev, c in [('dinov2', 'neco', C_NECO), ('neco', 'dinov2', C_DINO)]:
        p = os.path.join(D, f'xfer_{tr}head_{ev}.json')
        if os.path.exists(p):
            r = load(p)
            labels.append(f'{NAMES[tr]} head\non {NAMES[ev]}')
            vals.append(100 * r['mIoU'])
            colors.append(c)
    ax.bar(range(len(vals)), vals, color=colors, alpha=.85)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('mIoU (%)')
    ax.set_title('(b) Own head vs. frozen trained-head transfer')
    ax.grid(axis='y', alpha=.3)

    # ---- (c) gradient alignment ----
    ax = axes[1, 0]
    xs, cos, err, ratio, noise = [], [], [], [], []
    for i, bb in enumerate(['dinov2', 'neco']):
        if bb not in grad:
            continue
        g = grad[bb]
        xs.append(i)
        cos.append(g['cosine']['mean'])
        err.append(g['cosine']['std'])
        ratio.append(g['norm_ratio']['mean'])
        noise.append((g.get('ce_vs_ce_noise') or {}).get('mean', np.nan))
    if xs:
        ax.bar(xs, cos, yerr=err, color=[C_DINO, C_NECO][:len(xs)],
               alpha=.85, capsize=4)
        ax.axhline(0, color='k', lw=.8)
        for x, n, c in zip(xs, noise, [C_DINO, C_NECO]):
            if np.isfinite(n):
                ax.hlines(n, x - 0.4, x + 0.4, color=c, ls=':', lw=1.6)
        for x, c, r, n in zip(xs, cos, ratio, noise):
            ax.text(x, 0.015, f'cos={c:+.3f}\nnoise={n:+.3f}\n'
                              f'||ssl||/||ce||={r:.2f}',
                    ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(['DINOv2', 'NeCo'][:len(xs)])
        ax.set_ylabel('cos(g_ssl, g_ce)')
        ax.set_ylim(min(0, min(cos) - 0.05), max(0.15, max(cos) + 0.15))
        ax.set_title('(c) SSL vs. segmentation gradient (backbone)')
        ax.plot([], [], ls=':', color='gray', label='CE-vs-CE noise floor')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=.3)

    # ---- (d) patch kNN vs GT labels ----
    ax = axes[1, 1]
    metrics = ['within', 'cross']
    w = 0.35
    for i, (bb, c) in enumerate([('dinov2', C_DINO), ('neco', C_NECO)]):
        if bb not in knn:
            continue
        k = knn[bb]
        vals = [k['within_image']['knn_same_label'],
                k['cross_image']['knn_same_label']]
        ax.bar(np.arange(2) + (i - 0.5) * w, vals, w, color=c, alpha=.85,
               label=f'{NAMES[bb]} (NMI {k["kmeans"]["nmi"]:.3f})')
    if knn:
        chance = [np.mean([knn[b]['within_image']['chance'] for b in knn]),
                  np.mean([knn[b]['cross_image']['chance'] for b in knn])]
        for x, ch in enumerate(chance):
            ax.hlines(ch, x - 0.48, x + 0.48, color='k', lw=1.4,
                      label='chance' if x == 0 else None)
    ax.set_xticks(range(2))
    ax.set_xticklabels(['within image', 'cross image'])
    ax.set_ylabel(f'same-GT-label fraction in top-k neighbours')
    ax.set_title('(d) Patch neighbourhood vs. GT labels (k=10)')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print('wrote', args.out)

    # ---- markdown table ----
    print('\n| metric | DINOv2 | NeCo |')
    print('|---|---|---|')
    for key, fn in [
        ('linear probe mIoU', lambda b: f"{100 * probe[b]['mIoU']:.2f}" if b in probe else '-'),
        ('gradient cosine', lambda b: f"{grad[b]['cosine']['mean']:+.3f}" if b in grad else '-'),
        ('gradient norm ratio', lambda b: f"{grad[b]['norm_ratio']['mean']:.2f}" if b in grad else '-'),
        ('kNN same label, within', lambda b: f"{knn[b]['within_image']['knn_same_label']:.3f}" if b in knn else '-'),
        ('kNN same label, cross', lambda b: f"{knn[b]['cross_image']['knn_same_label']:.3f}" if b in knn else '-'),
        ('kNN vote acc, within', lambda b: f"{knn[b]['within_image']['vote_acc']:.3f}" if b in knn else '-'),
        ('k-means NMI', lambda b: f"{knn[b]['kmeans']['nmi']:.3f}" if b in knn else '-'),
        ('k-means ARI', lambda b: f"{knn[b]['kmeans']['ari']:.3f}" if b in knn else '-'),
    ]:
        print(f'| {key} | {fn("dinov2")} | {fn("neco")} |')


if __name__ == '__main__':
    main()
