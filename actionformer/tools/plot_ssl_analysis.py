"""
Produce the analysis figures for "why SSL on the ActionFormer encoder does not
help": encoder weight movement vs init, encoder representation similarity (CKA)
between trained models, and downstream mAP. Run on ALICE.

Inputs: SSL / supervised checkpoints (for the weight-movement computation),
`figs/encoder_similarity_*/summary.json` (for CKA), and training logs (for mAP).
Outputs: figs/ssl_analysis_*.png and figs/ssl_analysis_summary.json.
"""
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT2 = '#52514e'
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#7d4cba', '#8a8a8a']

FIG = os.path.expanduser('~/thesis/actionformer/figs')


def rel_delta(ckpt, cfg_path):
    """Relative L2 change of backbone/neck/head weights from a fresh seed init."""
    ck = torch.load(ckpt, map_location='cpu')
    cfg = load_config(cfg_path)
    fix_random_seed(cfg['init_rand_seed'], include_cuda=False)
    init = {k: v.detach().clone() for k, v in
            make_meta_arch(cfg['model_name'], **cfg['model']).state_dict().items()}
    stu = {k.replace('module.', '', 1): v for k, v in ck['state_dict'].items()}
    out = {}
    for p in ['backbone', 'neck', 'head']:
        ks = [k for k in stu if k.startswith(p) and k in init]
        if not ks:
            out[p] = None
            continue
        a = torch.cat([stu[k].flatten().float() for k in ks])
        b = torch.cat([init[k].flatten().float() for k in ks])
        out[p] = float((a - b).norm() / b.norm())
    return out


def read_summary(name):
    p = os.path.join(FIG, 'encoder_similarity_' + name, 'summary.json')
    return json.load(open(p))


def map_from_log(pattern):
    files = sorted(glob.glob(os.path.expanduser('~/thesis/actionformer/' + pattern)))
    if not files:
        return None
    vals = []
    with open(files[0]) as f:
        for line in f:
            if 'Average mAP' in line:
                vals.append(float(line.split(':')[1].strip().rstrip(' (%)')))
    return vals[-1] if vals else None


def main():
    os.makedirs(FIG, exist_ok=True)

    # ---------------- 1. encoder weight movement ----------------
    runs = [
        ('supervised (16 ep, lr 1e-4)',
         'ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar',
         'configs/epic_slowfast_verb.yaml'),
        ('SSL v1 SlowFast',
         'ckpt_ssl/ssl_epic_slowfast_verb_neco/epoch_012.pth.tar',
         'configs/ssl_epic_slowfast_verb.yaml'),
        ('SSL v1 VideoMAE (spatial)',
         'ckpt_ssl/ssl_epic_videomae_verb_spatial_neco_spatial/epoch_012.pth.tar',
         'configs/ssl_epic_videomae_verb_spatial.yaml'),
        ('SSL v2 VideoMAE (spatial)',
         'ckpt_ssl/ssl_epic_videomae_verb_spatial_v2_neco_spatial_v2/epoch_020.pth.tar',
         'configs/ssl_epic_videomae_verb_spatial_v2.yaml'),
    ]
    move = {}
    for name, ckpt, cfg in runs:
        move[name] = rel_delta(os.path.expanduser('~/thesis/actionformer/' + ckpt), cfg)
        print(name, move[name])

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    labels = list(move.keys())
    groups = ['backbone', 'neck', 'head']
    x = np.arange(len(labels))
    w = 0.26
    for gi, g in enumerate(groups):
        vals = [move[l][g] if move[l][g] is not None else 0.0 for l in labels]
        ax.bar(x + (gi - 1) * w, vals, w, label=g, color=COLORS[gi])
        for xi, v in zip(x + (gi - 1) * w, vals):
            if v > 0:
                ax.text(xi, v * 1.12, '%.3f' % v, ha='center', va='bottom', fontsize=7,
                        color=TEXT2)
    ax.set_yscale('log')
    ax.set_ylim(2e-4, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('relative L2 weight change from init (log scale)', color=TEXT2)
    ax.set_title('How much each training stage moves the encoder weights', color=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.15, which='both', axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ssl_analysis_encoder_movement.png'),
                facecolor=fig.get_facecolor())

    # ---------------- 2. representation similarity (CKA) ----------------
    pairs = [
        ('baseline vs SSL v1 (VideoMAE)', 'videomae'),
        ('baseline vs SSL v2 (VideoMAE)', 'videomae_v2'),
        ('baseline vs SSL v1 (SlowFast)', 'slowfast'),
        ('baseline vs baseline seed1 (SlowFast)', 'slowfast_seedcontrol'),
        ('SSL v1 vs baseline seed1 (SlowFast)', 'slowfast_ssl_vs_seed1'),
        ('SSL stage-A vs supervised (VideoMAE)', 'videomae_ssla_vs_sup'),
        ('random init vs supervised (VideoMAE)', 'videomae_rand_vs_sup'),
    ]
    cka = {}
    for label, key in pairs:
        s = read_summary(key)
        cka[label] = {'per_level': s['cka_per_level'],
                      'mean': s['cka_mean_over_levels'],
                      'cosine': s['cosine_mean'],
                      'wcos_backbone': s['weight_cosine_backbone']}
    nl = len(cka[pairs[0][0]]['per_level'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]; ax.set_facecolor(SURFACE)
    xs = np.arange(nl)
    for i, (label, _) in enumerate(pairs):
        ax.plot(xs, cka[label]['per_level'], '-o', ms=4, lw=1.8,
                color=COLORS[i % len(COLORS)], label=label)
    ax.set_xticks(xs); ax.set_xticklabels(['L%d' % l for l in range(nl)])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('linear CKA', color=TEXT2)
    ax.set_title('Encoder representation similarity per FPN level', color=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.15)

    ax = axes[1]; ax.set_facecolor(SURFACE)
    names = list(cka.keys())
    means = [cka[k]['mean'] for k in names]
    ax.barh(np.arange(len(names)), means, color=COLORS[0])
    ax.axvline(cka['baseline vs baseline seed1 (SlowFast)']['mean'],
               color=COLORS[1], ls='--', lw=1.5, label='seed-noise floor')
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('mean linear CKA', color=TEXT2)
    ax.set_title('Mean CKA between encoder representations', color=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT, loc='lower left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.15, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ssl_analysis_cka.png'),
                facecolor=fig.get_facecolor())

    # ---------------- 3. downstream mAP ----------------
    mseries = [
        ('baseline', 29.60),
        ('temporal SSL v1', 29.38),
        ('spatial SSL v1', 29.38),
        ('spatial+temporal', 28.90),
        ('spatial SSL v2 (fixed training)', 28.65),
    ]
    maps = list(mseries)
    print('mAP:', maps)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    labels = [l for l, _ in maps]
    vals = [v for _, v in maps]
    bars = ax.bar(np.arange(len(vals)), vals, color=[COLORS[1]] + [COLORS[0]] * (len(vals) - 1))
    for xi, v in zip(np.arange(len(vals)), vals):
        if v is not None:
            ax.text(xi, v + 0.05, '%.2f' % v, ha='center', va='bottom', fontsize=9, color=TEXT2)
    ax.axhline(29.60, color=COLORS[1], ls='--', lw=1.2)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(27, 30.2)
    ax.set_ylabel('avg mAP @ tIoU 0.1-0.5 (%)', color=TEXT2)
    ax.set_title('VideoMAE-L downstream: no SSL variant beats the supervised baseline', color=TEXT)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.15, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ssl_analysis_map.png'),
                facecolor=fig.get_facecolor())

    summary = {'weight_movement': move, 'cka': cka, 'mAP': dict(maps)}
    with open(os.path.join(FIG, 'ssl_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('saved figures to', FIG)


if __name__ == '__main__':
    main()
