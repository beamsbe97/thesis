"""Build every data figure of the paper from the JSON files in ../data.

usage (from paper/):  python scripts/make_figures.py
needs numpy + matplotlib only. Writes vector PDFs to figs/.

Data provenance (all produced by the project's own runs):
  data/curves.json                per-epoch val mAP / NeCo metrics, scripts/parse_logs.py
  data/durations.json             EPIC-100 training-video durations (annotation file)
  data/teacher_student_delta.json actionformer/tools/ssl_teacher_movement.py
  data/ssl_mixer_levels.json      actionformer/tools/ssl_mixer_levels.py
  data/ssl_analysis_summary.json  actionformer/tools/plot_ssl_analysis.py (CKA)
  data/loss_start_comparison.json actionformer/tools/analyze_loss_start.py
  data/probe_*, xfer_*, neighbor_*, grad_align_*   neco_diag/*.py
"""
import json
import math
import os
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
FIGS = os.path.join(HERE, '..', 'figs')
os.makedirs(FIGS, exist_ok=True)

# validated reference palette (dataviz skill), first three categorical slots + neutrals
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, LIGHTGRAY = '#8a8a8a', '#d9d8d4'
TEXT, TEXT2 = '#0b0b0b', '#52514e'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8,
    'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.8, 'axes.edgecolor': TEXT2, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT2, 'ytick.color': TEXT2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ecebe8',
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'lines.linewidth': 1.6,
    'legend.frameon': False, 'pdf.fonttype': 42, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left',
})
FULL_W = 6.3  # inches, \textwidth of the paper


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def save(fig, name):
    """Vector PDF for the paper + a PNG copy (figs/png/) for Markdown docs."""
    fig.savefig(os.path.join(FIGS, name), bbox_inches='tight', pad_inches=0.02)
    os.makedirs(os.path.join(FIGS, 'png'), exist_ok=True)
    fig.savefig(os.path.join(FIGS, 'png', name.replace('.pdf', '.png')), dpi=200,
                bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print('wrote', name)


C = load('curves.json')
SUP = C['supervised']

# ---------------------------------------------------------------------------
# Fig. 2 - every SSL variant as a difference to its matched reference
# ---------------------------------------------------------------------------
SF_BASE = [23.07, 22.84, 24.23]
VM_BASE = 29.60


def opt(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d or (isinstance(d[k], list) and not d[k]):
            return None
        d = d[k]
    return d


def fig_results():
    sf_mean, sf_sd = np.mean(SF_BASE), np.std(SF_BASE, ddof=1)
    vm_sp = opt(SUP, 'vm_p3f_spatial')
    rows = [
        ('SlowFast', None, None),
        ('post-training, frozen heads (P2)', [21.73 - 23.07], 'post'),
        ('pre-training (P3), 3 seeds', [SUP[k][-1] - sf_mean for k in ('sf_p3f_s0', 'sf_p3f_s1', 'sf_p3f_s2')], 'pre'),
        ('P3 with the post-training SSL recipe', [SUP['sf_p3f_lowlr'][-1] - sf_mean], 'var'),
        ('VideoMAE-L', None, None),
        ('P3, temporal crops', [SUP['vm_p3f_temporal'][-1] - VM_BASE], 'pre'),
    ]
    if vm_sp:
        rows.append(('P3, spatial crops', [vm_sp[-1] - VM_BASE], 'var'))
    style = {'post': (ORANGE, 's', 'post-training (P2)'),
             'pre': (BLUE, 'o', 'pre-training (P3)'),
             'var': (AQUA, 'D', 'P3 variant')}
    fig, ax = plt.subplots(figsize=(FULL_W, 0.27 * len(rows) + 0.75))
    n = len(rows)
    ax.axvspan(-2 * sf_sd, 2 * sf_sd, color='#f1f0ed', zorder=0, lw=0)
    ax.axvspan(-sf_sd, sf_sd, color='#e4e3df', zorder=0, lw=0)
    ax.axvline(0, color=TEXT2, lw=0.8)
    seen, yt, yl = set(), [], []
    for i, (lab, d, kind) in enumerate(rows):
        y = n - 1 - i
        if d is None:
            ax.text(-1.95, y, lab, fontweight='bold', va='center', ha='left', fontsize=7.5, color=TEXT)
            continue
        col, mk, name = style[kind]
        ax.scatter(d, [y] * len(d), color=col, marker=mk, s=26, zorder=3, edgecolor='white',
                   linewidth=0.6, label=None if kind in seen else name)
        seen.add(kind)
        if len(d) > 1:
            ax.plot([np.mean(d)] * 2, [y - 0.32, y + 0.32], color=col, lw=1.4)
        ax.text(max(d) + 0.07, y, ('%+.2f' % np.mean(d)) + (' (mean)' if len(d) > 1 else ''),
                va='center', fontsize=6.5, color=TEXT2)
        yt.append(y); yl.append(lab)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_xlim(-2.0, 1.7); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('$\\Delta$ average mAP vs. matched supervised reference (pp)')
    ax.grid(axis='y', visible=False)
    ax.text(sf_sd, n - 0.55, '$\\pm$1 sd', ha='right', va='top', fontsize=6, color=TEXT2)
    ax.text(2 * sf_sd, n - 0.55, '$\\pm$2 sd', ha='right', va='top', fontsize=6, color=TEXT2)
    ax.legend(loc='lower right', handletextpad=0.3, fontsize=6.3)
    save(fig, 'fig_results.pdf')


# ---------------------------------------------------------------------------
# Fig. 3 - training curves: supervised val mAP and the SSL objective
# ---------------------------------------------------------------------------
def fig_trajectories():
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.1))
    ep = np.arange(1, 22)
    ax = axs[0]
    for i, k in enumerate(['sf_base_s1', 'sf_base_s2']):
        ax.plot(ep, SUP[k], color=BLUE, lw=1.2, label='random init (seeds 1, 2)' if i == 0 else None)
    for i, k in enumerate(['sf_p3f_s0', 'sf_p3f_s1', 'sf_p3f_s2']):
        ax.plot(ep, SUP[k], color=ORANGE, lw=1.2, label='SSL init (seeds 0-2)' if i == 0 else None)
    ax.set_xlim(8, 21.4); ax.set_ylim(14, 25.5); ax.set_xticks([10, 15, 20])
    ax.set_title('(a) SlowFast'); ax.set_xlabel('epoch'); ax.set_ylabel('val. average mAP (%)')
    ax.legend(loc='lower right', fontsize=6.2)

    ax = axs[1]
    ax.plot(ep, SUP['vm_base'], color=BLUE, lw=1.6, label='random init', zorder=4)
    ax.plot(ep, SUP['vm_p3f_temporal'], color=ORANGE, lw=1.2, label='SSL init, temporal crops')
    if opt(SUP, 'vm_p3f_spatial'):
        ax.plot(ep, SUP['vm_p3f_spatial'], color=AQUA, lw=1.2, ls='--', label='SSL init, spatial crops')
    ax.set_xlim(8, 21.4); ax.set_ylim(20, 30.8); ax.set_xticks([10, 15, 20])
    ax.set_title('(b) VideoMAE-L'); ax.set_xlabel('epoch')
    ax.legend(loc='lower right', fontsize=6.2)

    ax = axs[2]
    S = C['ssl']
    for k, col, lab in [('sf_f_s0', BLUE, 'SlowFast'), ('vm_f_temporal', ORANGE, 'VideoMAE-L temporal'),
                        ('vm_f_spatial', AQUA, 'VideoMAE-L spatial')]:
        if opt(S, k, 'agree'):
            ax.plot(np.arange(1, len(S[k]['agree']) + 1), S[k]['agree'], color=col, label=lab)
    for k in ['sf_f_s1', 'sf_f_s2']:
        ax.plot(np.arange(1, len(S[k]['agree']) + 1), S[k]['agree'], color=GRAY, lw=0.7, alpha=0.7)
    ax.plot([], [], color=GRAY, lw=0.7, label='SlowFast seeds 1, 2')
    ax.set_ylim(0.3, 1.0); ax.set_xlim(0.5, 20.5)
    ax.set_title('(c) SSL objective (stage A)'); ax.set_xlabel('SSL epoch')
    ax.set_ylabel('val. top-1 neighbour agreement')
    ax.legend(loc='lower right', fontsize=5.8, handlelength=1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig_trajectories.pdf')


# ---------------------------------------------------------------------------
# Fig. 5 - the SSL stage does train the encoder, but gives no head start
# ---------------------------------------------------------------------------
def fig_movement():
    M = load('fixed_movement.json')
    E1 = C['epoch1_avg_loss']
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W * 0.86, 2.2), gridspec_kw={'width_ratios': [1.5, 1.0]})
    ax = axs[0]
    rows = [('sup.\nfrom\nrandom', M['supervised (SlowFast, 21 ep)']['ema'], GRAY),
            ('sup.\nfrom\nSSL', M['P3 stage B, supervised from the SSL weights (SlowFast, mean of 3 seeds)']['ema'], TEXT2),
            ('SSL\nSlow-\nFast', M['P3 SSL, SlowFast']['ema'], BLUE),
            ('SSL\nVideo-\nMAE', M['P3 SSL, VideoMAE-L']['ema'], BLUE),
            ('P2\nSSL', M['P2 SSL post-training (SlowFast)']['ema'], ORANGE),
            ('SSL,\npost-tr.\nrecipe', M['P3 SSL, post-training recipe (SlowFast)']['ema'], AQUA)]
    x = np.arange(len(rows))
    ax.bar(x, [100 * r[1] for r in rows], 0.6, color=[r[2] for r in rows])
    for xi, r in zip(x, rows):
        ax.text(xi, 100 * r[1] * 1.15, '%.1f%%' % (100 * r[1]) if r[1] >= 0.01 else '%.2f%%' % (100 * r[1]),
                ha='center', va='bottom', fontsize=6, color=TEXT2)
    ax.set_yscale('log'); ax.set_ylim(0.2, 60)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_ylabel('backbone change (%)')
    ax.set_title('(a) encoder change from start')
    ax.grid(axis='x', visible=False)

    ax = axs[1]
    pairs = [('SF s1', E1['sf_base_s1'], E1['sf_p3f_s1']), ('SF s2', E1['sf_base_s2'], E1['sf_p3f_s2']),
             ('VM', E1['vm_base'], E1['vm_p3f_temporal'])]
    voc = [('VOC\ntrain', 0.6103, 0.5785), ('VOC\nval', 0.1993, 0.1793)]
    labs = [p[0] for p in pairs] + [v[0] for v in voc]
    vals = [100 * (b - a) / a for _, a, b in pairs] + [100 * (b - a) / a for _, a, b in voc]
    cols = [BLUE] * len(pairs) + [ORANGE] * len(voc)
    ax.bar(np.arange(len(vals)), vals, 0.6, color=cols)
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.axvline(len(pairs) - 0.5, color=LIGHTGRAY, lw=0.8)
    ax.text(1.0, 6.6, 'TAL (ours)', ha='center', fontsize=6, color=TEXT2)
    ax.text(3.5, 6.6, 'NeCo on VOC', ha='center', fontsize=6, color=TEXT2)
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylim(-11.5, 8)
    ax.set_ylabel('$\\Delta$ first-epoch loss (%)')
    ax.set_title('(b) head start?')
    ax.grid(axis='x', visible=False)
    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_movement.pdf')


# ---------------------------------------------------------------------------
# Fig. 5 - how much of the ROI the two temporal views see identically
# ---------------------------------------------------------------------------
def sample_temporal_crops(T, crop_scale, min_overlap, min_crop_len, num_trials=50):
    """Verbatim port of libs/datasets/ssl.py:sample_temporal_crops (successful branch)."""
    lo_ratio, hi_ratio = crop_scale
    mc = max(4, int(min(min_crop_len, T)))

    def rand_len():
        rng_lo = min(max(lo_ratio * T, mc, 8), T)
        rng_hi = max(min(hi_ratio * T, T), rng_lo)
        return int(round(random.uniform(rng_lo, rng_hi)))
    for _ in range(num_trials):
        l1, l2 = rand_len(), rand_len()
        s1 = random.randint(0, T - l1)
        inter = int(round(min_overlap * min(l1, l2)))
        lo2, hi2 = max(0, s1 + inter - l2), min(s1 + l1 - inter, T - l2)
        if hi2 < lo2:
            continue
        s2 = random.randint(lo2, hi2)
        return (s1, s1 + l1), (s2, s2 + l2)
    return (0, T), (0, T)


# receptive-field radius (input positions) at the output of each FPN level:
# 2 embedding convs (k=3): r=2; 2 stem blocks: q/k/v conv k=3 (+1) and local attention
# with window 9 (+4) each -> 12; each down-sampling block l=1..5 adds a k=3/stride-2
# conv at the input stride (+2^(l-1)) and +-4 attention at the output stride (+4*2^l).
RF = [12]
for l in range(1, 6):
    RF.append(RF[-1] + 2 ** (l - 1) + 4 * 2 ** l)


def view_identity(durations, fps_feat, frames, stride, max_len, crop_scale, min_crop, n_per_video=40,
                  n_bins=32, seed=0):
    random.seed(seed)
    frac = np.zeros(len(RF)); n = 0; roi_lens = []
    for dur in durations:
        T = int((dur * 30 - frames) // stride) + 1
        T = min(T, max_len)
        if T < 16:
            continue
        for _ in range(n_per_video):
            (s1, e1), (s2, e2) = sample_temporal_crops(T, crop_scale, 0.2, min_crop)
            rs, re = max(s1, s2), min(e1, e2)
            roi_lens.append(re - rs)
            p = rs + (re - rs) * (np.arange(n_bins) + 0.5) / n_bins
            for l, r in enumerate(RF):
                same = (np.maximum(p - r, s1) == np.maximum(p - r, s2)) & \
                       (np.minimum(p + r, e1) == np.minimum(p + r, e2))
                frac[l] += same.mean()
            n += 1
    return frac / n, np.array(roi_lens)


def fig_views():
    dur = list(load('durations.json').values())
    sf, sf_roi = view_identity(dur, 1.875, 32, 16, 2304, (0.4, 1.0), 128)
    vm, vm_roi = view_identity(dur, 3.75, 16, 8, 4608, (0.4, 1.0), 256)
    print('identical fraction SlowFast', np.round(sf, 3), 'median ROI', np.median(sf_roi))
    print('identical fraction VideoMAE', np.round(vm, 3), 'median ROI', np.median(vm_roi))
    json.dump({'rf_radius_positions': RF, 'slowfast_identical': sf.tolist(),
               'videomae_identical': vm.tolist(),
               'slowfast_roi_median': float(np.median(sf_roi)),
               'videomae_roi_median': float(np.median(vm_roi))},
              open(os.path.join(DATA, 'view_identity.json'), 'w'), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W * 0.78, 2.0))
    lv = np.arange(6)
    ax = axs[0]
    ax.bar(lv - 0.19, [2 * r * 16 / 30 for r in RF], 0.36, color=BLUE, label='SlowFast (stride 16)')
    ax.bar(lv + 0.19, [2 * r * 8 / 30 for r in RF], 0.36, color=ORANGE, label='VideoMAE-L (stride 8)')
    ax.axhline(np.median(sf_roi) * 16 / 30, color=TEXT2, ls='--', lw=0.9)
    ax.text(-0.4, np.median(sf_roi) * 16 / 30 * 1.03, 'median ROI length (both tracks)', ha='left', va='bottom',
            fontsize=6, color=TEXT2)
    ax.set_xticks(lv); ax.set_xticklabels(['L%d' % i for i in lv])
    ax.set_ylabel('receptive field (s)')
    ax.set_title('(a) receptive field vs. ROI')
    ax.legend(loc='upper left', fontsize=6)
    ax.grid(axis='x', visible=False)
    ax = axs[1]
    ax.plot(lv, 100 * sf, color=BLUE, marker='o', ms=4, label='SlowFast')
    ax.plot(lv, 100 * vm, color=ORANGE, marker='s', ms=4, label='VideoMAE-L (temporal)')
    ax.axhline(0, color=GRAY, lw=0.8, ls='--')
    ax.text(0.05, 3, 'image NeCo (photometric aug.): 0%', fontsize=6, color=TEXT2)
    ax.set_xticks(lv); ax.set_xticklabels(['L%d' % i for i in lv])
    ax.set_ylim(-5, 105)
    ax.set_ylabel('ROI bins with identical input (%)')
    ax.set_title('(b) view identity per FPN level')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.08), fontsize=6)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_views.pdf')


# ---------------------------------------------------------------------------
# Fig. 6 - neighbour structure: does SSL bring same-verb timesteps together?
# ---------------------------------------------------------------------------
def fig_representation():
    sf, vm = load('knn_label_slowfast.json'), load('knn_label_videomae.json')
    for extra, tgt in [('knn_label_slowfast_p2.json', sf), ('knn_label_videomae_spatial.json', vm)]:
        if os.path.exists(os.path.join(DATA, extra)):
            for k, v in load(extra).items():
                if isinstance(v, dict) and 'cross' in v:
                    tgt[k] = v
    order = [('input', 'input features', LIGHTGRAY), ('random', 'random encoder', GRAY),
             ('ssl', 'after NeCo SSL', ORANGE), ('ssl_spatial', 'after NeCo SSL (spatial)', '#f2a27f'),
             ('post_trained', 'supervised + NeCo post-train', '#b8481d'),
             ('supervised', 'supervised', BLUE), ('ssl_then_sup', 'NeCo SSL $\\rightarrow$ supervised', AQUA)]
    has_kitchen = 'cross_kitchen' in sf.get('ssl', {})
    ncol = 3 if has_kitchen else 2
    fig, axs = plt.subplots(1, ncol, figsize=(FULL_W, 2.35))
    for ai, (D, title) in enumerate([(sf, '(a) SlowFast'), (vm, '(b) VideoMAE-L')]):
        ax = axs[ai]
        present = [o for o in order if o[0] in D]
        w = 0.8 / len(present)
        for j, (key, lab, col) in enumerate(present):
            vals = [D[key]['cross']['lift'], D[key]['within']['lift']]
            ax.bar(np.arange(2) + (j - (len(present) - 1) / 2) * w, vals, w * 0.92, color=col, label=lab)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['across videos', 'within a video'])
        ax.set_ylim(0.2, 0.52 if ai == 0 else 0.56)
        ax.set_title(title)
        if ai == 0:
            ax.set_ylabel('verb-label lift over chance')
        ax.grid(axis='x', visible=False)
    handles, labels = [], []
    for a in axs[:2]:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.08))
    if has_kitchen:
        # same-kitchen fraction of cross-video neighbours, both tracks, same colours
        ax = axs[2]
        for gi, D in enumerate([sf, vm]):
            present = [o for o in order if o[0] in D and 'cross_kitchen' in D[o[0]]]
            w = 0.8 / len(present)
            for j, (key, lab, col) in enumerate(present):
                ax.bar(gi + (j - (len(present) - 1) / 2) * w, D[key]['cross_kitchen']['knn_same_kitchen'],
                       w * 0.92, color=col)
        ch = sf['input']['cross_kitchen']['chance']
        ax.axhline(ch, color=TEXT2, ls='--', lw=0.8)
        ax.text(1.45, ch + 0.012, 'chance', ha='right', va='bottom', fontsize=6, color=TEXT2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['SlowFast', 'VideoMAE-L'])
        ax.set_ylim(0, 0.6)
        ax.set_ylabel('cross-video neighbours\nfrom the same kitchen')
        ax.set_title('(c) scene identity')
        ax.grid(axis='x', visible=False)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.06, 1, 1))
    save(fig, 'fig_representation.pdf')


# ---------------------------------------------------------------------------
# Fig. 7 - NeCo in its own setting (DINOv2 ViT-S/14 -> PASCAL VOC)
# ---------------------------------------------------------------------------
def fig_voc():
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.0),
                            gridspec_kw={'width_ratios': [1.2, 1.0, 1.0]})
    ax = axs[0]
    for k, col, lab in [('probe_dinov2_voc.json', BLUE, 'DINOv2'), ('probe_neco_voc.json', ORANGE, 'NeCo')]:
        h = load(k)['history']
        ax.plot([e['epoch'] for e in h], [100 * e['miou'] for e in h], color=col,
                label='%s (best %.1f)' % (lab, 100 * load(k)['mIoU']))
    ax.set_xlabel('probe epoch'); ax.set_ylabel('val. mIoU (%)')
    ax.set_title('(a) frozen linear probe')
    ax.legend(loc='lower right')
    ax = axs[1]
    vals = [100 * load('probe_dinov2_voc.json')['mIoU'], 100 * load('xfer_dinov2head_neco.json')['mIoU'],
            100 * load('probe_neco_voc.json')['mIoU'], 100 * load('xfer_necohead_dinov2.json')['mIoU']]
    labs = ['DINOv2\n(own)', 'NeCo\n(DINOv2\nhead)', 'NeCo\n(own)', 'DINOv2\n(NeCo\nhead)']
    cols = [BLUE, ORANGE, ORANGE, BLUE]  # colour = backbone evaluated
    ax.bar(np.arange(4), vals, 0.62, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, '%.1f' % v, ha='center', fontsize=6, color=TEXT2)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labs, fontsize=5.8)
    ax.set_xlabel('backbone evaluated (head used)', fontsize=6.3)
    ax.set_ylim(0, 92); ax.set_ylabel('mIoU (%)')
    ax.set_title('(b) cross-head transfer')
    ax.grid(axis='x', visible=False)
    ax = axs[2]
    nd, nn_ = load('neighbor_agreement_dinov2.json'), load('neighbor_agreement_neco.json')
    for j, (d, col, lab) in enumerate([(nd, BLUE, 'DINOv2'), (nn_, ORANGE, 'NeCo')]):
        v = [d['within_image']['lift'], d['cross_image']['lift']]
        ax.bar(np.arange(2) + (j - 0.5) * 0.38, v, 0.36, color=col, label=lab)
        for i, vv in enumerate(v):
            ax.text(i + (j - 0.5) * 0.38, vv + 0.01, '%.2f' % vv, ha='center', fontsize=6, color=TEXT2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['within image', 'across images'])
    ax.set_ylim(0.7, 1.0); ax.set_ylabel('k-NN label lift over chance')
    ax.set_title('(c) neighbours share labels')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(axis='x', visible=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig_voc.pdf')


if __name__ == '__main__':
    fig_results()
    fig_trajectories()
    fig_movement()
    fig_views()
    fig_representation()
    fig_voc()
