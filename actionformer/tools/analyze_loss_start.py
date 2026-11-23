#!/usr/bin/env python
"""
Where does supervised training *start* for different encoder initialisations?

Motivation
----------
If the NeCo SSL objective were close to the downstream TAL objective, an
SSL-initialised encoder should make the supervised loss lower from the very
first optimisation steps, while the cls/reg/center heads are still at their
(seed-matched, identical) random initialisation and have not yet adapted.

This script compares the supervised training-loss trace of an SSL-initialised
run against a random-init run *at the start* of training:

  * first logged minibatch (step 10),
  * mean loss over epoch 1, with a paired standard error over its minibatches,
  * per-epoch mean difference (SSL - random) for the whole run.

The first epoch is the informative window: when both runs share the seed, they
also share the data order and the head init, so the difference is attributable
to the encoder. After epoch 1 the models have diverged and a batch-size-2 loss
is extremely noisy.

Usage
-----
python tools/analyze_loss_start.py \
    --pair "SlowFast" \
        ckpt/epic_slowfast_verb_reproduce/logs \
        ckpt/epic_slowfast_verb_from_ssl_neco/logs \
    --out figs/loss_start_comparison.png
"""
import argparse
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#8b5cf6', '#e0a800', '#6b7280']
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'


def load_scalar(logdir, tag='train/final_loss'):
    ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        raise SystemExit(f"tag '{tag}' not in {logdir}")
    ev = sorted(ea.Scalars(tag), key=lambda e: e.step)
    steps = np.array([e.step for e in ev], dtype=float)
    vals = np.array([e.value for e in ev], dtype=float)
    # some logdirs contain several event files from restarted jobs and thus
    # repeat a step; average the duplicates so the trace has one value/step
    uniq, inv = np.unique(steps, return_inverse=True)
    if len(uniq) != len(steps):
        sums = np.zeros(len(uniq))
        cnts = np.zeros(len(uniq))
        np.add.at(sums, inv, vals)
        np.add.at(cnts, inv, 1.0)
        steps, vals = uniq, sums / cnts
    return steps, vals


def epoch_means(steps, vals, ep):
    n = int(steps.max() // ep) + 1
    out = np.full(n, np.nan)
    for e in range(n):
        m = (steps >= e * ep) & (steps < (e + 1) * ep)
        if m.any():
            out[e] = vals[m].mean()
    return out


def paired_epoch(steps, base, ssl, ep, epoch):
    """Mean +- SE of (ssl - base) over the minibatches of one epoch."""
    m = (steps >= epoch * ep) & (steps < (epoch + 1) * ep)
    d = ssl[m] - base[m]
    if len(d) < 2:
        return float('nan'), float('nan'), int(len(d))
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))), int(len(d))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_SECONDARY)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(TEXT_SECONDARY)
    ax.grid(True, alpha=0.15)


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pair', action='append', nargs=3,
                   metavar=('LABEL', 'RANDOM_LOGDIR', 'SSL_LOGDIR'),
                   required=True,
                   help='repeat per encoder-init comparison')
    p.add_argument('--curve-labels', nargs='*', default=None,
                   help='labels (as given to --pair) to draw as curves; '
                        'default: the first two pairs')
    p.add_argument('--iters-per-epoch', type=int, default=247,
                   help='optimisation iterations per epoch (247 for the '
                        'batch-size-2 EPIC verb recipe, 21 epochs total)')
    p.add_argument('--zoom-epochs', type=int, default=2,
                   help='epochs shown in the curve panels')
    p.add_argument('--ylim', type=float, nargs=2, default=(0.0, 2.6),
                   help='y limits for the curve panels (the first minibatch '
                        'of an unlucky seed can be far above this)')
    p.add_argument('--tag', default='train/final_loss')
    p.add_argument('--out', default='figs/loss_start_comparison.png')
    p.add_argument('--json-out', default=None)
    args = p.parse_args()

    ep = args.iters_per_epoch
    fig = plt.figure(figsize=(12, 6.4), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15],
                          hspace=0.42, wspace=0.18)
    ax_curves = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_bar = fig.add_subplot(gs[1, :])
    fig.patch.set_facecolor(SURFACE)

    curve_labels = (args.curve_labels if args.curve_labels is not None
                    else [x[0] for x in args.pair[:2]])
    curve_axes = dict(zip(curve_labels, ax_curves))

    summary, bar_labels, bar_vals, bar_errs, bar_colors = [], [], [], [], []

    for i, (label, rand_dir, ssl_dir) in enumerate(args.pair):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        print(f"\n=== {label} ===")
        bs, bv = load_scalar(rand_dir, args.tag)
        ss, sv = load_scalar(ssl_dir, args.tag)

        if not np.array_equal(bs, ss):
            raise SystemExit(f"{label}: step grids differ; cannot pair exactly")
        d = sv - bv

        b_ep = epoch_means(bs, bv, ep)
        s_ep = epoch_means(ss, sv, ep)
        d0, d0_se, n0 = paired_epoch(bs, bv, sv, ep, 0)
        d1, d1_se, n1 = paired_epoch(bs, bv, sv, ep, 1)

        print(f"  first minibatch (step {bs[0]:.0f}): random {bv[0]:.4f} | "
              f"SSL {sv[0]:.4f} | diff {d[0]:+.4f}")
        print(f"  epoch 1 mean ({n0} minibatches): random {b_ep[0]:.4f} | "
              f"SSL {s_ep[0]:.4f} | diff {d0:+.4f} +/- {d0_se:.4f} "
              f"({100 * d0 / b_ep[0]:+.2f}%)")
        print(f"  epoch 2 mean ({n1} minibatches): random {b_ep[1]:.4f} | "
              f"SSL {s_ep[1]:.4f} | diff {d1:+.4f} +/- {d1_se:.4f}")
        print("  per-epoch diff (SSL - random): "
              + " ".join(f"{x:+.3f}" for x in (s_ep - b_ep)))

        summary.append(dict(label=label,
                            first_random=float(bv[0]), first_ssl=float(sv[0]),
                            epoch1_random=float(b_ep[0]),
                            epoch1_ssl=float(s_ep[0]),
                            epoch1_diff=d0, epoch1_diff_se=d0_se,
                            epoch1_rel_pct=100.0 * d0 / b_ep[0],
                            epoch2_diff=d1, epoch2_diff_se=d1_se,
                            per_epoch_diff=[float(x) for x in (s_ep - b_ep)]))

        if label in curve_axes:
            ax = curve_axes[label]
            m = bs <= args.zoom_epochs * ep
            for vals in (bv, sv):
                ax.plot(bs[m], vals[m], color=color, alpha=0.25, linewidth=0.8)
            k = 5
            for vals, name, ls in ((bv, 'random init', '-'),
                                   (sv, 'NeCo-SSL init', '--')):
                v = np.convolve(vals[m], np.ones(k) / k, mode='valid')
                ax.plot(bs[m][k - 1:], v, color=color, linewidth=2.2,
                        linestyle=ls, label=name)
            ax.axvline(ep, color=TEXT_SECONDARY, linewidth=0.8, alpha=0.4,
                       linestyle=':')
            ax.text(ep + 8, args.ylim[1] * 0.97, 'epoch 1', fontsize=7,
                    color=TEXT_SECONDARY, va='top')
            ax.set_ylim(*args.ylim)
            ax.set_xlabel('training iteration', color=TEXT_SECONDARY)
            ax.set_ylabel('training loss', color=TEXT_SECONDARY)
            ax.set_title(f'{label}: first {args.zoom_epochs} epochs',
                         color=TEXT_PRIMARY, fontsize=10)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=8)
            style(ax)

        bar_labels.append(label)
        bar_vals.append(100.0 * d0 / b_ep[0])
        bar_errs.append(100.0 * d0_se / b_ep[0])
        bar_colors.append(color)

    y = np.arange(len(bar_labels))
    ax_bar.barh(y, bar_vals, xerr=bar_errs, color=bar_colors, height=0.6,
                error_kw=dict(ecolor=TEXT_SECONDARY, lw=1, capsize=3))
    ax_bar.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax_bar.set_yticks(y, bar_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('epoch-1 loss difference  SSL $-$ random  '
                      '(% of the random-init epoch-1 loss)', color=TEXT_SECONDARY)
    ax_bar.set_title('Starting loss gap: positive = SSL init starts HIGHER, '
                     'i.e. no transfer advantage', color=TEXT_PRIMARY, fontsize=10)
    style(ax_bar)

    fig.tight_layout()
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\nSaved {args.out}")

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {args.json_out}")


if __name__ == '__main__':
    main()
