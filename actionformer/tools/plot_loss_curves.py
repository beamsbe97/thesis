#!/usr/bin/env python
"""
Plot ActionFormer / NeCo-SSL training curves from TensorBoard event logs.

Reads one or more TensorBoard run directories (each containing
`events.out.tfevents.*` files, e.g. a `ckpt*/**/logs/` folder written by
train.py / train_ssl.py) and overlays a chosen scalar tag across runs.

Examples
--------
# NeCo-style SSL pretraining curve (train + val loss) for a single run
python tools/plot_loss_curves.py \\
    --run "SSL (untrained init)"=ckpt_ssl/ssl_epic_slowfast_verb_neco/logs \\
    --tag train/final_loss --val-tag validation/final_loss \\
    --ylabel "NeCo loss" --title "NeCo SSL pretraining (SlowFast, untrained init)" \\
    --out figs/ssl_pretrain_loss.png

# Supervised training loss: baseline vs. SSL-initialized encoder
python tools/plot_loss_curves.py \\
    --run "Supervised (random init)"=ckpt/epic_slowfast_verb_reproduce/logs \\
    --run "Supervised (from NeCo SSL init)"=ckpt/epic_slowfast_verb_from_ssl_neco/logs \\
    --tag train/final_loss --val-tag validation/mAP \\
    --ylabel "training loss" --title "ActionFormer supervised training loss" \\
    --out figs/supervised_loss_comparison.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# dataviz reference palette, categorical slots 1-3 (validated for both light
# and dark, all-pairs safe for up to 3 series)
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'


def load_scalar(logdir, tag):
    """Load one scalar tag from a TensorBoard logdir. Returns (steps, values)."""
    ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
    ea.Reload()
    available = ea.Tags().get('scalars', [])
    if tag not in available:
        print(f"  [warn] tag '{tag}' not found in {logdir} "
              f"(available: {available})")
        return [], []
    events = ea.Scalars(tag)
    # sort by step: event files from resumed/restarted jobs can be merged
    # out of order within a single logdir
    events = sorted(events, key=lambda e: e.step)
    return [e.step for e in events], [e.value for e in events]


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i + 1 - lo))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=LOGDIR',
                        help='a named TensorBoard run directory; repeat for '
                             'multiple runs to overlay')
    parser.add_argument('--tag', default='train/final_loss',
                        help='scalar tag to plot as the main (solid) curve')
    parser.add_argument('--val-tag', default=None,
                        help='optional second scalar tag, plotted dashed on '
                             'the same axes (e.g. a validation curve)')
    parser.add_argument('--smooth', type=int, default=1,
                        help='moving-average window (in logged points) '
                             'applied to the main tag (default: 1, no smoothing)')
    parser.add_argument('--xlabel', default='training step')
    parser.add_argument('--ylabel', default='loss')
    parser.add_argument('--title', default='')
    parser.add_argument('--out', required=True, help='output image path (.png)')
    parser.add_argument('--no-align-val-x', action='store_true',
                        help='by default, validation steps (logged once per '
                             'epoch) are rescaled onto the training-step '
                             'x-axis (logged every few iters); pass this to '
                             'plot the raw epoch index instead')
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        if '=' not in spec:
            parser.error(f"--run must be LABEL=LOGDIR, got: {spec}")
        label, logdir = spec.split('=', 1)
        if not os.path.isdir(logdir):
            parser.error(f"logdir does not exist: {logdir}")
        runs.append((label, logdir))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    any_data = False
    for i, (label, logdir) in enumerate(runs):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        print(f"Loading '{label}' from {logdir} ...")

        steps, vals = load_scalar(logdir, args.tag)
        if steps:
            any_data = True
            vals_s = moving_average(vals, args.smooth)
            ax.plot(steps, vals_s, color=color, linewidth=2, label=label)

        if args.val_tag:
            vsteps, vvals = load_scalar(logdir, args.val_tag)
            if vsteps:
                any_data = True
                # validation is typically logged once per epoch (small step
                # indices); rescale onto the training-step x-axis so both
                # curves share a meaningful axis
                if not args.no_align_val_x and steps and max(vsteps) > 0 \
                        and max(vsteps) < max(steps):
                    scale = max(steps) / max(vsteps)
                    vsteps = [s * scale for s in vsteps]
                ax.plot(vsteps, vvals, color=color, linewidth=2,
                        linestyle='--', marker='o', markersize=4,
                        label=f'{label} ({args.val_tag.split("/")[0]})')

    if not any_data:
        raise SystemExit("No matching scalar data found in any run; "
                          "check --tag / --val-tag and the logdir paths.")

    ax.set_xlabel(args.xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(args.ylabel, color=TEXT_SECONDARY)
    if args.title:
        ax.set_title(args.title, color=TEXT_PRIMARY, fontsize=12)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(TEXT_SECONDARY)
    ax.grid(True, alpha=0.15)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
