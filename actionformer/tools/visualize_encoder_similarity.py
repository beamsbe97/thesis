"""
Visualise the similarity/difference between two trained ActionFormer encoders.

Both models are the *same* supervised architecture trained from different
initialisations (e.g. baseline vs SSL-pretrained encoder). For the same input
features we compare the encoder outputs (`backbone` -> `neck`, i.e. the FPN
feature pyramid that feeds the task heads):

  1. linear CKA per FPN level (representation similarity, linear-transform
     invariant);
  2. per-position cosine similarity distribution (are the feature vectors
     pointing the same way?);
  3. a 2D PCA of the concatenated features coloured by model (visual overlap);
  4. a weight-space cosine per module group (how close the learned weights are).

Example
-------
python tools/visualize_encoder_similarity.py \
    --config configs/epic_videomae_verb.yaml \
    --ckpt-a ckpt/epic_videomae_verb_reproduce/epoch_021.pth.tar --label-a "regular (supervised)" \
    --ckpt-b ckpt/epic_videomae_verb_from_ssl_neco_spatial/epoch_021.pth.tar --label-b "SSL-init (spatial)" \
    --num-videos 60 --out-dir figs/encoder_similarity_videomae
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT2 = '#52514e'
C_A = '#2a78d6'
C_B = '#eb6834'


def load_model(cfg, ckpt, device, use_ema=False, random_init=False):
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    if not random_init:
        sd = torch.load(ckpt, map_location='cpu')
        if use_ema and isinstance(sd, dict) and 'state_dict_ema' in sd:
            sd = sd['state_dict_ema']
        else:
            sd = sd.get('state_dict', sd)
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing
                   if not (m.startswith('point_generator')
                           or m.startswith('cls_head') or m.startswith('reg_head'))]
        unexpected = [u for u in unexpected if not u.startswith('head.')]
        if missing or unexpected:
            print('  [warn] missing=%d unexpected=%d (showing up to 5)' %
                  (len(missing), len(unexpected)))
            print('    missing:', missing[:5], 'unexpected:', unexpected[:5])
    model.eval().to(device)
    return model


@torch.no_grad()
def encode(model, video_list):
    """Return list per FPN level of (C, T_l) and valid mask (T_l,) bool."""
    x, mask = model.preprocessing(video_list)
    feats, masks = model.backbone(x, mask)
    fpn_feats, fpn_masks = model.neck(feats, masks)
    feats = [f[0].cpu() for f in fpn_feats]
    mks = [m[0].cpu().squeeze(0).bool() for m in fpn_masks]
    return feats, mks


def linear_cka(X, Y):
    """Linear CKA between (n, d) arrays (columns centered)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    num = (xty ** 2).sum()
    den = np.sqrt((xtx ** 2).sum()) * np.sqrt((yty ** 2).sum())
    return float(num / (den + 1e-12))


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :2] * S[:2]).astype(np.float32)


def weight_cosine(model_a, model_b, prefix):
    va, vb = [], []
    sd_a = dict(model_a.named_parameters())
    sd_b = dict(model_b.named_parameters())
    for k in sd_a:
        if k.startswith(prefix) and k in sd_b:
            va.append(sd_a[k].detach().cpu().flatten())
            vb.append(sd_b[k].detach().cpu().flatten())
    if not va:
        return float('nan')
    va = torch.cat(va).numpy()
    vb = torch.cat(vb).numpy()
    return float((va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt-a', required=True)
    ap.add_argument('--ckpt-b', required=True)
    ap.add_argument('--label-a', default='model A')
    ap.add_argument('--label-b', default='model B')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--num-videos', type=int, default=60)
    ap.add_argument('--max-positions', type=int, default=20000,
                    help='cap positions sampled per FPN level for CKA/PCA')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--use-ema', action='store_true',
                    help='load state_dict_ema from the checkpoint for model A '
                         '(SSL checkpoints store the EMA teacher there)')
    ap.add_argument('--random-a', action='store_true',
                    help='do not load any checkpoint for model A (random init)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print('device', args.device)
    model_a = load_model(cfg, args.ckpt_a, args.device,
                         use_ema=args.use_ema, random_init=args.random_a)
    model_b = load_model(cfg, args.ckpt_b, args.device)

    dataset = make_dataset(cfg['dataset_name'], False, [args.split], **cfg['dataset'])
    n = min(args.num_videos, len(dataset))
    print('using %d / %d videos' % (n, len(dataset)))

    n_levels = None
    feats_a, feats_b = [], []          # per level: list of (n_i, C)
    cos_all = []
    per_video_cka = []
    for vi in range(n):
        item = dataset[vi]
        fa, ma = encode(model_a, [item])
        fb, mb = encode(model_b, [item])
        if n_levels is None:
            n_levels = len(fa)
            feats_a = [[] for _ in range(n_levels)]
            feats_b = [[] for _ in range(n_levels)]
        cos_vid = []
        for l in range(n_levels):
            m = ma[l] & mb[l]
            if m.sum() < 4:
                continue
            A = fa[l][:, m].T.numpy()      # (n_pos, C)
            B = fb[l][:, m].T.numpy()
            feats_a[l].append(A)
            feats_b[l].append(B)
            an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
            cos_vid.append((an * bn).sum(1))
        if cos_vid:
            cos_vid = np.concatenate(cos_vid)
            cos_all.append(cos_vid)
            # per-video CKA averaged over levels (on a sample of positions)
            ckas = []
            for l in range(n_levels):
                if not feats_a[l]:
                    continue
                A = feats_a[l][-1]
                B = feats_b[l][-1]
                if len(A) > 2000:
                    idx = rng.choice(len(A), 2000, replace=False)
                    A, B = A[idx], B[idx]
                ckas.append(linear_cka(A, B))
            per_video_cka.append(float(np.mean(ckas)))
        if (vi + 1) % 20 == 0:
            print('  %d/%d' % (vi + 1, n))

    cos_all = np.concatenate(cos_all)
    per_video_cka = np.array(per_video_cka)

    # per-level CKA + PCA
    cka_levels = []
    pca_data = []
    for l in range(n_levels):
        if not feats_a[l]:
            cka_levels.append(float('nan'))
            continue
        A = np.concatenate(feats_a[l], 0)
        B = np.concatenate(feats_b[l], 0)
        if len(A) > args.max_positions:
            idx = rng.choice(len(A), args.max_positions, replace=False)
            A, B = A[idx], B[idx]
        cka_levels.append(linear_cka(A, B))
        pca_data.append((A, B, l))

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(TEXT2)
        ax.tick_params(colors=TEXT2)
        ax.grid(True, alpha=0.15)

    # (1) CKA per level
    xs = np.arange(n_levels)
    axes[0].bar(xs, cka_levels, color=C_A)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(['L%d' % l for l in range(n_levels)])
    axes[0].set_ylim(0, 1.02)
    axes[0].axhline(1.0, color=TEXT2, lw=0.8, ls='--')
    axes[0].set_ylabel('linear CKA', color=TEXT2)
    axes[0].set_title('Encoder feature similarity per FPN level', color=TEXT)

    # (2) cosine histogram
    axes[1].hist(cos_all, bins=60, range=(-1, 1), color=C_A, alpha=0.9)
    axes[1].axvline(cos_all.mean(), color=C_B, lw=2,
                    label='mean %.3f' % cos_all.mean())
    axes[1].set_xlabel('per-timestep cosine(feat_A, feat_B)', color=TEXT2)
    axes[1].set_ylabel('count', color=TEXT2)
    axes[1].set_title('Per-position feature cosine', color=TEXT)
    axes[1].legend(frameon=False, labelcolor=TEXT)

    # (3) PCA of the finest level (level 0) from both models
    if pca_data:
        A, B, l = pca_data[0]
        P = pca2(np.concatenate([A, B], 0))
        axes[2].scatter(P[:len(A), 0], P[:len(A), 1], s=4, alpha=0.35,
                        color=C_A, label=args.label_a)
        axes[2].scatter(P[len(A):, 0], P[len(A):, 1], s=4, alpha=0.35,
                        color=C_B, label=args.label_b)
        axes[2].set_xlabel('PC1', color=TEXT2)
        axes[2].set_ylabel('PC2', color=TEXT2)
        axes[2].set_title('PCA of level-%d features' % l, color=TEXT)
        axes[2].legend(frameon=False, labelcolor=TEXT, markerscale=3)

    fig.suptitle('Encoder representation: %s  vs  %s' % (args.label_a, args.label_b),
                 color=TEXT, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(args.out_dir, 'encoder_similarity.png')
    fig.savefig(out, facecolor=fig.get_facecolor())
    print('saved', out)

    # ---- numeric summary ----
    summary = {
        'config': args.config,
        'ckpt_a': args.ckpt_a, 'label_a': args.label_a,
        'ckpt_b': args.ckpt_b, 'label_b': args.label_b,
        'num_videos': int(n),
        'cka_per_level': cka_levels,
        'cka_mean_over_levels': float(np.nanmean(cka_levels)),
        'cosine_mean': float(cos_all.mean()),
        'cosine_median': float(np.median(cos_all)),
        'cosine_p10': float(np.percentile(cos_all, 10)),
        'cosine_p90': float(np.percentile(cos_all, 90)),
        'per_video_cka_mean': float(per_video_cka.mean()),
        'per_video_cka_std': float(per_video_cka.std()),
        'weight_cosine_backbone': weight_cosine(model_a, model_b, 'backbone'),
        'weight_cosine_neck': weight_cosine(model_a, model_b, 'neck'),
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
