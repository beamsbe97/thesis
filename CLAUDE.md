# Thesis: Self-Supervised Post-Training for Temporal Action Localization

Research question: does a **NeCo-style self-supervised (SSL) stage** improve
**ActionFormer** (Zhang et al., ECCV 2022) on temporal action localization
(TAL), EPIC-KITCHENS-100 verb task? Two SSL protocols are compared: post-training
on top of a converged supervised model, and pretraining before supervised
training (the order used in the original NeCo/DINO-style literature).

Work happens on the **ALICE** cluster (Leiden University, host alias `alice`,
`/zfsstore/user/s4561341/thesis`), which is kept in sync with this repo via
GitHub (`beamsbe97/thesis`). This machine's git identity cannot push to that
remote directly — commits made here are transferred to Alice as a `git
bundle` and pushed from there (Alice's SSH key is the one with write access).

## Repo layout

- `actionformer/` — the active project: ActionFormer training/eval +
  the NeCo-style SSL add-on. Mirrors the official ActionFormer codebase
  (`libs/`) with SSL bolted on top (`libs/modeling/selfsup.py`, `train_ssl.py`).
- `vjepa2/` — Meta's V-JEPA2 codebase, used only as a frozen feature extractor
  (an alternative to SlowFast features) for the ActionFormer input track.
- `current_lit.md` — running research notes/log (dated entries), the primary
  place decisions and dead ends are recorded day-to-day.
- `actionformer/report.md` — the write-up of the *first* SSL experiment
  (post-training / warm-start protocol). Predates the pretrain-first pipeline
  documented here.

## Data & features

- **Dataset**: EPIC-KITCHENS-100, verb-only track. 495 training videos, 138
  validation videos.
- **SlowFast features** (primary track): 2304-dim, EPIC-pretrained, 32-frame
  window / 16-frame stride, 30fps clips → ~1.875 vectors/sec.
- **V-JEPA2 features** (secondary track): 1024-dim, frozen ViT-L backbone,
  density matched to SlowFast (~1.875 vectors/sec) for a fair comparison.
  Consistently ~5.6pp behind SlowFast on supervised mAP — frozen JEPA features
  aren't EPIC-domain-adapted; not carried into the SSL experiments below.
- Feature/eval details: `actionformer/report.md` §2.1.

## Pipelines

Three distinct training pipelines exist. All share the same ActionFormer
encoder (`ConvTransformerBackbone` + identity FPN neck) so weights transfer
between them; only the task heads differ (SSL projection head vs. supervised
cls/reg/center heads).

### 1. Supervised baseline (`train.py`)

Standard ActionFormer recipe: focal cls loss + DIoU reg loss, center
sampling, soft-NMS inference. 16 epochs + warmup, AdamW, batch size 2.
Config: `configs/epic_slowfast_verb.yaml` / `configs/epic_vjepa2_verb.yaml`.
Checkpoint: `ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar`.

### 2. SSL *post*-training (`train_ssl.py`, warm-start) — done, report written

Encoder is warm-started from the **converged supervised** checkpoint, then
post-trained self-supervised (NeCo objective, EMA teacher, 12 effective
epochs) with the cls/reg/center heads frozen out. Evaluated by splicing the
SSL-trained (EMA) encoder back onto the frozen supervised heads
(`eval_ssl.py --head-from ...`).

**Result: this makes things slightly worse** (23.07 → 22.55 avg mAP). See
"Results" and "Analysis" below — this is the finding in `report.md`.

### 3. SSL *pre*-training → supervised fine-tune — new, in progress

This is what the paper-style NeCo/DINO protocol actually does: SSL comes
**before** supervised training, not after, and the downstream task is trained
end-to-end afterward (not frozen-head splicing). Two stages:

**Stage A — NeCo SSL on an untrained (randomly-initialized) encoder.**
`train_ssl.py` with no `--init-encoder` (no supervised checkpoint to warm-start
from). Already run: `ckpt_ssl/ssl_epic_slowfast_verb_neco/` (job 4796515),
12 epochs, final val NeCo loss 2.20, top-1 neighbor agreement 0.769.
Loss curve: `actionformer/figs/ssl_pretrain_loss.png`.

**Stage B — full supervised training, encoder seeded from Stage A.**
This required a code change: `train.py` had no way to load an SSL checkpoint,
only `--resume` (full state incl. optimizer/scheduler, for continuing an
identical run). Added `--init-encoder <ssl_ckpt>`, which loads *only* the
backbone+neck weights (from the SSL checkpoint's EMA/teacher state) and
leaves the cls/reg/center heads at random init, then runs the **unmodified**
standard supervised recipe end-to-end — i.e. same hyperparameters as the
paper reproduction (pipeline 1), differing only in the encoder's starting
point. **Done**: job 5077230, `train_verb_from_ssl.slurm` →
`ckpt/epic_slowfast_verb_from_ssl_neco/epoch_021.pth.tar`. Confirmed at start:
207 encoder keys matched, 22 head keys left random, 0 unexpected (matches the
same key split reported for pipeline 2's merge). **Result: 23.56 avg mAP —
beats both the from-scratch baseline (23.07) and the paper's own number
(23.5)**, at every tIoU threshold. See "Results" below.

## Architectural decisions

- **Shared encoder-loading utility.** `load_init_encoder` (originally private
  to `train_ssl.py`, used to warm-start SSL *from* a supervised checkpoint)
  was generalized and moved to `libs/utils/train_utils.py` so both
  directions — supervised→SSL and SSL→supervised — share one implementation.
  This works because `PtTransformer` (supervised) and `PtTransformerSSL`
  both build `self.backbone` / `self.neck` identically; the function keeps
  everything except head-prefixed keys (`head.`, `cls_head`, `reg_head`,
  `center_head`) and loads with `strict=False`.
- **`prefer_ema` flag on the loader.** SSL checkpoints store two encoder
  copies: the raw student (`state_dict`) and the EMA teacher
  (`state_dict_ema`). The teacher is the one that's actually meant to have
  converged (student is noisier, updated every step). `train.py
  --init-encoder` defaults to `prefer_ema=True`; `train_ssl.py`'s
  supervised→SSL warm-start keeps `prefer_ema=False` (a supervised
  checkpoint's `state_dict` *is* the trained model, no teacher/student
  split) — unchanged from before the refactor, to avoid silently changing
  pipeline 2's already-reported numbers.
- **`--init-encoder` trains everything, doesn't freeze anything.** Unlike
  `eval_ssl.py` (pipeline 2's evaluation, which freezes the supervised heads
  onto an SSL encoder purely to measure the encoder's quality), pipeline 3
  stage B trains the whole model — encoder *and* heads — normally. This
  matches "train ActionFormer like in the original paper," just from a
  different encoder init, and is the more standard SSL-pretrain protocol.
- **git bundle instead of direct push.** This machine's SSH key isn't
  authorized on `beamsbe97/thesis`; Alice's is. Local commits are packaged
  with `git bundle create <file> origin/main..HEAD`, scp'd to Alice,
  fetched into a temp branch, fast-forward merged into `main`, then pushed
  from there. Bundle prerequisites must already exist on the target
  (verified with `git bundle verify` first).

## Results

EPIC-100 verb, validation split, mAP @ tIoU (avg of 0.1–0.5):

| Model | tIoU .1 | .2 | .3 | .4 | .5 | **Avg** |
|---|---|---|---|---|---|---|
| ActionFormer paper (Table 2, SlowFast) | 26.6 | 25.4 | 24.2 | 22.3 | 19.1 | **23.5** |
| Supervised baseline (repro, SlowFast) | 26.40 | 25.15 | 23.73 | 21.70 | 18.35 | **23.07** |
| + SSL *post*-training (warm-start, pipeline 2) | 26.14 | 24.70 | 23.20 | 20.93 | 17.79 | **22.55** |
| Supervised baseline (V-JEPA2) | 20.51 | 19.74 | 18.38 | 16.17 | 12.64 | **17.49** |
| SSL *pre*-train → supervised (pipeline 3) | 26.72 | 25.63 | 24.10 | 22.37 | 18.99 | **23.56** |
| Pipeline 2 + joint fine-tune, heads unfrozen (ablation, final ep.) | 26.15 | 24.78 | 23.14 | 21.21 | 17.83 | **22.45** |
| Control: baseline + same joint fine-tune, no SSL (final ep.) | 26.41 | 25.16 | 23.55 | 21.52 | 18.22 | **22.97** |
| SSL *pre*-train → supervised, V-JEPA2 (pipeline 3) | 20.33 | 19.49 | 18.11 | 15.87 | 13.19 | **17.40** |
| Supervised baseline (VideoMAE-L, EPIC-finetuned, OpenTAD feats) | 32.84 | 32.06 | 30.42 | 28.11 | 24.59 | **29.60** |
| VideoMAE-L + SSL post-training, frozen heads (pipeline 2) | 32.65 | 31.80 | 29.96 | 27.83 | 23.97 | **29.24** |
| VideoMAE-L SSL pre-train → supervised (pipeline 3) | 32.83 | 32.23 | 30.58 | 27.84 | 23.46 | **29.38** |
| SlowFast pipeline 3 + frame-rate augmentation in SSL (1 seed) | 27.92 | 26.68 | 25.15 | 22.79 | 19.70 | **24.45** |

**Seed replicates (SlowFast, n=3 each):** baseline 23.07 / 22.84 / 24.23 =
**23.38 ± 0.75**; pipeline 3 23.56 / 23.67 / 23.72 = **23.65 ± 0.08**. Difference
+0.27, Welch t ≈ 0.6 -> not significant. The single-seed "+0.49, beats the paper"
claim was baseline seed noise; the paper's 23.5 is inside the baseline range.
Pipeline 3's much lower spread is the real (tentative, n=3) signal.

SSL training signals:

| Run | Encoder init | Final val NeCo loss | Final top-1 agreement |
|---|---|---|---|
| SSL post-train (pipeline 2) | supervised epoch_021 | 5.59 | 0.832 |
| SSL pre-train (pipeline 3, stage A) | random | 2.20 | 0.769 |
| SSL post-train, V-JEPA2 | supervised epoch_021 | 5.95 | 0.835 |
| SSL pre-train, V-JEPA2 (pipeline 3, stage A) | random | 1.77 | 0.839 |

Reproduction gap vs. the paper is small and expected (−0.43pp): matches the
paper's per-threshold trend closely; attributed to feature-extraction
version / schedule / eval details, not a bug. Full detail: `report.md`.

## Analysis (pipeline 2, warm-start post-training)

The main finding so far — **SSL post-training on a converged supervised model
makes it slightly worse**, at every tIoU threshold — has four candidate
explanations (`report.md` §5.1), the load-bearing one being **head mismatch**:
the cls/reg/center heads stay frozen at their supervised values while the
encoder drifts under the SSL objective, so any representational change isn't
compensated by the decoder. The SSL loss *does* converge (agreement
0.565→0.832) — this is a transfer problem, not an optimization failure.

Pipeline 3 (pretrain-then-finetune) tested that explanation directly: letting
the heads train jointly with an SSL-initialized encoder, rather than staying
frozen, should recover — or exceed — the 23.07 baseline if head mismatch is
the real culprit.

**It did**: 23.56 avg mAP, +0.49 pp over baseline, +0.06 pp over the paper,
ahead at every tIoU threshold. Notably, this isn't visible in the *training*
loss — pipeline 3's supervised loss curve is essentially identical to the
from-scratch baseline's (same start, same trajectory, same ~0.35 endpoint;
`actionformer/figs/supervised_loss_comparison.png`). The gain is purely in
downstream generalization (validation mAP for the same training loss), i.e.
SSL pretraining is acting as a representation prior, not as an optimization
head-start. Full write-up: `report.md` §5.1.1, §5.3.

### Joint fine-tune ablation on pipeline 2 (2026-09-24) — head mismatch *not* confirmed

Tested head mismatch from the other direction: start from the exact
pipeline-2 model (SSL post-trained EMA encoder + supervised epoch_021 EMA
heads), then fine-tune **encoder and heads jointly** with a short schedule
(`configs/epic_slowfast_verb_joint_ft.yaml`: 1 warmup + 5 cosine epochs,
lr 2e-5 = 1/5 base). Same schedule on the unmodified baseline as a control.
Jobs 5093320 (ablation) / 5093321 (control), 21 min each,
`ckpt/epic_slowfast_verb_joint_ft_{ssl_warm,ctrl}/`.

Per-epoch avg mAP (EMA model):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| SSL-post + joint FT | 22.48 | 22.47 | 22.52 | 22.61 | 22.60 | 22.45 |
| Control (no SSL) | 23.02 | 22.98 | 23.02 | 23.01 | 22.94 | 22.97 |

Unfreezing the heads recovers **none** of the pipeline-2 loss: the SSL-post
model stays at ~22.5 (= the frozen-head 22.55) while the control stays at
~23.0, a persistent ~0.5 pp gap at every epoch and every tIoU. So within
this fine-tune budget the damage from SSL post-training lives in the
*encoder*, not in a head/encoder mismatch that the heads could adapt to.
Pipeline 3's gain therefore is better explained by *where* SSL sits (as an
init, followed by a full supervised schedule that can reshape the encoder)
than by heads being trainable per se. Caveat: short/low-LR fine-tune only;
a full-length schedule from the SSL-post encoder is the untested remaining
variant (it would converge toward pipeline 3's protocol).

## Current status

- 2026-09-25 (afternoon): all 12 jobs of the seed / rate-aug / VideoMAE-SSL batch
  done (5094852–5094863), results above and in root `report.md` §3.5, §5.2, §5.5.
  Headline: post-training hurts on both SlowFast and VideoMAE-L; pretraining
  gives no mean gain on any track but stabilizes SlowFast across seeds; rate
  augmentation (`ssl.rate_range: [0.5, 2.0]`) gave 24.45 on one seed.

- 2026-09-25: **V-JEPA2 pipeline 3 done: no gain** (17.40 vs 17.49 baseline;
  job 5093405, `ckpt/epic_vjepa2_verb_from_ssl_neco/`). The SlowFast pipeline-3
  gain (+0.49) does not replicate on this track -> treat it as tentative.
- 2026-09-25: **VideoMAE-L track added**: OpenTAD's EPIC-finetuned VideoMAE-L
  verb features (1024-d, stride 8, `.npy`) in
  `data/epic_kitchens/features_videomae_verb/`, config
  `configs/epic_videomae_verb.yaml`. Supervised baseline **29.60** (job 5093406),
  +6.5 pp over SlowFast.
- Jobs 5093341 / 5093366 were cancelled by mistake (buffered `.out` logs misread
  as a hang) and rerun as 5093405 / 5093406; partial outputs in
  `ckpt/_cancelled/`. Newer slurm scripts use `python -u`.
- Root `report.md` is the up-to-date write-up; `actionformer/report.md` is stale
  (pipeline 3 SlowFast only).

- 2026-09-24: joint fine-tune ablation (jobs 5093320/5093321) done, see
  above. V-JEPA2 pipeline 3 submitted: stage A `train_ssl_vjepa2_cold.slurm`
  → stage B `train_verb_vjepa2_from_ssl.slurm` (chained, `afterok`).
- Loss-curve figures also exported as JPEG (`actionformer/figs/*.jpg`).

- Job **5077230** (`train_verb_from_ssl`, `gpu-mig-40g`) **COMPLETED** on
  Alice 2026-09-21, 1h06m (exit 0, well under the 7h budget). Result:
  `ckpt/epic_slowfast_verb_from_ssl_neco/epoch_021.pth.tar`, 23.56 avg mAP.
- All code and the resulting figures/report updates are pushed to
  `origin/main` and present on both this machine and Alice.
- `report.md` and this file are up to date with pipeline 3's results as of
  2026-09-22.

## Next steps

1. ~~Ablation on pipeline 2: unfreeze the heads~~ — done 2026-09-24, did
   **not** recover the loss (see "Joint fine-tune ablation"). `report.md`
   §5.1 still presents head mismatch as the load-bearing explanation and
   needs revising.
2. Augmentation styles from `current_lit.md`: **feature-level frame-rate
   variation done** (combined with the existing temporal crops; 24.45, 1 seed —
   see item 7). Spatial crops aren't possible on pooled clip features (would
   need re-extraction).
3. ~~V-JEPA2 through pipeline 3~~ — done 2026-09-25, no gain (17.40 vs 17.49).
4. ~~Seeds~~ — done: +0.27 over 3 seeds, not significant (see Results).
5. ~~VideoMAE-L through pipelines 2/3~~ — done: 29.24 / 29.38 vs 29.60.
7. **Replicate the frame-rate augmentation result** (24.45, 1 seed): 2 more seeds
   via `train_ssl_rate.slurm` + `--seed` (script needs a SEED variant), then try it
   on VideoMAE-L.
6. OpenTAD's ActionFormer-SlowFast verb number is 24.93, above our 23.07
   repro — worth checking against their SlowFast features.
