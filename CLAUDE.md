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

- **Dataset**: EPIC-KITCHENS-100, verb-only track. 272 training videos, 138
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
point. Currently running on Alice: job 5077230,
`train_verb_from_ssl.slurm` → `ckpt/epic_slowfast_verb_from_ssl_neco/`.
Confirmed at start: 207 encoder keys matched, 22 head keys left random,
0 unexpected (matches the same key split reported for pipeline 2's merge).

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
| SSL *pre*-train → supervised (pipeline 3) | — | — | — | — | — | *pending (job 5077230)* |

SSL training signals:

| Run | Encoder init | Final val NeCo loss | Final top-1 agreement |
|---|---|---|---|
| SSL post-train (pipeline 2) | supervised epoch_021 | 5.59 | 0.832 |
| SSL pre-train (pipeline 3, stage A) | random | 2.20 | 0.769 |
| SSL post-train, V-JEPA2 | supervised epoch_021 | 5.95 | 0.835 |

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

Pipeline 3 (pretrain-then-finetune) is specifically designed to test whether
that head-mismatch explanation holds: if it does, letting the heads
train jointly with an SSL-initialized encoder (rather than staying frozen)
should recover — or exceed — the 23.07 baseline, since nothing here is frozen
out of the loop.

## Current status

- Job **5077230** (`train_verb_from_ssl`, `gpu-mig-40g`) running on Alice
  since 2026-09-21, ~7h budget. Confirmed healthy at epoch 0 (loss curve
  looks like normal supervised training, no key-mismatch errors).
- All code (`train.py --init-encoder`, `train_ssl.py` refactor,
  `train_verb_from_ssl.slurm`, `tools/plot_loss_curves.py`) is pushed to
  `origin/main` and present on both this machine and Alice.

## Next steps

1. **Once job 5077230 finishes**: run `eval.py` on
   `ckpt/epic_slowfast_verb_from_ssl_neco/` for the mAP row in the Results
   table above, and:
   ```
   python actionformer/tools/plot_loss_curves.py \
     --run "Supervised (random init)"=actionformer/ckpt/epic_slowfast_verb_reproduce/logs \
     --run "Supervised (from NeCo SSL init)"=actionformer/ckpt/epic_slowfast_verb_from_ssl_neco/logs \
     --tag train/final_loss --val-tag validation/mAP \
     --out actionformer/figs/supervised_loss_comparison.png
   ```
   to check the question posed in `current_lit.md`: does the SSL-seeded run
   start lower/same and converge to the same point, or does it actually beat
   23.07?
2. **Write up pipeline 3** as a new section of `report.md` (or a v2),
   parallel to the existing pipeline-2 write-up, with the same
   paper-comparison framing.
3. If pipeline 3 *does* beat the baseline, it directly supports the
   head-mismatch explanation from §5.1 — worth a short joint-finetune
   ablation on pipeline 2 (unfreeze the heads for a few epochs after SSL
   post-training) to test the same hypothesis from the other direction.
4. Open thread from `current_lit.md`: only one SSL protocol variant
   (temporal crop augmentation) has been tried; alternative augmentation
   styles (frame-rate variation, combined spatial+temporal crops) are noted
   but not yet run.
