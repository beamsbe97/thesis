# Self-Supervised Pre/Post-Training for Temporal Action Localization with ActionFormer

*Last updated: 2026-09-25 (seed replicates, frame-rate augmentation, VideoMAE-L SSL runs added). Paths below are relative to `actionformer/`.*

## 1. Overview

This report documents a set of experiments on the **EPIC-KITCHENS-100 verb** task
(temporal action localization, TAL) using **ActionFormer** (Zhang et al., ECCV 2022).
The research question is whether a *NeCo-style self-supervised (SSL)* stage can
improve downstream TAL performance, measured as mAP at different temporal-IoU (tIoU)
thresholds on the validation split. Two protocols are compared:

- SSL **post-training** on top of a converged supervised model (pipeline 2, §2.3), and
- SSL **pretraining** on an untrained encoder followed by standard supervised training
  (pipeline 3, §2.5) -- the ordering used in the original NeCo/DINO-style literature.

A follow-up ablation (§2.6) fine-tunes the post-trained model with its heads unfrozen
to test *why* post-training hurts.

**Summary of findings.**

| Protocol | SlowFast | VideoMAE-L | V-JEPA2 |
|---|---|---|---|
| Supervised baseline | 23.38 ± 0.75 (n=3) | 29.60 | 17.49 |
| SSL post-training, frozen heads (pipeline 2) | 22.55 | 29.24 | -- |
| ... + joint encoder/head fine-tune (ablation) | 22.45 (control: 22.97) | -- | -- |
| SSL pretraining → supervised (pipeline 3) | 23.65 ± 0.08 (n=3) | 29.38 | 17.40 |
| ... + frame-rate augmentation in SSL | **24.45** (n=1) | -- | -- |

(Avg mAP, EPIC-100 verb validation; mean ± sample sd over seeds where n > 1.)

1. **SSL post-training hurts**, on both feature tracks tested (SlowFast −0.52,
   VideoMAE-L −0.36), and re-training the heads jointly does not recover it: the
   damage is in the encoder.
2. **SSL pretraining does not raise mean mAP.** The single-seed +0.49 pp on SlowFast
   reported earlier shrinks to +0.27 pp over three seeds each, well inside seed
   noise (baseline sd 0.75), and there is no gain on V-JEPA2 or VideoMAE-L.
   It does make SlowFast training markedly **more stable across seeds**
   (sd 0.08 vs 0.75).
3. **Frame-rate augmentation in the SSL stage** gives the best SlowFast result so far
   (24.45, +0.80 over pipeline 3 without it), but on a single seed.
4. **Features dominate**: EPIC-finetuned VideoMAE-L features are +6.2 pp over
   SlowFast -- an order of magnitude more than any SSL effect.

Experiments were run on the ALICE cluster (Leiden University). The project mirrors the
official ActionFormer codebase with an added self-supervised training stage.

## 2. Method and Experimental Setup

### 2.1 Task, data and features

- **Dataset**: EPIC-KITCHENS-100, verb-only track, trained on `training` split
  (495 videos), evaluated on `validation` split (138 videos).
- **Features**:
  - **SlowFast**: 2304-dim features extracted with a SlowFast network pre-trained on
    EPIC-KITCHENS (official features, 30 fps, 32-frame window, feature stride 16;
    density ~1.875 vectors/sec).
  - **V-JEPA2**: 1024-dim features extracted with a frozen V-JEPA2 (ViT-L fpc32)
    backbone, density matched to SlowFast (~1.875 vectors/sec).
  - **VideoMAE-L**: 1024-dim features from a VideoMAE-L-16x4x1 backbone
    **finetuned on EPIC-KITCHENS verbs**, as released by OpenTAD (16-frame clips,
    stride 8 at 30 fps, i.e. ~3.75 vectors/sec, 2x SlowFast). Config
    `epic_videomae_verb.yaml`: identical recipe to SlowFast except `input_dim`,
    `feat_stride: 8`, `num_frames: 16` and `max_seq_len: 4608` (same temporal span).
- **Evaluation metric**: mAP at tIoU in {0.1, 0.2, 0.3, 0.4, 0.5} and the average,
  following the ActionFormer paper.

All three pipelines share the same encoder (`ConvTransformerBackbone` + identity FPN
neck, 207 parameter tensors), so encoder weights transfer between them; only the task
heads differ (SSL projection head vs. supervised cls/reg/center heads, 22 tensors).

### 2.2 Supervised baseline (pipeline 1)

Standard ActionFormer training with focal classification loss + DIoU regression loss,
center sampling, and soft-NMS inference; 16 epochs + 5 warm-up, AdamW, lr 1e-4,
batch size 2. Configs `epic_slowfast_verb.yaml` / `epic_vjepa2_verb.yaml`, final
checkpoint `epoch_021.pth.tar`.

### 2.3 Self-supervised post-training (NeCo-style, pipeline 2)

The encoder is warm-started from the supervised `epoch_021` checkpoint and then
post-trained self-supervised for 12 effective epochs (10 + 2 warm-up, batch size 4)
using a NeCo-style objective: a teacher/student EMA framework with temporal ROI
alignment over two overlapping temporal crops of each video and a SoftSort-based
neighbour-ranking loss, on the same video features (no labels).

Conjecture tested: SSL post-training of the encoder improves the representation for
downstream TAL, and thus raises mAP.

### 2.4 Downstream evaluation of the post-trained model

The SSL-trained encoder (EMA weights) is merged with the **frozen supervised**
cls/reg/center heads from `epoch_021.pth.tar` and evaluated with the standard
ActionFormer inference pipeline (`eval_ssl.py --head-from ...`; 207 encoder keys from
the SSL checkpoint, 22 head keys from the supervised checkpoint).

### 2.5 SSL pretraining then supervised training (pipeline 3)

1. **Stage A -- SSL pretraining on an untrained encoder.** `train_ssl.py` from a
   **random encoder init**, 12 effective epochs, same NeCo objective as §2.3.
2. **Stage B -- full supervised training from the SSL-pretrained encoder.**
   `train.py --init-encoder` loads only the backbone+neck weights from the SSL
   checkpoint's **EMA (teacher)** state, leaves the heads at random init, then trains
   the **entire model** with the unmodified supervised recipe (§2.2). The only
   difference from the baseline is the encoder's starting point.

Run for SlowFast (3 seeds, §2.8), V-JEPA2 and VideoMAE-L.

### 2.6 Joint fine-tune ablation on pipeline 2

Tests the head-mismatch explanation for pipeline 2's drop (§5.1) directly. Starting
from the *exact* pipeline-2 model (SSL post-trained EMA encoder + supervised
`epoch_021` EMA heads; `train.py --init-model <supervised> --init-encoder <ssl>`),
encoder **and** heads are fine-tuned jointly with the supervised loss, using a short
schedule suited to an already-converged model
(`configs/epic_slowfast_verb_joint_ft.yaml`: 1 warm-up + 5 cosine epochs, lr 2e-5 =
1/5 of the base lr).

A **control** runs the identical schedule from the unmodified supervised baseline
(no SSL), so the effect of extra supervised epochs is separated from the effect of
the SSL encoder.

### 2.7 Feature-level frame-rate augmentation

A variant of the SSL view sampling (§2.3) that also changes the *playback rate* of
each view. After the two temporal crops are sampled, each training view gets an
independent rate r ~ log-uniform[0.5, 2]. The crop [s, e) is resampled by linear
interpolation at raw feature positions s, s + r, s + 2r, ...: r > 1 plays it faster
(fewer vectors), r < 1 slower (more, interpolated vectors). The shared ROI is mapped
into each view with the view's stride × r, so both views are still aligned on the
same raw moments and the NeCo target is unchanged; the model must now match
neighbourhoods across different apparent action durations. Validation views always
use r = 1, so validation losses stay comparable with the other runs.

This operates on the pre-extracted feature sequence -- no re-extraction. Each SlowFast
vector still encodes ~1 s of normal-speed motion, so this is sequence-rate
(action-duration) augmentation rather than true motion-speed augmentation.
Implementation: `ssl.rate_range` in the SSL config (`[1, 1]`, the default, is
bit-identical to the original code); geometry test in
`tools/test_rate_augmentation.py`. Run through pipeline 3 on SlowFast
(`configs/ssl_epic_slowfast_verb_rate.yaml`).

### 2.8 Seed replicates

The SlowFast baseline and SlowFast pipeline 3 were repeated with two additional seeds
(`--seed 1`, `--seed 2`, overriding the config's `init_rand_seed`; the original runs
use the default seed). For pipeline 3 **both** stages are re-seeded, so the spread
includes SSL-stage variance as well as supervised-stage variance.

## 3. Results

### 3.1 Main results (EPIC-100 verb, validation)

| Model | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP |
|---|---|---|---|---|---|---|
| Supervised baseline (SlowFast) | 26.40 | 25.15 | 23.73 | 21.70 | 18.35 | **23.07** |
| + SSL post-training, frozen heads (pipeline 2) | 26.14 | 24.70 | 23.20 | 20.93 | 17.79 | **22.55** |
| + SSL post-training + joint fine-tune (§2.6), final epoch | 26.15 | 24.78 | 23.14 | 21.21 | 17.83 | **22.45** |
| Control: baseline + same joint fine-tune, no SSL, final epoch | 26.41 | 25.16 | 23.55 | 21.52 | 18.22 | **22.97** |
| SSL pretrain → supervised (SlowFast, pipeline 3) | 26.72 | 25.63 | 24.10 | 22.37 | 18.99 | **23.56** |
| Supervised baseline (V-JEPA2) | 20.51 | 19.74 | 18.38 | 16.17 | 12.64 | **17.49** |
| SSL pretrain → supervised (V-JEPA2, pipeline 3) | 20.33 | 19.49 | 18.11 | 15.87 | 13.19 | **17.40** |
| SSL pretrain → supervised, rate-augmented SSL (SlowFast, §2.7) | 27.92 | 26.68 | 25.15 | 22.79 | 19.70 | **24.45** |
| Supervised baseline (VideoMAE-L, EPIC-finetuned) | 32.84 | 32.06 | 30.42 | 28.11 | 24.59 | **29.60** |
| + SSL post-training, frozen heads (VideoMAE-L, pipeline 2) | 32.65 | 31.80 | 29.96 | 27.83 | 23.97 | **29.24** |
| SSL pretrain → supervised (VideoMAE-L, pipeline 3) | 32.83 | 32.23 | 30.58 | 27.84 | 23.46 | **29.38** |

SlowFast rows without a seed are the default-seed runs; see §3.5 for the seed
replicates.

### 3.2 Joint fine-tune ablation, per epoch

Average mAP of the EMA model after each fine-tune epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| SSL post-trained + joint fine-tune | 22.48 | 22.47 | 22.52 | 22.61 | 22.60 | 22.45 |
| Control (no SSL) | 23.02 | 22.98 | 23.02 | 23.01 | 22.94 | 22.97 |

The gap to the control (~0.5 pp) is flat across all six epochs: there is no upward
trend that a longer fine-tune at this learning rate would plausibly extend.

### 3.2.1 V-JEPA2 pipeline 3 vs. baseline, per epoch

| Epoch | 1 | 5 | 10 | 15 | 20 | 21 |
|---|---|---|---|---|---|---|
| V-JEPA2 baseline (random init) | 0.07 | 5.56 | 12.53 | 16.03 | 17.40 | 17.49 |
| V-JEPA2, SSL-pretrained encoder | 0.06 | 5.54 | 12.84 | 16.24 | 17.35 | 17.40 |

The SSL-initialized run is marginally ahead mid-training (+0.2–0.3 pp at epochs
10/15) but converges to the same value; the two curves are indistinguishable at
the end.

### 3.3 SSL training signals

| Run | Encoder init | Final val NeCo loss | Final top-1 agreement |
|---|---|---|---|
| SlowFast, post-training (pipeline 2) | supervised epoch_021 | 5.59 | 0.832 |
| SlowFast, pretraining (pipeline 3, stage A) | random | 2.20 | 0.769 |
| V-JEPA2, post-training | supervised epoch_021 | 5.95 | 0.835 |
| V-JEPA2, pretraining (pipeline 3, stage A) | random | 1.77 | 0.839 |
| SlowFast, pretraining, seed 1 | random | 2.16 | 0.786 |
| SlowFast, pretraining, seed 2 | random | 2.21 | 0.780 |
| SlowFast, pretraining + rate augmentation | random | 2.29 | 0.769 |
| VideoMAE-L, post-training (pipeline 2) | supervised epoch_021 | 5.16 | 0.849 |
| VideoMAE-L, pretraining (pipeline 3, stage A) | random | 2.41 | 0.849 |

NeCo losses are not comparable *across* encoder inits: a supervised encoder starts
with a very different feature geometry (the warm SlowFast run starts at ~17 and ends
at 5.6), so a lower loss for the random-init runs does not mean a better
representation. Within an init they are consistent: the three random-init SlowFast
seeds end at 2.16–2.21, and the rate-augmented run (validated at rate 1) at 2.29 --
the harder training views barely change how well the objective fits.

### 3.4 Training-loss curves

![NeCo SSL pretraining loss](actionformer/figs/ssl_pretrain_loss.jpg)

*Pipeline 3 stage A (SlowFast): train and validation NeCo loss from random init.*

![Supervised training loss, random vs. SSL init](actionformer/figs/supervised_loss_comparison.jpg)

*Supervised training loss, baseline (random encoder) vs. pipeline 3 stage B
(SSL-initialized encoder).*

The two supervised loss curves are effectively indistinguishable: same starting value,
same noisy trajectory, same final training loss (~0.35). SSL pretraining does **not**
show up as a lower starting loss or faster convergence; whatever it changes shows up
only in validation mAP and its seed-to-seed spread (§5.2).

### 3.5 Seed replicates (SlowFast)

| Run | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP |
|---|---|---|---|---|---|---|
| Baseline, default seed | 26.40 | 25.15 | 23.73 | 21.70 | 18.35 | 23.07 |
| Baseline, seed 1 | 26.15 | 25.01 | 23.66 | 21.32 | 18.06 | 22.84 |
| Baseline, seed 2 | 27.45 | 26.36 | 25.09 | 22.91 | 19.32 | 24.23 |
| **Baseline, mean ± sd** | | | | | | **23.38 ± 0.75** |
| Pipeline 3, default seed | 26.72 | 25.63 | 24.10 | 22.37 | 18.99 | 23.56 |
| Pipeline 3, seed 1 | 26.85 | 25.81 | 24.51 | 22.09 | 19.11 | 23.67 |
| Pipeline 3, seed 2 | 26.86 | 25.99 | 24.73 | 22.56 | 18.48 | 23.72 |
| **Pipeline 3, mean ± sd** | | | | | | **23.65 ± 0.08** |

Difference of means: **+0.27 pp** (Welch t ≈ 0.6, n = 3 each; not significant).
Seed-to-seed spread of the baseline (range 1.39 pp) is ~3× the originally reported
single-seed gain, while the three pipeline-3 runs lie within 0.16 pp of each other.

## 4. Comparison with ActionFormer (paper, Table 2)

ActionFormer, Table 2 (EPIC-Kitchens 100 validation set, verb task, SlowFast
features):

| Method | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP |
|---|---|---|---|---|---|---|
| BMN | 10.8 | 9.8 | 8.4 | 7.1 | 5.6 | 8.4 |
| G-TAD | 12.1 | 11.0 | 9.4 | 8.1 | 6.5 | 9.4 |
| ActionFormer (paper) | 26.6 | 25.4 | 24.2 | 22.3 | 19.1 | **23.5** |

### 4.1 Ours vs. paper

| Method (this work) | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP | Δ vs. paper |
|---|---|---|---|---|---|---|---|
| Supervised baseline (repro) | 26.40 | 25.15 | 23.73 | 21.70 | 18.35 | 23.07 | −0.43 |
| Baseline + SSL post-training | 26.14 | 24.70 | 23.20 | 20.93 | 17.79 | 22.55 | −0.95 |
| SSL pretrain → supervised (pipeline 3) | 26.72 | 25.63 | 24.10 | 22.37 | 18.99 | 23.56 | **+0.06** |
| ActionFormer paper | 26.6 | 25.4 | 24.2 | 22.3 | 19.1 | 23.5 | -- |

Notes on comparability:

- The reproduced baseline (23.07) is 0.43 pp below the paper (23.5), with closely
  matching per-threshold trends. With seed replicates the baseline is
  23.38 ± 0.75 (range 22.84–24.23), which **contains the paper's 23.5**: the
  "reproduction gap" is within seed noise, and "pipeline 3 beats the paper" (+0.06)
  is not a meaningful claim.
- The OpenTAD reimplementation reports **24.93** avg mAP for ActionFormer on
  EPIC-100 verb with their own EPIC-finetuned SlowFast features, above both the
  paper and our reproduction. Absolute numbers are therefore sensitive to the exact
  feature release; comparisons within this report all use the same features and
  recipe.
- The V-JEPA2 track has no counterpart in the paper.
- The VideoMAE-L baseline (29.60) is not comparable to the paper's SlowFast number:
  its backbone was finetuned on EPIC verbs by OpenTAD. It is included as a stronger
  feature track, not as a reproduction.

## 5. Analysis

### 5.1 SSL post-training does not improve TAL performance

NeCo-style post-training on top of the supervised model **decreases** average mAP
from 23.07 to 22.55 (−0.52 pp, −2.3% relative), at every tIoU threshold, with the
largest drops at 0.4/0.5 where boundary quality matters. The SSL objective itself
converges (agreement 0.565 → 0.832, loss ~17 → 5.6), so this is a transfer problem,
not an optimization failure.

Candidate explanations considered originally:

1. **The supervised model is already converged**, and the SSL objective pulls the
   encoder toward a different solution that is worse for localization.
2. **Head mismatch**: the heads are frozen while the encoder drifts, so the decoder
   cannot compensate.
3. **Short post-training budget** (12 effective epochs).
4. **SSL convergence ≠ downstream quality**, especially for boundary regression.

Explanation 2 was initially considered the load-bearing one. §5.3 tests it and
**rules it out** as the main cause.

### 5.2 SSL pretraining followed by supervised training: more stable, not better on average

With a single seed, pipeline 3 appeared to give **+0.49 pp** on SlowFast (23.56 vs
23.07), ahead at every tIoU threshold. The seed replicates (§3.5) change this:

- **Mean effect is within noise.** 23.65 ± 0.08 vs 23.38 ± 0.75, a +0.27 pp
  difference with t ≈ 0.6. The original baseline run happened to be on the low side
  of its own distribution.
- **Variance drops sharply.** The three SSL-initialized runs span 0.16 pp; the three
  baselines span 1.39 pp. Every SSL-initialized run is above the baseline *median*
  (23.07), but one baseline seed (24.23) beats all of them. With n = 3 per arm, the
  variance difference is suggestive rather than established, but it is the most
  consistent signal in the pipeline-3 data.
- **No gain on the other tracks.** V-JEPA2: 17.40 vs 17.49. VideoMAE-L: 29.38 vs
  29.60 (single seeds; both within the noise level measured on SlowFast).

The supervised *training* loss (§3.4) is indistinguishable between SSL-initialized
and random-init runs, consistent with SSL init affecting which solution training
settles in (a representation prior / stabilizer) rather than how fast the loss is
fit.

### 5.3 Joint fine-tune ablation: head mismatch is not the explanation

If pipeline 2 lost accuracy only because the frozen heads no longer fit the drifted
encoder, then fine-tuning heads and encoder together should recover the baseline.
It does not (§3.1, §3.2):

- The SSL post-trained model stays at **22.45–22.61** throughout the fine-tune,
  i.e. at the frozen-head level (22.55).
- The control, given the same extra supervised epochs, stays at **22.94–23.02**,
  i.e. at the baseline level. Extra fine-tuning by itself neither helps nor hurts.
- The ~0.5 pp gap persists at every epoch and every tIoU threshold.

So the damage from SSL post-training lives in the **encoder itself**: within this
fine-tune budget, supervised gradients do not move it back to an equally good
solution. This leaves explanations 1 and 4 of §5.1.

It also changes how pipeline 3 should be read: the difference between the protocols
is not "the heads were allowed to adapt" -- the ablation shows that alone is not
enough. It is that SSL is used as an **initialization**, followed by a *full*
supervised schedule (warm-up at the base learning rate) that can reshape the whole
encoder around the task. That is enough to avoid the post-training damage, but, per
§5.2, not enough to produce a reliable mean gain.

**Caveat.** The fine-tune was deliberately short and low-learning-rate (6 epochs at
lr 2e-5). A full-length, base-learning-rate schedule starting from the post-trained
encoder is untested; it would effectively become pipeline 3 with a different
initialization.

### 5.4 Feature track

- SlowFast features (2304-dim, EPIC pre-trained) are the stronger feature set;
  frozen V-JEPA2 features lag by ~5.6 pp (17.49 vs 23.07). V-JEPA2 is not
  EPIC-adapted, consistent with TAL features benefiting from in-domain fine-tuning.
- Feature densities are matched (~1.875 vectors/sec), so the gap is not a sampling
  artifact.
- **EPIC-finetuned VideoMAE-L features are far stronger**: 29.60 avg mAP, +6.53 pp
  over SlowFast and +12.1 pp over frozen V-JEPA2, ahead at every tIoU threshold
  with the unmodified SlowFast recipe (only feature-shape parameters changed).
  Feature quality dominates everything else in this report: the gap between
  feature sets is an order of magnitude larger than any SSL effect.
- **V-JEPA2 through pipeline 3: no gain.** 17.40 vs 17.49 (−0.09 pp). The only
  threshold where the SSL-initialized model is ahead is tIoU 0.5 (13.19 vs 12.64);
  the rest are 0.2–0.3 pp lower. The SSL stage itself converged normally (final val
  NeCo loss 1.77, agreement 0.839, §3.3).
- **VideoMAE-L through both SSL protocols** reproduces the SlowFast pattern on a much
  stronger baseline:
  - post-training (pipeline 2) **hurts**: 29.24 vs 29.60 (−0.36), lower at every
    tIoU threshold, as on SlowFast (−0.52);
  - pretraining (pipeline 3) gives **no gain**: 29.38 vs 29.60 (−0.22; ahead at
    tIoU 0.2/0.3, behind at 0.4/0.5).

### 5.5 Frame-rate augmentation

Adding the per-view rate augmentation (§2.7) to pipeline 3 gives **24.45 avg mAP** on
SlowFast -- the best SlowFast result in this report, and ahead of pipeline 3 without
it at every tIoU threshold (27.92 / 26.68 / 25.15 / 22.79 / 19.70 vs the 3-seed means
26.81 / 25.81 / 24.45 / 22.34 / 18.86).

- vs. pipeline 3 without rate augmentation: **+0.80** over its 3-seed mean. Given
  that pipeline 3's seeds span only 0.16 pp, this is well outside its observed spread.
- vs. the baseline: +1.07 over its mean, and above its best seed (24.23).
- The SSL objective itself is fit about as well as without the augmentation (val
  NeCo loss 2.29 vs 2.16–2.21, measured at rate 1), so the difference comes from
  *what* is learned, not how well the objective converges.

This is a **single seed**, and it is the first result in this study where an SSL
variant plausibly beats the baseline distribution rather than a single baseline
run. It is the obvious candidate for seed replication, and for testing on VideoMAE-L.


### 5.6 Conclusions

- **SSL post-training on a converged model hurts** on both SlowFast (−0.52) and
  VideoMAE-L (−0.36), and **jointly re-training the heads does not rescue it**
  (22.45 vs a 22.97 control). The post-trained encoder is itself worse for
  localization. This is the most robust finding of the study: it holds across
  feature tracks and has a clean control.
- **SSL pretraining with the original temporal-crop views does not improve mean
  mAP** on any feature track (SlowFast +0.27 over 3 seeds, within noise; V-JEPA2
  −0.09; VideoMAE-L −0.22). The earlier single-seed "+0.49 pp, beats the paper"
  result was seed noise in the baseline. On SlowFast it does make training much more
  stable across seeds (sd 0.08 vs 0.75).
- **Frame-rate augmentation changes the picture**, at least on one seed: 24.45 on
  SlowFast, +0.8 above pipeline 3 without it. Whether SSL pretraining helps may
  depend on *which invariance* the views teach -- matching crops of the same speed
  gives no mean gain; matching across playback rates might.
- **Feature quality dominates**: VideoMAE-L features give +6.2 pp over SlowFast,
  an order of magnitude more than any SSL effect.

**Limitations.** Seed replicates exist only for the SlowFast baseline and pipeline 3
(n = 3 each); every other row, including the rate-augmented result, is a single
seed. The SlowFast baseline's seed spread (sd 0.75) means single-seed differences
below ~1.5 pp should not be interpreted.

## 6. Reproducibility Notes

- All results: EPIC-100 **verb** task, validation split.
- **Supervised baselines**: `train_verb_4621944.out` (V-JEPA2, 17.49) and
  `logs/eval_1657178.out` (SlowFast, 23.07); checkpoint
  `ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar`.
- **Pipeline 2 SSL runs**: `ssl_neco_slowfast_warm_4988444.out`,
  `ssl_neco_vjepa2_4917434.out`. Downstream eval: `eval_ssl.py
  configs/epic_slowfast_verb.yaml ckpt_ssl/ssl_epic_slowfast_verb_neco_warm/epoch_012.pth.tar
  --head-from ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar` (job 4991145,
  48 min; 207 encoder + 22 head keys, EMA on both sides).
- **Pipeline 3, SlowFast**: stage A job 4796515 → `ckpt_ssl/ssl_epic_slowfast_verb_neco/`;
  stage B `train_verb_from_ssl.slurm` (job 5077230, 1h06m; 207 encoder keys matched,
  22 head keys random, 0 unexpected) → `ckpt/epic_slowfast_verb_from_ssl_neco/epoch_021.pth.tar`.
- **Joint fine-tune ablation**: `train_verb_ft_joint_ssl_warm.slurm` (job 5093320) and
  control `train_verb_ft_joint_ctrl.slurm` (job 5093321), 21 min each;
  `ckpt/epic_slowfast_verb_joint_ft_{ssl_warm,ctrl}/`.
- **Pipeline 3, V-JEPA2**: stage A `train_ssl_vjepa2_cold.slurm` (job 5093340, 7m45s)
  → `ckpt_ssl/ssl_epic_vjepa2_verb_neco_vjepa2_cold/`; stage B
  `train_verb_vjepa2_from_ssl.slurm` (job 5093405, 1h04; 207 encoder keys matched,
  22 missing, 0 unexpected) → `ckpt/epic_vjepa2_verb_from_ssl_neco/epoch_021.pth.tar`.
  A first attempt (job 5093341) was cancelled at epoch 15 by mistake and rerun from
  scratch; its partial output is in `ckpt/_cancelled/`.
- **VideoMAE-L baseline**: features in
  `data/epic_kitchens/features_videomae_verb/` (OpenTAD release, 700 videos);
  `train_verb_videomae.slurm` (job 5093406, 1h16) →
  `ckpt/epic_videomae_verb_reproduce/epoch_021.pth.tar`. A first attempt (job
  5093366) was likewise cancelled at epoch 5 and rerun.
- **Seed replicates** (`--seed 1`, `--seed 2`): baselines `train_verb_seed.slurm`
  (jobs 5094858, 5094861) → `ckpt/epic_slowfast_verb_reproduce_seed{1,2}/`;
  pipeline 3 stage A `train_ssl_seed.slurm` (5094859, 5094862) →
  `ckpt_ssl/ssl_epic_slowfast_verb_neco_seed{1,2}/`, stage B
  `train_verb_from_ssl_seed.slurm` (5094860, 5094863) →
  `ckpt/epic_slowfast_verb_from_ssl_neco_seed{1,2}/`. Submit with
  `sbatch --export=ALL,SEED=<n> <script>`.
- **Frame-rate augmentation**: stage A `train_ssl_rate.slurm` (job 5094852) →
  `ckpt_ssl/ssl_epic_slowfast_verb_rate_neco/`; stage B `train_verb_from_ssl_rate.slurm`
  (5094853, 1h06) → `ckpt/epic_slowfast_verb_from_ssl_neco_rate/epoch_021.pth.tar`.
- **VideoMAE-L SSL**: pipeline 3 `train_ssl_videomae_cold.slurm` (5094854) →
  `train_verb_videomae_from_ssl.slurm` (5094855, 1h16) →
  `ckpt/epic_videomae_verb_from_ssl_neco/`; pipeline 2 `train_ssl_videomae_warm.slurm`
  (5094856) → `eval_ssl_videomae_warm.slurm` (5094857; 207 encoder + 22 head keys).
- All encoder inits verified at load: 207 encoder keys matched, 0 unexpected.
- **Figures**: `figs/ssl_pretrain_loss.{png,jpg}`,
  `figs/supervised_loss_comparison.{png,jpg}`, produced by `tools/plot_loss_curves.py`.
