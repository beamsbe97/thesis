# Self-Supervised Post-Training for Temporal Action Localization with ActionFormer

## 1. Overview

This report documents a set of experiments on the **EPIC-KITCHENS-100 verb** task
(temporal action localization, TAL) using **ActionFormer** (Zhang et al., ECCV 2022).
The main research question is whether a *NeCo-style self-supervised post-training*
step applied on top of a fully-supervised ActionFormer model can improve downstream
TAL performance, measured as mAP at different temporal-IoU (tIoU) thresholds on the
validation split.

Experiments were run on the ALICE cluster (Leiden University). The project mirrors the
official ActionFormer codebase with an added self-supervised training stage.

## 2. Method and Experimental Setup

### 2.1 Task, data and features

- **Dataset**: EPIC-KITCHENS-100, verb-only track, trained on `training` split
  (272 videos), evaluated on `validation` split (138 videos).
- **Features**:
  - **SlowFast**: 2304-dim features extracted with a SlowFast network pre-trained on
    EPIC-KITCHENS (official features, fps = 30, feature stride = 16; density
    ~1.875 vectors/sec).
  - **V-JEPA2**: 1024-dim features extracted with a frozen V-JEPA2 (ViT-L fpc32)
    backbone (density ~1.875 vectors/sec).
- **Evaluation metric**: mAP at tIoU in {0.1, 0.2, 0.3, 0.4, 0.5} and the average,
  following the ActionFormer paper.

### 2.2 Supervised baseline

Standard ActionFormer training with focal classification loss + DIoU regression loss,
center sampling, and soft-NMS inference. Baselines are reproduced with the configs
`epic_slowfast_verb.yaml` / `epic_vjepa2_verb.yaml` and the final checkpoint
`epoch_021.pth.tar`.

### 2.3 Self-supervised post-training (NeCo-style)

The supervised ActionFormer (encoder + supervised cls/reg heads) is first warm-started
from the supervised `epoch_021` checkpoint. Its encoder is then post-trained in a
self-supervised fashion for 12 effective epochs (10 epochs + 2 warm-up epochs,
batch size 4) using a NeCo-style objective: a teacher/student EMA framework with
temporal ROI alignment and a SoftSort-based ranking loss, run on the same video
features (no labels used).

The conjecture to be tested: self-supervised post-training of the encoder should
improve the feature representation for downstream TAL, and thus raise mAP.

### 2.4 Downstream evaluation of the post-trained model

The SSL-trained encoder (EMA weights) is merged with the **supervised** cls/reg heads
(from `epoch_021.pth.tar`) and evaluated on the validation split with the standard
ActionFormer inference pipeline. 207 encoder keys come from the SSL checkpoint,
22 head keys from the supervised checkpoint.

## 3. Results

### 3.1 Main results (EPIC-100 verb, validation)

| Model / checkpoint | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP |
|---|---|---|---|---|---|---|
| Supervised baseline (SlowFast features), epoch_021 | 26.40 | 25.15 | 23.73 | 21.70 | 18.35 | **23.07** |
| Supervised + SSL post-training (SlowFast features), SSL epoch_012 | 26.14 | 24.70 | 23.20 | 20.93 | 17.79 | **22.55** |
| Supervised baseline (V-JEPA2 features), epoch_021 | 20.51 | 19.74 | 18.38 | 16.17 | 12.64 | **17.49** |

### 3.2 SSL training signals

| Run | Warm start from | Final Val neco-loss | Final top-1 agreement |
|---|---|---|---|
| SlowFast + SSL (random init) | none (invalid) | 2.20 | 0.769 |
| SlowFast + SSL (warm) | epoch_021 | 5.59 | 0.832 |
| V-JEPA2 + SSL (warm) | epoch_021 | 5.95 | 0.835 |

The random-init run is excluded from downstream comparison: self-supervised
post-training must start from the supervised model, otherwise the improvement claim
does not hold.

## 4. Comparison with ActionFormer (paper, Table 2)

ActionFormer, Table 2 (results on the **EPIC-Kitchens 100 validation set**, verb task;
all methods use the **same SlowFast features**; mAP at tIoU 0.1-0.5 and average):

| Method | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP |
|---|---|---|---|---|---|---|
| BMN | 10.8 | 9.8 | 8.4 | 7.1 | 5.6 | 8.4 |
| G-TAD | 12.1 | 11.0 | 9.4 | 8.1 | 6.5 | 9.4 |
| ActionFormer (Ours, paper) | 26.6 | 25.4 | 24.2 | 22.3 | 19.1 | **23.5** |

### 4.1 Ours vs. paper

| Method (this work) | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP | Δ avg vs. paper 23.5 |
|---|---|---|---|---|---|---|---|
| Supervised baseline (repro) | 26.40 | 25.15 | 23.73 | 21.70 | 18.35 | 23.07 | **−0.43** |
| Baseline + SSL post-training | 26.14 | 24.70 | 23.20 | 20.93 | 17.79 | 22.55 | −0.95 |
| ActionFormer paper | 26.6 | 25.4 | 24.2 | 22.3 | 19.1 | 23.5 | -- |

Notes on comparability:

- Our reproduced supervised baseline (23.07) is 0.43 pp below the paper (23.5).
  This is a standard reproduction gap (feature-extraction version, training
  schedule/number of epochs, and evaluation details), and the per-threshold trends
  match the paper closely.
- The paper's Table 2 reports only verb results using SlowFast features; the V-JEPA2
  track is an additional experiment of this thesis and has no direct counterpart in
  the paper.

## 5. Analysis

### 5.1 SSL post-training does not improve TAL performance

Applying NeCo-style self-supervised post-training on top of the supervised ActionFormer
**decreases** average mAP slightly, from 23.07 to 22.55 (−0.52 pp, i.e. −2.3%
relative). At every individual tIoU threshold the SSL post-trained model scores lower
than its supervised start.

Possible reasons:

1. **The supervised model is already well-trained.** NeCo-style post-training is
   expected to help when features are under-trained (e.g., few-epoch supervised runs or
   scratch-trained encoders). Our baseline was trained to convergence (epoch_021 with
   the best/plateau validation mAP); the SSL objective then regularizes toward a
   different solution without recovering the task head, and can perturb the encoder.
2. **Head mismatch (no joint fine-tuning).** The cls/reg heads are frozen at their
   supervised values while the encoder moves within the SSL objective. Any drift of
   the encoder features is therefore not compensated by the decoder heads. A
   joint-optimization or a short supervised fine-tuning stage after post-training
   would likely close the gap.
3. **Short post-training budget.** 12 effective epochs may be insufficient for the
   SSL objective to converge to a stable representation that beats the supervised one.
4. **The SSL loss does converge** (agreement rises 0.565 → 0.832 for SlowFast, and the
   loss drops from ~17 → ~5.6), but convergence of the SSL metric does not imply
   better downstream localization, particularly for boundary regression
   (the largest drops are at higher tIoU, 0.4/0.5, where boundary quality matters).

### 5.2 Feature track

- SlowFast features (2304-dim, EPIC pre-trained) remain the stronger surveillance of
  the two; V-JEPA2 frozen features lag by ~5.6 pp (17.49 vs 23.07). JEPA features are
  not EPIC-supervised; the gap is consistent with the practice that TAL features
  benefit from in-domain fine-tuning.
- Feature densities are matched (~1.875 vectors/sec for both tracks), so the gap is
  not a sampling artifact.

### 5.3 Conclusions

The main hypothesis -- that NeCo-style self-supervised post-training improves
ActionFormer TAL performance -- is **not supported** by the current experiments
(22.55 vs 23.07 avg mAP, SlowFast). The positive SSL-training dynamics (loss
decrease, agreement increase) indicate the SSL objective learns, but it does not
transfer into better action localization under the current protocol.

## 6. Reproducibility Notes

- All results are on the EPIC-100 **verb** task, validation split.
- SlowFast SSL downstream eval: `eval_ssl.py configs/epic_slowfast_verb.yaml
  ckpt_ssl/ssl_epic_slowfast_verb_neco_warm/epoch_012.pth.tar
  --head-from ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar`
  (job 4991145, COMPLETED, exit 0, 48 min; merge verified: 207 encoder keys + 22 head
  keys; EMA weights on both sides).
- Supervised baselines: `train_verb_4621944.out` (V-JEPA2, 17.49 avg) and
  `logs/eval_1657178.out` (SlowFast, 23.07 avg).
- SSL training runs: `ssl_neco_slowfast_warm_4988444.out`,
  `ssl_neco_vjepa2_4917434.out`.