# Self-Supervised Pre/Post-Training for Temporal Action Localization with ActionFormer

## 1. Overview

This report documents a set of experiments on the **EPIC-KITCHENS-100 verb** task
(temporal action localization, TAL) using **ActionFormer** (Zhang et al., ECCV 2022).
The main research question is whether a *NeCo-style self-supervised (SSL)* stage
can improve downstream TAL performance, measured as mAP at different temporal-IoU
(tIoU) thresholds on the validation split. Two protocols are compared: SSL
**post-training** on top of a converged supervised model (§2.3), and SSL
**pretraining** on an untrained encoder followed by standard supervised training
(§2.5) -- the ordering used in the original NeCo/DINO-style literature.

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

### 2.5 SSL pretraining then supervised fine-tuning (pipeline 3)

Rather than post-training a converged model with the heads frozen, this pipeline
follows the more standard SSL-pretrain protocol:

1. **Stage A -- SSL pretraining on an untrained encoder.** `train_ssl.py` is run
   from a **random encoder init** (no supervised checkpoint to warm-start from) for
   12 effective epochs, same NeCo objective as §2.3. This is the "random init" row
   in Table 3.2 below -- excluded from the post-training comparison in §3.1/§4.1
   there, but used here as the starting point for stage B.
2. **Stage B -- full supervised training from the SSL-pretrained encoder.**
   `train.py` gained an `--init-encoder` flag: it loads only the backbone+neck
   weights from the SSL checkpoint's **EMA (teacher)** state, leaves the cls/reg/
   center heads at random init, then trains the **entire model** (encoder + heads)
   with the unmodified standard supervised recipe (§2.2) -- same hyperparameters as
   the paper reproduction, differing only in the encoder's starting point. Unlike
   §2.4, nothing is frozen: this directly tests whether the head-mismatch
   explanation in §5.1 holds.

## 3. Results

### 3.1 Main results (EPIC-100 verb, validation)

| Model / checkpoint | tIoU 0.1 | tIoU 0.2 | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | Avg mAP |
|---|---|---|---|---|---|---|
| Supervised baseline (SlowFast features), epoch_021 | 26.40 | 25.15 | 23.73 | 21.70 | 18.35 | **23.07** |
| Supervised + SSL post-training (SlowFast features), SSL epoch_012 | 26.14 | 24.70 | 23.20 | 20.93 | 17.79 | **22.55** |
| SSL pretrain -> supervised fine-tune (SlowFast, pipeline 3), epoch_021 | 26.72 | 25.63 | 24.10 | 22.37 | 18.99 | **23.56** |
| Supervised baseline (V-JEPA2 features), epoch_021 | 20.51 | 19.74 | 18.38 | 16.17 | 12.64 | **17.49** |

### 3.2 SSL training signals

| Run | Encoder init | Final Val neco-loss | Final top-1 agreement |
|---|---|---|---|
| SlowFast + SSL (random init, pipeline 3 stage A) | none | 2.20 | 0.769 |
| SlowFast + SSL (warm, pipeline 2) | epoch_021 | 5.59 | 0.832 |
| V-JEPA2 + SSL (warm) | epoch_021 | 5.95 | 0.835 |

The random-init SSL run is not itself evaluated for downstream TAL (its heads
are the SSL projection head, not cls/reg/center heads); it instead supplies the
encoder that pipeline 3 stage B fine-tunes on top of (§2.5, §3.3).

### 3.3 Pipeline 3 training-loss comparison

`tools/plot_loss_curves.py` overlays stage-B's supervised training loss against
the from-scratch baseline (`figs/supervised_loss_comparison.png`). The two loss
curves are effectively indistinguishable: both start near the same value, follow
the same noisy decreasing trajectory, and converge to the same final training loss
(~0.35). The SSL pretraining stage does **not** show up as a lower starting loss or
faster convergence in the *training* loss -- the benefit only shows up in the
downstream validation mAP (Table 3.1), not in how easy the supervised objective is
to fit. See §5.1.1 for interpretation.

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
| SSL pretrain -> supervised (pipeline 3) | 26.72 | 25.63 | 24.10 | 22.37 | 18.99 | 23.56 | **+0.06** |
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

### 5.1.1 SSL pretraining, with joint fine-tuning, *does* improve TAL performance

Pipeline 3 (§2.5) reverses the SSL/supervised ordering relative to §2.3-2.4 and
lets the heads train jointly with the SSL-initialized encoder instead of staying
frozen. Result: **23.56 avg mAP, beating both the from-scratch baseline (23.07,
+0.49 pp) and the paper's own reported number (23.5, +0.06 pp)**, and it does so
at *every* tIoU threshold, including 0.4/0.5 where §5.1's post-training run lost
the most ground. This is the outcome predicted by explanation 2 in §5.1: once the
heads are no longer frozen, the encoder drift induced by SSL is no longer a
liability -- it becomes a useful prior for the supervised heads to build on.

Notably, this is *not* visible in the supervised training loss curve (§3.3): the
SSL-pretrained run's training loss is statistically indistinguishable from the
random-init run's, same starting point, same trajectory, same convergence value.
The advantage is specifically a generalization effect (lower validation mAP for
the same training loss achieved), not an optimization-speed effect -- consistent
with SSL pretraining acting as a representation prior/regularizer rather than
simply giving supervised training a "head start" on the same loss landscape.

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

The picture is protocol-dependent, not a flat "SSL doesn't help":

- **SSL post-training on a converged, frozen-head model does not help**
  (22.55 vs 23.07 avg mAP, §5.1) -- the encoder drifts under the SSL objective
  and the frozen heads can't compensate.
- **SSL pretraining on an untrained encoder, followed by ordinary joint
  supervised training, does help** (23.56 vs 23.07 avg mAP, +0.49 pp, §5.1.1) --
  matching (and marginally beating) the paper's own number, at every tIoU
  threshold, with no change to the supervised recipe beyond the encoder's
  starting point.

Both results are consistent with the same underlying explanation: the SSL
objective **is** learning a useful signal (agreement rises, loss falls in both
cases), but only pays off downstream when the task heads are free to adapt to
whatever representation the encoder ends up with. Frozen heads turn a useful
representation shift into a mismatch; joint fine-tuning turns the same kind of
shift into a mild but real improvement.

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
- Pipeline 3, stage A (random-init SSL pretraining): job 4796515,
  `ckpt_ssl/ssl_epic_slowfast_verb_neco/` (12 epochs, final val NeCo loss 2.20,
  agreement 0.769). Loss curve: `figs/ssl_pretrain_loss.png`.
- Pipeline 3, stage B (supervised fine-tune from the SSL checkpoint):
  `train.py configs/epic_slowfast_verb.yaml --init-encoder
  ckpt_ssl/ssl_epic_slowfast_verb_neco/epoch_012.pth.tar --output from_ssl_neco`
  (job 5077230, `train_verb_from_ssl.slurm`, COMPLETED, exit 0, 1h06m;
  encoder-init verified: 207 encoder keys matched, 22 head keys left at random
  init, 0 unexpected). Result: `ckpt/epic_slowfast_verb_from_ssl_neco/epoch_021.pth.tar`,
  23.56 avg mAP. Loss comparison: `figs/supervised_loss_comparison.png`.