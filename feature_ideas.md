# Ideas: improving the features for downstream TAL

*Created 2026-10-09. Research notes for the thesis — EPIC-KITCHENS-100 verb
temporal action localization (TAL) with ActionFormer.*

## Why this list exists

The thesis's central empirical finding is that the **features dominate** and that
self-supervised training on the *detector encoder* does not help:

| Experiment | avg mAP |
|---|---|
| VideoMAE-L supervised baseline | 29.60 |
| VideoMAE-L + temporal-crop SSL -> supervised | 29.38 |
| VideoMAE-L + spatial-crop SSL -> supervised | 29.38 |
| SlowFast supervised baseline | 23.38 ± 0.75 (n=3) |
| SlowFast + SSL -> supervised | 23.65 ± 0.08 (n=3) |
| V-JEPA2 supervised baseline | 17.49 |
| V-JEPA2 + SSL -> supervised | 17.40 |

- Feature choice moves the metric by **+6.2 pp** (VideoMAE-L over SlowFast),
  an order of magnitude more than any SSL effect we've observed.
- SSL *post*-training actively hurts; SSL *pre*-training gives no mean gain on
  any track.
- Our spatial-crop SSL only reshapes the ActionFormer encoder on top of a
  **frozen** backbone, so it cannot change the information content of the
  features — which is the binding constraint.

Conclusion: the highest-leverage directions all **change the features
themselves** (the backbone, the modality, or the temporal/spatial detail kept),
not the detector or the SSL augmentation.

---

## A. Change the features (backbone / data) — highest leverage

### A1. End-to-end fine-tuning of the backbone with the detector
Train VideoMAE-L (or SlowFast) jointly with ActionFormer on EPIC verbs instead
of freezing it. This is exactly what OpenTAD's **AdaTAD** does
(EPIC verb 29.69 vs our 29.60 feature-based).
- Feasible on 40 GB MIG GPUs via **parameter-efficient adapters** (AdaTAD's
  temporal-informative adapter, TIA) or **LoRA** rather than full fine-tuning.
- Directly attacks "the features are fixed".

### A2. Domain-adaptive SSL on the backbone itself
Instead of NeCo on the ActionFormer encoder (shown to do nothing), continue
**MAE / VideoMAE pretraining on unlabeled EPIC videos** with the target
backbone, then fine-tune on EPIC verbs.
- This is NeCo's *actual* regime (post-train the backbone) and genuinely changes
  the representation.
- Our token-preserving VideoMAE extractor is most of the machinery already.

### A3. Stronger / larger backbones
Swap VideoMAE-L for **InternVideo2** or **VideoMAEv2-g** features (OpenTAD
releases them). Pure "download + retrain detector"; no architecture work.
- Feature quality has been the dominant lever, so a better backbone is the
  cheapest way to a higher number.

### A4. Multi-modal features
EPIC-KITCHENS is egocentric, where **audio is very informative** (EPIC-Sounds
detection exists in OpenTAD). Fuse visual + audio, and optionally
**hand / object** cues (hand-object interaction is the crux of EPIC actions).
- Concatenate or cross-attend; typically worth several pp.
- Also consider optical-flow / motion features as an extra stream.

### A5. Longer temporal context in the features
Current features are independently pooled 16-frame snippets. Use
**frame-encoding** backbones (VideoSwin / VideoMAEv2) that keep temporal context
across a long clip, or simply keep the **tubelet tokens** instead of
mean-pooling them, so the detector sees sub-clip dynamics.
- Also: denser vs sparser temporal stride. Note our VideoMAE features are 2x
  denser than SlowFast (3.75 vs 1.875 vectors/s); matching density is needed for
  apples-to-apples comparisons.

---

## B. Change how the features are consumed

### B1. Keep the spatial dimension (true 2D NeCo)
Run a spatio-temporal model that keeps the token grid, rather than pooling each
crop to one vector per timestep. This is the direction we started (spatial-crop
SSL) but cut short by design A. Real 2D ROI-Align + patch-order loss.

### B2. Temporal aggregation in the detector
Replace ActionFormer's local transformer with causal / state-space temporal
modeling: **CausalTAD**, **VideoMambaSuite** (Mamba), hybrid causal blocks.
These report gains on EPIC.

### B3. Multi-scale / multi-stride features
Feed the detector features at several temporal strides (e.g. stride 8 + stride
16 + pooled) as extra input channels.

---

## C. Data & supervision

### C1. Noun + verb multi-task training
EPIC has both noun and verb labels; shared representations often help the verb
task.

### C2. Distillation / pseudo-labeling
Distill from a stronger teacher (e.g. an end-to-end model) into the
feature-based detector.

### C3. Feature-level augmentation that actually helps
Our frame-rate augmentation gave +0.8 pp on one seed — worth replicating with
seeds and combining with other augmentations.

---

## D. Cheap wins worth doing regardless

### D1. Feature fusion
Concatenate SlowFast + VideoMAE (+ audio) — usually beats any single track.

### D2. Close the reproduction gap
Reproduce OpenTAD's ActionFormer-SlowFast = **24.93** (vs our 23.07) to close the
inference / recipe gap before attributing everything to features.

---

## Priority / cost table

| # | Idea | Expected leverage | Cost / feasibility | Notes |
|---|---|---|---|---|
| A1 | Adapter / LoRA end-to-end fine-tune | High | Medium–high | AdaTAD-style TIA; 40 GB MIG feasible |
| A2 | Domain-adaptive MAE on EPIC | High | Medium | Uses our existing extractor machinery |
| A3 | InternVideo2 / VideoMAEv2-g features | High | Low | Download + retrain detector |
| A4 | Audio (+ hand/object) fusion | Medium–high | Low–medium | Distinctive for egocentric EPIC |
| A5 | Longer context / keep tubelet tokens | Medium | Medium | Changes feature format |
| B1 | True 2D NeCo (keep token grid) | Medium | Medium–high | Decouples from ActionFormer encoder |
| B2 | Causal / Mamba temporal model | Medium | Medium | CausalTAD / VideoMambaSuite |
| B3 | Multi-scale features | Low–medium | Low | Extra input channels |
| C1 | Noun+verb multi-task | Low–medium | Low | |
| C2 | Distillation / pseudo-labels | Medium | Medium | |
| C3 | Replicate frame-rate aug | Low | Low | +0.8 on 1 seed |
| D1 | Feature fusion | Medium | Low | Cheap ensemble |
| D2 | Reproduce OpenTAD 24.93 | — | Low | Sanity check before attributing to features |

## Recommendation

- **Highest leverage / moderate cost:** A1 (adapter/LoRA end-to-end) or A2
  (domain-adaptive MAE on EPIC). Both directly change the features and are the
  natural next thesis chapter.
- **Cheapest high leverage:** A3 (InternVideo2 features) and A4 (audio fusion).
- **Most distinctive for egocentric EPIC:** A4 (audio) and hand-object cues.

## References / resources
- OpenTAD (feature releases, AdaTAD, E2E training): https://github.com/sming256/OpenTAD
- AdaTAD: *End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames* (CVPR 2024)
- NeCo: *Improving DINOv2's Spatial Representations with Patch Neighbor Consistency* (ICLR 2025)
- CausalTAD / VideoMambaSuite (temporal modeling) — OpenTAD configs
- EPIC-Sounds (audio-based detection) — OpenTAD configs
- InternVideo2 / VideoMAEv2 (stronger backbones)
