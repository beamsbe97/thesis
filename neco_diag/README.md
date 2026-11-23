# NeCo diagnostics in its own setting (DINOv2 → PASCAL VOC)

Three probes that test whether the NeCo objective is "too far from the downstream
objective", run in the setting the method was designed for: NeCo post-training of
a DINOv2 ViT-S/14 backbone, evaluated on PASCAL VOC 2012 semantic segmentation.
Results and interpretation are in `report.md` §7.

| script | question |
|---|---|
| `linear_probe.py` | does an SSL-initialised backbone give a lower linear-probe loss/mIoU? (option 2) |
| `grad_alignment.py` | does the NeCo gradient point the same way as the segmentation gradient on the backbone? (option 4) |
| `neighbor_label_agreement.py` | do NeCo's patch neighbours share the query patch's GT label? (option 5) |
| `make_figs.py` | builds `figs/neco_diagnostics.png` + the summary table from `results/*.json` |

## Environment

Runs against the official repository (`vpariza/NeCo`) for the model, the NeCo
loss and the VOC data module, so `PYTHONPATH` must point at it:

```bash
conda activate neco_diag          # torch 2.5.1+cu124, timm, pytorch-lightning, diffsort, faiss
export PYTHONPATH=/path/to/neco_repo
```

Data layout expected by the loaders (VOC 2012 + `SegmentationClassAug`):

```
VOCSegmentation/
  images/{JPEGImages,SegmentationClass,SegmentationClassAug}
  sets/{trainaug.txt,val.txt}
```

Checkpoints: `dinov2_vits14_pretrain.pth` and the released NeCo
`neco_on_dinov2_vit14_{model,teacher}.ckpt`. The released NeCo checkpoints are
plain dicts (backbone + `projection_head.*`), not PyTorch-Lightning wrappers.

## Option 2 — linear probe

```bash
python linear_probe.py --backbone dinov2 --ckpt ckpt/dinov2_vits14_pretrain.pth \
  --data voc --data-root data/voc/VOCSegmentation --epochs 25 --batch-size 128 \
  --out results/probe_dinov2_voc.json
```

Frozen backbone, one 1×1 convolution on the patch tokens resized to 100×100,
SGD lr 0.01 / wd 1e-4, step decay ×0.1 at epoch 20 (the paper's protocol).
Both backbones must be run with the same `--seed` (default 400) **and the same
`--batch-size`** (128): with different batch sizes the number of optimizer steps
per epoch changes, so the epoch-0 loss no longer isolates the features. Only then
is the epoch-0 loss a clean comparison.

Cross-head transfer freezes a trained head and evaluates it on the other
backbone:

```bash
python linear_probe.py --backbone neco --ckpt ckpt/neco_dinov2_vits14_student.ckpt \
  --eval-only --head results/probe_dinov2_voc_head.pt --out results/xfer_dinov2head_neco.json
```

## Option 4 — gradient alignment

```bash
python grad_alignment.py --backbone neco \
  --student-ckpt ckpt/neco_dinov2_vits14_student.ckpt \
  --teacher-ckpt ckpt/neco_dinov2_vits14_teacher.ckpt \
  --head results/probe_neco_voc_head.pt --iters 20 --batch-size 8 \
  --out results/grad_align_neco.json
```

Cosine between `∇_backbone L_NeCo` (the real `shared_step` loss) and
`∇_backbone L_CE` (through the trained probe head), over the full backbone
parameter vector, plus a per-parameter-group breakdown. `ce_vs_ce_noise` is the
same cosine between two CE gradients on different batches — the noise floor.

## Option 5 — neighbourhood vs. GT labels

```bash
python neighbor_label_agreement.py --backbone neco \
  --ckpt ckpt/neco_dinov2_vits14_student.ckpt --n-images 400 --k 10 \
  --out results/neighbor_agreement_neco.json
```

For each 14×14 patch of 400 val images, the fraction of its top-k cosine
neighbours (within the image, and across the whole set) that share its majority
GT label, with the label-prior chance level and the resulting lift; plus k-means
NMI/ARI against the 21 VOC classes.

## SLURM

`scripts/*.slurm` are the ALICE batch versions of the same jobs
(`probe.slurm` takes `BACKBONE`, `DATASET`, `CKPT`, `DATA_ROOT`, `EPOCHS`, `BS`).
They assume `Miniforge3` is loaded and unset `LD_LIBRARY_PATH` after activating
the env, otherwise the module CUDA stack conflicts with the env's torch.
