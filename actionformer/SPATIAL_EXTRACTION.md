# Spatial-crop VideoMAE extraction — status

*Last updated: 2026-10-08. Work is running on ALICE; the local PC can be
suspended — everything needed to check or resume is below.*

## What is running

Two SLURM jobs on partition `gpu-mig-40g` (exclusive 40 GB MIG, 60 CPUs each),
launched from `~/vm_spatial/` on ALICE:

| Job ID | Name | Shards |
|---|---|---|
| **5166540** | `vmsp_b0` | global shards 0–5 of 12 |
| **5166541** | `vmsp_b6` | global shards 6–11 of 12 |

Each job runs **6 worker processes**; each worker handles one shard and
processes its videos sequentially, then the job repeats for the `validation`
split. The extractor skips any video that already has a `meta.json`, so the jobs
are resumable.

**Why only 2 jobs:** the per-user QOS CPU cap allows ~120 CPUs total, so a
larger fan-out just queues. Expected wall time ≈ **17 h** for all 633 videos
(495 train + 138 val, ~1.19 M snippets). Extraction is CPU/decode-bound; the
VideoMAE forward is negligible (~0.003 s/snippet).

**Progress (2026-10-09):** extraction **complete** — **628 / 633** videos,
1,152,051 snippets. The 5 missing are corrupt MP4s PyAV cannot decode
(`P06_05, P29_01, P30_05` train; `P29_05, P30_08` val; `InvalidDataError`); the
SSL dataset skips absent videos, so 628/633 is used. Spot-check clean.

**Stage A (spatial-crop NeCo SSL) done 2026-10-09:** job **5175368** (random
init, 12 epochs) converged cleanly — val NeCo loss 11.97 → **2.80**, neighbor
agreement → **0.843** (cleaner than the earlier temporal VideoMAE SSL run).

**Stage B (supervised VideoMAE-L from the SSL EMA encoder) done:** job
**5175414**, `init-encoder` loaded 207 encoder keys (22 head keys random, 0
unexpected). **Result: avg mAP 29.38** (best 29.46) — **no gain** vs the
supervised baseline **29.60**, and identical to the temporal-crop SSL result
(29.38). Curves: `figs/ssl_pretrain_loss_spatial.png`,
`figs/supervised_loss_comparison_spatial.png`.

### How to check

```bash
# on ALICE
squeue -u s4561341                       # job state
ls ~/thesis/actionformer/data/epic_kitchens/features_videomae_verb_spatial | wc -l   # videos done (target: 633)
tail ~/vm_spatial/log_training_*.log     # per-shard progress
tail ~/vm_spatial/vmsp_b0_*.out          # job stdout
```
`ls ... | wc -l` climbing to 633 = done. A healthy log line looks like
`[3/41] P01_17 -> (444, 16, 1024) in 260.3s`.

### How to resume / relaunch

```bash
cd ~/vm_spatial
sbatch --export=ALL,NPROC=6,GLOBAL_SHARDS=12,SHARD_BASE=0 --job-name=vmsp_b0 extract_videomae_spatial.slurm
sbatch --export=ALL,NPROC=6,GLOBAL_SHARDS=12,SHARD_BASE=6 --job-name=vmsp_b6 extract_videomae_spatial.slurm
```
Already-finished videos are skipped automatically.

## Paths on ALICE

- Code (mirror of `actionformer/tools/` in this repo):
  `~/vm_spatial/tools/videomae_backbone.py`, `~/vm_spatial/tools/extract_videomae_spatial.py`
- Launcher: `~/vm_spatial/extract_videomae_spatial.slurm`
- Checkpoint: `~/thesis/pretrained/videomae_l_epic_verb.pth` (OpenTAD VideoMAE-L, EPIC-Verb finetuned)
- Raw videos: `/zfsstore/courses/2025-2026/4343SADL6/g10/EPIC-KITCHENS/P*/videos/*.MP4`
- **Output** (data, not in git; ~39 GB):
  `~/thesis/actionformer/data/epic_kitchens/features_videomae_verb_spatial/<video_id>/`
  ```
  meta.json         {"K": 16, "T": <snippets>, "grid": [14, 14]}
  boxes.npy         (K, 4) normalized [x0,y0,x1,y1]; index 0 = full frame
  view_000.npy ...  (T, 1024) fp16
  ```

## Pipeline results (done 2026-10-09)

1. Sanity-check: 628 video folders (5 corrupt MP4s skipped); spot-check clean.
2. **Stage A — spatial-crop NeCo SSL from random init** (`train_ssl_videomae_spatial.slurm`):
   done, job 5175368, val loss 11.97 → 2.80, agreement 0.843.
3. **Stage B — supervised VideoMAE-L from the SSL EMA encoder**
   (`train_verb_videomae_from_ssl_spatial.slurm`): done, job 5175414,
   `init-encoder` 207 matched / 22 missing / 0 unexpected.
4. **Result: avg mAP 29.38** (best 29.46) vs baseline **29.60** and temporal-crop
   SSL **29.38** → no gain (replicates the thesis finding).

## Follow-up: combined spatial + temporal crops (running 2026-10-09)

Tests whether the crop *style* is the limiting factor: each view is a spatial
crop (`sslv2` layout) further restricted to a random temporal crop, with the two
views aligned on their temporal intersection (`sslv3`, reuses `build_views`
unchanged).

- Dataset `sslv3` (`SelfSupSpatialTemporalDataset` in `libs/datasets/ssl2d.py`),
  config `configs/ssl_epic_videomae_verb_spatial_temporal.yaml`
  (`temporal_crop_scale [0.4, 1.0]`, `temporal_min_overlap 0.2`,
  `temporal_min_crop_len 256`).
- Stage A: job **5176610** (`train_ssl_videomae_spatial_temporal.slurm`,
  `--output neco_spatial_temporal`). Epoch-0 val loss 13.46 — in line with the
  other runs.
- Stage B: job **5176611** (`--dependency=afterok:5176610`).
- **Result: pending.** If it also lands at ~29.38 it is a third null for "crop
  style"; if it improves, the frozen-features hypothesis is wrong.

## Code state (this repo, not yet committed/pushed)

New/changed files for this workstream:

- `actionformer/libs/datasets/ssl2d.py` (new, `sslv2` dataset)
- `actionformer/libs/datasets/__init__.py` (registers `ssl2d`)
- `actionformer/train_ssl.py` (`build_views` accepts a per-view `view_feats` source)
- `actionformer/configs/ssl_epic_videomae_verb_spatial.yaml` (new)
- `actionformer/train_ssl_videomae_spatial.slurm`, `actionformer/train_verb_videomae_from_ssl_spatial.slurm` (new)
- `actionformer/tools/videomae_backbone.py`, `actionformer/tools/extract_videomae_spatial.py` (new)
- `actionformer/extract_videomae_spatial.slurm` (new)
- `actionformer/README.md` (documentation section)

Nothing is committed yet. To move to ALICE the usual way: commit locally →
`git bundle` → fetch/merge/push from ALICE (see `CLAUDE.md`).

## Validation already done

- Checkpoint loads strictly into the reconstructed VideoMAE-L (305.5 M params);
  token grid `(1, 8, 14, 14, 1024)` confirmed.
- Extractor runs end-to-end; sample output shape `(T, 16, 1024)`.
- Full-frame view (crop 0) vs released OpenTAD pooled features: row cosine
  ~0.88 (0.91 after a 1-snippet centering offset that does not matter to the
  SSL encoder) — preprocessing is consistent.
