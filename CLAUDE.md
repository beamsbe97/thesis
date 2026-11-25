# Thesis: Self-Supervised Post-Training for Temporal Action Localization

Research question: does a **NeCo-style self-supervised (SSL) stage** improve
**ActionFormer** (Zhang et al., ECCV 2022) on temporal action localization
(TAL), EPIC-KITCHENS-100 verb task? Two SSL protocols are compared: post-training
on top of a converged supervised model, and pretraining before supervised
training (the order used in the original NeCo/DINO-style literature).

**Answer (2026-11-23, corrected pipeline): no.** Post-training hurts (-1.34 pp),
pre-training gives no significant gain on SlowFast (3 seeds) or VideoMAE-L. The
thesis write-up is `paper/main.pdf` (see "Paper" below).

Work happens on the **ALICE** cluster (Leiden University, host alias `alice`,
`/zfsstore/user/s4561341/thesis`), which is kept in sync with this repo via
GitHub (`beamsbe97/thesis`). This machine's git identity cannot push to that
remote directly — commits made here are transferred to Alice as a `git
bundle` and pushed from there (Alice's SSH key is the one with write access).

## IMPORTANT: results before 2026-11-22 are superseded

Three problems in the SSL pipeline were found and fixed on 2026-11-22 (details in
"Architectural decisions"). **Every SSL result produced before that date is
invalid**: the old pipeline-3 numbers (23.56 / 23.65 +- 0.08, the 24.45 frame-rate
result, VideoMAE 29.38 / 28.90 / 28.65, V-JEPA2 17.40), the old pipeline-2 numbers
(22.55, VideoMAE 29.24), the joint-fine-tune ablation (22.45 vs 22.97), and every
diagnostic computed on those encoders (CKA / weight-cosine, the 0.47% / 0.08%
encoder movement, the loss-start comparison, mixer analysis). Only the supervised
baselines and the NeCo/VOC control (`neco_diag/`) are unaffected. Use only the
"Results" tables below. The paper must not mention the bugs or the pre-fix results
(user instruction).

## Repo layout

- `actionformer/` — the active project: ActionFormer training/eval +
  the NeCo-style SSL add-on. Mirrors the official ActionFormer codebase
  (`libs/`) with SSL bolted on top (`libs/modeling/selfsup.py`, `train_ssl.py`).
- `paper/` — **the thesis** (LaTeX, `main.tex` -> `main.pdf`, 11 pp); figures are
  regenerated from `paper/data/*.json` by `paper/scripts/make_figures.py`; build steps
  and data provenance in `paper/README.md`.
- `neco_diag/` — NeCo in its own setting (DINOv2 -> PASCAL VOC): linear probe,
  gradient alignment, neighbour-label agreement. Valid control for the paper.
- `vjepa2/` — Meta's V-JEPA2 codebase, used only as a frozen feature extractor.
- `current_lit.md` — running research notes/log (dated entries).
- `feature_ideas.md` — brainstormed directions for improving the *features*.
- `report.md` (root) — long-form experiment log; its body describes the
  **pre-fix** runs (see its header). `actionformer/report.md` is older still.

## Data & features

- **Dataset**: EPIC-KITCHENS-100, verb-only track (97 classes). 495 training
  videos, 138 validation videos (32 kitchens / participants in validation).
- **SlowFast** (primary): 2304-d, EPIC-pretrained, 32-frame window / 16-frame
  stride at 30 fps -> ~1.875 vectors/s. Config `configs/epic_slowfast_verb.yaml`.
- **VideoMAE-L** (OpenTAD, fine-tuned on EPIC verbs): 1024-d, stride 8 (2x
  SlowFast density). Config `configs/epic_videomae_verb.yaml`. Spatial-crop
  variant re-extracted into `data/epic_kitchens/features_videomae_verb_spatial`
  (16 crops/snippet; `actionformer/SPATIAL_EXTRACTION.md`).
- **V-JEPA2**: 1024-d frozen ViT-L, not EPIC-adapted; supervised baseline only.

## Pipelines

All share the same ActionFormer encoder (`ConvTransformerBackbone` + identity FPN
neck, 207 tensors), so weights transfer; only the heads differ.

1. **P1 supervised** (`train.py`): focal + DIoU, center sampling, soft-NMS;
   5 warm-up + 16 epochs, AdamW lr 1e-4, batch 2. Checkpoint
   `ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar` (EMA model is reported).
2. **P2 SSL post-training**: `train_ssl.py --init-encoder <sup ckpt>` (loads the
   supervised **EMA** weights) with `configs/ssl_epic_slowfast_verb_fixed.yaml`
   (NeCo's post-training recipe: encoder lr 2e-5, head x10, clip 1, 12 ep), then
   `eval_ssl.py` splices the SSL EMA encoder onto the frozen supervised EMA heads.
   SLURM: `ssl_fixed.slurm` (with `WARM=`) -> `eval_ssl_fixed.slurm`.
3. **P3 SSL pre-training -> supervised**: stage A `train_ssl.py --from-scratch`
   with `configs/ssl_epic_{slowfast,videomae}_verb_fixed_v2.yaml` (from-scratch
   recipe: encoder lr 1e-4, head x1, clip 5, 2+18 ep) or
   `configs/ssl_epic_videomae_verb_spatial_fixed_v2.yaml`; stage B
   `train.py --init-encoder <ssl ckpt>` (loads the SSL EMA teacher's backbone+neck,
   heads random, unmodified P1 recipe). SLURM: `ssl_fixed.slurm` ->
   `train_verb_from_ssl_fixed.slurm` (`SUP_CONFIG=` for VideoMAE).

## Architectural decisions

- **NeCo references come from the teacher** (`loss.ref_source: teacher`, the
  default), as in the NeCo paper. All pre-2026-11-22 SSL configs are pinned to
  `ref_source: student` (the old, wrong behaviour) only so they still reproduce.
- **EMA teacher momentum 0.99 -> 1** in the fixed configs. NeCo's 0.9995 is meant
  for post-training a pretrained model over tens of thousands of steps; in our
  1476-2460-step runs it left the teacher (= the transferred encoder) at its init.
  Check `tools/ssl_teacher_movement.py`: the teacher should move about as much as
  the student (fixed runs: 4.4% vs 4.5%).
- **SSL warm start loads the supervised EMA weights** (`train_ssl.py` default;
  `--init-encoder-raw` = old behaviour, pinned in the old warm slurm scripts).
  The EMA model is what is evaluated and whose heads `eval_ssl.py` uses; raw and
  EMA encoders differ ~1.7%, which alone cost 0.37 mAP in the old P2 numbers.
- **`--from-scratch`** for random-init SSL (disables checkpoint auto-discovery;
  previously random init relied on `--ckpt-dir` pointing where nothing matched).
- **Shared `load_init_encoder`** (`libs/utils/train_utils.py`): keeps everything
  except head-prefixed keys, `strict=False`, `prefer_ema` selects the EMA state.
  `train.py --init-encoder` trains everything (nothing frozen).
- **SLURM**: the `*_fixed.slurm` scripts request `--cpus-per-task=4` — with 1 CPU
  the 4 DataLoader workers deadlock (job 5703156 froze; 4 CPUs also halve stage-B
  time to ~35 min). Use `python -u`. Copy files to Alice by `tar | ssh` into the
  repo, not `scp` to `/tmp` (login nodes have separate `/tmp`). Scripts run from
  `tools/` need the repo root on `sys.path`.
- **git bundle instead of direct push**: `git bundle create <file>
  origin/main..HEAD`, scp to Alice, `git bundle verify`, fetch into a temp branch,
  fast-forward `main`, push from Alice.

## Results (corrected pipeline; EPIC-100 verb, validation, mAP @ tIoU 0.1-0.5)

| Model | .1 | .2 | .3 | .4 | .5 | **Avg** |
|---|---|---|---|---|---|---|
| ActionFormer paper (SlowFast) | 26.6 | 25.4 | 24.2 | 22.3 | 19.1 | 23.5 |
| P1 SlowFast, seeds 0/1/2 = 23.07 / 22.84 / 24.23 | 26.67 | 25.51 | 24.16 | 21.98 | 18.58 | **23.38 +- 0.75** |
| P2 post-training, frozen heads (from seed 0, 23.07) | 25.05 | 24.05 | 22.49 | 20.07 | 17.01 | **21.73** |
| P3 SlowFast fixed_v2, seeds 0/1/2 = 24.02 / 23.72 / 23.71 | 26.89 | 25.96 | 24.38 | 22.59 | 19.27 | **23.82 +- 0.18** |
| P3 SlowFast, post-training recipe from scratch (enc. moves 0.43%) | 26.78 | 25.39 | 23.92 | 22.07 | 18.67 | 23.37 |
| P1 VideoMAE-L | 32.84 | 32.06 | 30.42 | 28.11 | 24.59 | **29.60** |
| P3 VideoMAE-L fixed_v2, temporal crops | 32.90 | 32.25 | 30.38 | 28.12 | 24.54 | **29.64** |
| P3 VideoMAE-L fixed_v2, spatial crops | 32.48 | 31.78 | 30.20 | 27.85 | 23.77 | **29.22** |
| P1 V-JEPA2 | 20.51 | 19.74 | 18.38 | 16.17 | 12.64 | 17.49 |

- SlowFast P3 vs P1: +0.44, Welch t ~ 1.0 (df ~2), **not significant**; seed-paired
  +0.95 / +0.88 / -0.52 (driven by the high baseline seed 2).
- Control (no SSL, raw supervised encoder + EMA heads): 22.70 — only relevant to
  explain the old P2 number; not in the paper.
- VideoMAE-L and P2 are single runs.

## Analysis (why NeCo does not help) — all on the corrected encoders

- **The SSL stage trains**: val agreement 0.92-0.96, NeCo loss 13-19 -> 1.0-1.6;
  transferred backbone moves 4.4% (SF) / 4.8% (VM temporal) / 4.5% (VM spatial)
  from init, head 8-9%; supervised training moves it ~23%. P2 moves the supervised
  encoder 0.87%. (`paper/data/fixed_movement.json`)
- **Stage B moves the encoder 23-24% away from the SSL weights** (SF seeds 23.4 /
  23.5 / 23.8%, VM temporal / spatial 23.3 / 23.4%; `tools/weight_distance.py`,
  `figs/stageB_distance_from_ssl.json`) = as far as P1 from random init (22.6%) and
  ~5x the SSL stage's movement -> paper Fig. 4a and §6.4.
- **Neighbour-label probe** (`tools/knn_label_agreement.py`, level-0 features, k=10,
  lift over chance, cross-video / within-video; `figs/knn_label_*.json`):
  SlowFast input .314/.391, random .311/.394, **NeCo SSL .261/.340**, supervised
  .375/.427, SSL->sup .370/.427, P2 .375/.426. VideoMAE input .397/.473, random
  .405/.482, **SSL .383/.458, SSL spatial .382/.468**, supervised .414/.485,
  SSL->sup .415/.485. => from scratch, NeCo moves neighbourhoods *away* from verbs;
  supervised training erases it; P2 keeps the verb grouping (its -1.34 is the
  features shifting under frozen heads).
- **Kitchen probe** (same-kitchen fraction of cross-video neighbours, chance .029):
  SF input .52, random .50, **SSL .31**, sup .34; VM input .48, random .43, **SSL
  .29 (both view types)**, sup .37 => NeCo *removes* scene identity: no scene
  shortcut. It flattens neighbourhoods (less kitchen and less verb structure).
- **Head-start probe** (seed-matched first-epoch supervised loss): SSL init is
  +4.0% / +4.7% (SF seeds 1, 2) and +1.4% (VM) *worse*; NeCo on VOC is 5-10% better.
- **Views**: temporal crops give identical input to 93% (SF) / 97% (VM) of level-0
  ROI bins, but spatial crops (differ everywhere) give the same lift drop and no
  gain -> not an augmentation effect.
- **VOC control** (`neco_diag/`): NeCo +2.0 mIoU, neighbour lift .83->.90 / .85->.92,
  5-10% lower first-epoch loss. Same probes, opposite sign to EPIC.
- **Interpretation (paper §6.7)**: views and scene shortcuts are ruled out; what
  remains is the starting representation — NeCo re-mixes clip features already
  supervised for the target verbs, and its other-video ordering target does not
  coincide with verb identity.

## Paper

`paper/main.pdf` (11 pp), rewritten 2026-11-23 from corrected results only. Open
items: author name placeholder, OpenTAD arXiv number in `refs.bib`. Previous
(pre-fix) draft kept as `paper/.main_prev_version.tex`, reference only.

## Current status

- 2026-11-23: corrected-pipeline experiments and analyses complete (jobs below);
  paper rewritten; `report.md` marked superseded; `CLAUDE.md` rewritten.
- 2026-11-22/23 job IDs (all corrected pipeline): SlowFast P3 fixed_v2 stage A
  5702531 / 5702532 / 5703155, stage B 5702534 / 5703154 / 5704521; SlowFast P3
  post-training recipe 5702537 -> 5702538; P2 5702539 -> eval 5702540; no-SSL
  raw/EMA control eval 5702541; VideoMAE temporal 5703157 -> 5704522; VideoMAE
  spatial 5705829 -> 5705830; probes 5705714 / 5705832 (main), 5705833 (P2),
  5705831 (spatial).

Older log (everything below describes **pre-fix** runs; kept for history only):

- 2026-11-22: first paper draft from pre-fix runs; found the EMA-teacher lag
  (teacher moved 0.08%) and view identity, which led to the bug hunt above.
- 2026-10-23: VideoMAE spatial v2 (28.65) — pre-fix, invalid.
- 2026-10-22: encoder-similarity / weight-movement diagnostic — pre-fix encoders.
- 2026-10-09: spatial-crop extraction complete (628/633 videos; 5 corrupt MP4s
  skipped) — the extracted features are valid; the SSL runs on them were pre-fix.
- 2026-10-08: spatial-crop workstream started (`libs/datasets/ssl2d.py`, `sslv2`;
  `tools/videomae_backbone.py`, `tools/extract_videomae_spatial.py`).
- 2026-09-25: VideoMAE-L track added (baseline 29.60, valid); seed replicates of
  the SlowFast baseline (valid: 23.07 / 22.84 / 24.23).
- 2026-09-21..25: pipeline 2/3, joint fine-tune, V-JEPA2, frame-rate runs — pre-fix.
- Jobs 5093341 / 5093366 were cancelled by mistake (buffered `.out` logs misread
  as a hang); newer slurm scripts use `python -u`.

## Next steps

1. Commit and push all 2026-11-22/23 work (git bundle via Alice).
2. Paper: author name; OpenTAD citation; final proofread.
3. Optional: re-fit the heads after P2 (tests the head-mismatch reading, ~20 min);
   extra seeds for VideoMAE-L P3 and P2.
4. Pre-fix figures in `actionformer/figs/` (`ssl_analysis_*`,
   `encoder_similarity_*`, `loss_start_*`, `ssl_mixer_levels.json`,
   `teacher_student_delta.json`) should be moved/labelled as pre-fix.
5. OpenTAD's ActionFormer-SlowFast verb number is 24.93, above our 23.38 repro —
   worth checking against their SlowFast features.
