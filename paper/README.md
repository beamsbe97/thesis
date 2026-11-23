# Thesis paper

`main.tex` + `refs.bib` -> `main.pdf` (11 pages). All data figures are vector PDFs in
`figs/`, regenerated from the JSON files in `data/` by `scripts/make_figures.py`.

## Rebuild

```bash
# 1. (only if job logs changed) parse ALICE job logs into data/curves.json + data/durations.json
python3 scripts/parse_logs.py <dir with *.out + data/epic_kitchens/annotations> data

# 2. figures (numpy + matplotlib; on ALICE: ~/.conda/envs/neco_diag/bin/python)
python scripts/make_figures.py

# 3. PDF (on ALICE: ~/.conda/envs/texenv/bin/tectonic, installed from conda-forge)
tectonic -X compile main.tex
```

## Data provenance

| file | produced by |
|---|---|
| `data/curves.json` | `scripts/parse_logs.py` over the ALICE `*.out` job logs |
| `data/durations.json` | EPIC-100 verb annotation file (training split) |
| `data/teacher_student_delta.json` | `actionformer/tools/ssl_teacher_movement.py` (new: student *and* EMA-teacher weight movement) |
| `data/ssl_mixer_levels.json` | `actionformer/tools/ssl_mixer_levels.py` (new: per-level SSL mixer weight norms) |
| `data/view_identity.json` | `scripts/make_figures.py` (receptive-field / view-identity analysis) |
| `data/ssl_analysis_summary.json`, `data/loss_start_comparison.json` | `actionformer/tools/plot_ssl_analysis.py`, `analyze_loss_start.py` |
| `data/probe_*`, `xfer_*`, `neighbor_*`, `grad_align_*` | `neco_diag/*.py` |
| `data/fixed_movement.json` | `actionformer/tools/ssl_teacher_movement.py` on the SSL / supervised checkpoints used in the paper |
| `data/knn_label_*.json` | `actionformer/tools/knn_label_agreement.py` (neighbour-label + kitchen probe; SLURM `knn_label_*.slurm`) |

The paper reports only the corrected SSL pipeline (teacher references, EMA momentum
0.99, EMA warm start; configs `*_fixed*.yaml`). `.main_prev_version.tex` is the earlier
draft based on the pre-fix runs, kept for reference only.
