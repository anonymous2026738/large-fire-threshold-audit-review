# Reproducibility notes

Build date: 2026-06-04

## Verified in this environment

| Step | Command | Result |
|------|---------|--------|
| Mini sample | `D:\.conda\envs\py310\python.exe scripts/build_mini_sample.py` | OK — `data/processed/mini_sample.pth` (12 samples) |
| Figure reproduction | `D:\.conda\envs\py310\python.exe scripts/run_audit_demo.py` | OK — figures under `results/figures_reproduced/` |

Environment: **conda `py310`** (Python 3.10). Default system Python 3.11 had a broken `torch` DLL and was not used.

### Figure reproduction output (subset)

- `performance_comparison_optimized.png`
- `threshold_tradeoff_curves.png`
- `roc_curve_overall_and_pop_group.png` (and GDP / continent variants)
- `xai_top5_features_by_pop_group.png`
- Fairlearn metrics printed for population_density optimized: EOD **0.1436**, DPD **0.1676** (matches published audit tables)

## Not verified here (requires full data / long GPU run)

| Step | Blocker | How to complete |
|------|---------|-----------------|
| Rebuild 7× `.pth` shards | Raw FireTracks + rasters not in repo | Follow `docs/BUILD_PTH_COMMANDS.md` after obtaining sources in `data/README.md` |
| Full training grid (16 + 4 runs) | ~hours GPU + full shards | `python scripts/train.py -m ...` |
| Full audit from checkpoint + shards | Full `.pth` not redistributed | `python scripts/fairness_analysis.py --checkpoint ... --data data/processed/*.pth --xai` |
| Fig. 1 input patch | No automated generator in repo | Use published `figures/manuscript/Fig1_input_patch.png` |

## Privacy / path scan (release tree)

- No `E:\FireEqual` paths in tracked source/docs (contact email only in `README.md` / `CITATION.cff` by design).
- No API tokens committed; Orion-style logger env vars are not used in this release configs.

## Recommended reviewer quick check

```bash
conda activate py310   # or fire_py310 / fire-danger-audit
pip install -r requirements.txt
python scripts/run_audit_demo.py
```

Expected runtime: ~3–5 minutes CPU for figure reproduction.
