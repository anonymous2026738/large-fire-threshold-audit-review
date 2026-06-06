# Release manifest (v1.0.0)

## New / reorganised for public release

| Path | Role |
|------|------|
| `README.md` | Project overview & quick start |
| `LICENSE` | MIT |
| `requirements.txt`, `environment.yml` | Dependencies |
| `.gitignore` | Excludes raw data, secrets, large shards |
| `CITATION.cff`, `zenodo_metadata.json` | Zenodo / citation |
| `ZENODO_UPLOAD_INSTRUCTIONS.md`, `docs/GITHUB_RELEASE.md` | Publishing steps |
| `REPRODUCIBILITY_NOTES.md` | Validation log |
| `scripts/_release_paths.py` | Relative path helper |
| `scripts/reproduce_figures.py` | Figure reproduction from cache |
| `scripts/run_audit_demo.py` | Minimal demo entry |
| `scripts/build_mini_sample.py` | Synthetic `.pth` format demo |
| `data/README.md` | Upstream data provenance |
| `data/processed/audit_*.csv` | Published group metrics tables |
| `data/processed/schema.json` | Tensor schema |
| `results/checkpoints/fire-equality-run18-epoch49.ckpt` | Best model (~4 MB) |
| `results/fairness_plot_cache.pkl` | Audit plot cache (~3.5 MB) |
| `results/audit/*.md` | Full text reports |
| `figures/manuscript/`, `figures/supplementary/` | Paper figures |

## Copied from private `FireEqual` (paths parameterised)

- `src/fire_equality/` — model & datamodule (from `code/fire_equality/`)
- `scripts/fairness_analysis.py`, `scripts/train.py` — from repo root
- `configs/` — from `code/fire_equality/conf/`
- `docs/REPRODUCTION.md`, `FEATURE_DATA_SOURCES.md`, etc.

## Intentionally excluded

- `dataset/` raw rasters, FireTracks HDF5, full 7× `.pth` shards
- `multirun/`, `outputs/`, `.cursor/`
- Orion submodule (attribution in `docs/ATTRIBUTION.md` only)

Original files under `E:\FireEqual\` were **not deleted or modified** for this release (except new sibling folder `large-fire-threshold-audit-review/`).
