# Group-wise threshold audit — global ConvLSTM fire danger model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Licence notice.** Code in this repository is released under the [MIT License](LICENSE). Original third-party datasets are not redistributed and remain subject to their respective licences and access terms. Processed audit outputs provided in this repository are shared for research transparency and reproducibility, unless otherwise noted.

Code, documentation, and **processed audit outputs** supporting:

> **Group-wise Threshold Effects in a Global Deep Learning Fire Danger Prediction Model**  
> Anonymous Author, Affiliation withheld for double-anonymous review

This release is scoped for **manuscript reproducibility** (threshold audit, Fairlearn metrics, Integrated Gradients figures)—not a full geospatial data redistribution package.

## Repository contents

| Directory | Purpose |
|-----------|---------|
| `src/fire_equality/` | ConvLSTM model, datamodule, training loop |
| `scripts/` | Training, fairness audit, figure reproduction, mini demo |
| `configs/` | Hydra configs (incl. scenarios `run16`–`run19`) |
| `data/processed/` | Audit CSV tables, schema, `mini_sample.pth` |
| `results/` | Plot cache, audit reports, best checkpoint |
| `figures/` | Manuscript & supplementary PNGs |
| `docs/` | Reproduction workflow, feature sources, attribution |

## Top-level text files

| File | Purpose |
|------|---------|
| `.gitignore` | Excludes caches, local data products, logs, and other non-release artefacts. |
| `LICENSE` | MIT licence for repository source code. |
| `README.md` | Repository overview, quick start, reproduction summary, and citation note. |
| `environment.yml` | Conda environment specification for the audit workflow. |
| `requirements.txt` | Python package requirements for pip-based setup. |

## Quick start (minimal demo, no full dataset)

```bash
conda create -n fire-danger-audit python=3.10 -y
conda activate fire-danger-audit
pip install -r requirements.txt

# Optional: synthetic format check
python scripts/build_mini_sample.py
python scripts/run_audit_demo.py --check-schema

# Reproduce audit figures from published cache (~1 min, CPU)
python scripts/reproduce_figures.py
# Output: results/figures_reproduced/
```

The minimal demo and figure reproduction do not require the full `.pth` shards. Full retraining requires the external datasets and processed tensors described below.

## Full reproduction (training + audit)

1. **Obtain raw data** listed in `data/README.md` and build seven `.pth` shards (`docs/BUILD_PTH_COMMANDS.md`).
2. Place shards under `data/processed/` (filenames `processed_firetracks_pixel_binary_*.pth`).
3. **Train** (16-run grid + optional fine grid):

```bash
python scripts/train.py -m model.lr=0.0003,0.001 model.hidden_size=32,64 model.positive_weight=0.5,1.0 seed=42,123
python scripts/train.py -m scenario=run16,run17,run18,run19
```

4. **Audit** (replace checkpoint path with your best run):

```bash
python scripts/fairness_analysis.py \
  --checkpoint results/checkpoints/fire-equality-run18-epoch49.ckpt \
  --data data/processed/processed_firetracks_pixel_binary_2002-2004.pth \
        data/processed/processed_firetracks_pixel_binary_2005-2007.pth \
        ... \
  --output results/audit_full --xai
```

Published manuscript numbers match `results/audit/` and `data/processed/audit_*.csv`.

## Figures and tables

| Manuscript asset | Source in this repo |
|------------------|---------------------|
| Fig. 2a–b | `figures/manuscript/Fig2a_*.png`, `Fig2b_*.png` |
| Fig. 3–5 | `figures/manuscript/Fig3–5*.png` or `scripts/reproduce_figures.py` |
| Supp. ROC | `figures/supplementary/supplementary_Fig1*.png` |
| Group threshold table | `data/processed/audit_group_metrics_population_density.csv` |
| Fairness gaps | `data/processed/audit_fairness_summary.csv` |

## Data and code availability

- **Code**: MIT — see [LICENSE](LICENSE) and the licence notice above; upstream geospatial products are **not** covered by MIT.
- **Processed audit outputs**: tables, plot cache, checkpoint metadata — in `data/processed/` and `results/` (transparency/reproducibility; not a substitute for third-party data licences).
- **Full training tensors & raw rasters**: not shipped; see [`data/README.md`](data/README.md).
- **Zenodo archive**: DOI withheld for double-anonymous review

## Citation

```bibtex
@software{anonymous2026firethresholdaudit,
  author       = {Anonymous Author},
  title        = {Group-wise Threshold Effects in a Global Deep Learning Fire Danger Prediction Model: Code and Audit Outputs},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {DOI withheld for double-anonymous review},
  url          = {https://github.com/anonymous2026738/large-fire-threshold-audit-review}
}
```

## Contact

Contact details are withheld for double-anonymous review. For full `.pth` shards or replication questions, contact information will be provided after peer review or to the editorial office on request.
