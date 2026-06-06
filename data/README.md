# Data availability

This repository supports reproducibility of the **group-wise threshold audit** for the manuscript *Group-wise Threshold Effects in a Global Deep Learning Fire Danger Prediction Model*. **Raw and full processed training tensors are not redistributed** here unless the upstream licence explicitly permits republication.

## What is included in this release

| Path | Description |
|------|-------------|
| `processed/schema.json` | Schema for `.pth` spatiotemporal cubes |
| `processed/mini_sample.pth` | 12-sample synthetic demo (format only) |
| `processed/audit_group_metrics_*.csv` | Group metrics at optimized thresholds (manuscript tables) |
| `processed/audit_fairness_summary.csv` | Equalized odds / demographic parity by grouping |
| `../results/fairness_plot_cache.pkl` | Held-out predictions + group labels for figure reproduction |
| `../results/audit/*.md` | Full fairness text reports |
| `../results/checkpoints/fire-equality-run18-epoch49.ckpt` | Best checkpoint (~4 MB) for re-running audit |

## What is **not** included (obtain separately)

- Seven shards `processed_firetracks_pixel_binary_2002-2004.pth` … `2019-2020.pth` (~full 2002–2020 sample)
- FireTracks HDF5, ERA5-Land NetCDF, FWI/VPD rasters, GIMMS3G+ NDVI, WorldPop GeoTIFF, MODIS MCD12Q1 tiles, World Bank GDP tables

Rebuild instructions: `docs/BUILD_PTH_COMMANDS.md` (after placing raw data under a local `data/raw/` tree mirroring the sources below).

## Upstream data sources

| Product | Role in model | Access |
|---------|---------------|--------|
| **FireTracks** | Fire event labels & spatiotemporal patches | Scientific dataset — obtain per [FireTracks data policy](https://www.firetracks.org/) / publication terms |
| **FWI** (Fire Weather Index) | Channel 0 | Public archives; annual NetCDF used locally — see project `dataset/FWI/download_from_urls.sh` in private build tree |
| **VPD** | Channel 1 | Derived from meteorological archives (annual NetCDF) |
| **GIMMS3G+ NDVI** | Channel 2 | https://ecocast.arc.nasa.gov/data/pub/gimms/3g.v1/ |
| **WorldPop** | Channel 3 (population density) | https://www.worldpop.org/ — `{year}` 1 km aggregated GeoTIFF |
| **World Bank / covariate GDP** | Channel 4 | World Development Indicators & harmonised country-year tables (see manuscript) |
| **MODIS MCD12Q1** | Channel 5 (land cover) | https://lpdaac.usgs.gov/products/mcd12q1v061/ — local GeoTIFF or GEE export |
| **ERA5-Land** | Channels 6–7 (max 2 m temperature, max 10 m wind) | https://cds.climate.copernicus.eu/ (Copernicus CDS; registration required) |

Detailed channel mapping: `docs/FEATURE_DATA_SOURCES.md`.

## Restricted / upon reasonable request

- **Full processed `.pth` shards** may be shared upon reasonable request to the corresponding author for non-commercial replication, subject to FireTracks and raster redistribution rules.
- **Author-processed audit tensors** beyond the published CSV/cache are not required to reproduce main figures if `results/fairness_plot_cache.pkl` is used with `scripts/reproduce_figures.py`.

## Licence note for data outputs

- **Code in this repository**: MIT (`LICENSE`) — applies to source code only.
- **Original third-party datasets**: not redistributed here; each product remains under its own licence and access terms (see table above).
- **Processed audit tables & plot cache in this repo**: shared for research transparency and reproducibility; recommended CC-BY-4.0 where applicable for author-derived tables/figures; **upstream rasters retain their original licences** and must not be republished without permission.
