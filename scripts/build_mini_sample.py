#!/usr/bin/env python
""" .pth  schema.json,()."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from _release_paths import p

T, H, W, C = 10, 25, 25, 8
N = 12


def make_sample(idx: int, label: int) -> dict:
    rng = np.random.default_rng(idx + label)
    features = rng.standard_normal((T, H, W, C)).astype(np.float32)
    return {
        "features": features,
        "target": int(label),
        "metadata": {
            "lat": float(-30 + idx * 3),
            "lon": float(10 + idx * 5),
            "year": 2010 + (idx % 5),
            "population_density_group": ["Low", "Medium", "High"][idx % 3],
            "demo": True,
        },
    }


def main() -> None:
    samples = [make_sample(i, i % 2) for i in range(N)]
    payload = {
        "spatiotemporal_samples": samples,
        "config": {
            "patch_size_km": 25,
            "time_steps": T,
            "target_years": (2002, 2020),
            "target_type": "binary_classification",
            "channels": ["FWI", "VPD", "NDVI", "population", "GDP", "land_cover", "max_temp", "max_wind"],
        },
        "metadata": {"n_samples": N, "purpose": "format_demo_only"},
    }
    out_pth = p("data", "processed", "mini_sample.pth")
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payloading, out_pth)

    schema = {
        "file_format": "PyTorch .pth dict",
        "required_top_level_keys": ["spatiotemporal_samples", "config"],
        "sample_keys": ["features", "target", "metadata"],
        "features_shape": "[T, H, W, C] with T=10, H=W=25, C=8",
        "dataloader_tensor_shape": "[C, T, H, W] after FireTracksDataset",
        "target": "integer 0/1 for binary_classification",
        "full_dataset": "Seven shards processed_firetracks_pixel_binary_YYYY-YYYY.pth (not redistributed; see README.md)",
    }
    schema_path = p("data", "processed", "schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote {out_pth} ({N} samples)")
    print(f"Wrote {schema_path}")


if __name__ == "__main__":
    main()
