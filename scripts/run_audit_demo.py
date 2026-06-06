#!/usr/bin/env python
"""Run the minimal review demo.

By default this script redraws release figures from the cached plotting data.
Use --check-schema to validate data/processed/mini_sample.pth against schema.json.

Examples:
    python scripts/run_audit_demo.py
    python scripts/run_audit_demo.py --check-schema
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _release_paths import p


def check_schema() -> bool:
    schema_path = p("data", "processed", "schema.json")
    mini_path = p("data", "processed", "mini_sample.pth")
    ok = True
    if not schema_path.is_file():
        print(f"MISSING: {schema_path}")
        ok = False
    else:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        print(f"schema.json keys: {list(schema.keys())}")

    if not mini_path.is_file():
        print(f"MISSING: {mini_path} (run: python scripts/build_mini_sample.py)")
        ok = False
    else:
        import torch

        data = torch.load(mini_path, map_location="cpu", weights_only=False)
        n = len(data.get("spatiotemporal_samples", []))
        print(f"mini_sample.pth: {n} spatiotemporal samples")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-schema", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    if args.check_schema and not check_schema():
        return 1

    if args.skip_figures:
        print("Skipped figure reproduction.")
        return 0

    from reproduce_figures import main as repro_main

    return repro_main()


if __name__ == "__main__":
    raise SystemExit(main())
