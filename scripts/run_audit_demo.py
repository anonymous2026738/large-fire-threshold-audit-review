#!/usr/bin/env python
"""
最小可运行演示：验证 release 布局、缓存与（可选）微型样本 schema。

默认仅运行 figures-only 重绘（与 reproduce_figures.py 相同，便于 CI/无数据环境）。
加 --check-schema 时检查 data/processed/mini_sample.pth 与 schema.json。

用法:
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
