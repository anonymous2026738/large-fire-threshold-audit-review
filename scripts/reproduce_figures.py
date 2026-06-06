#!/usr/bin/env python
"""
从已发布的 fairness_plot_cache.pkl 重绘论文审计图（无需 GPU / 完整 .pth 数据）。

用法（在仓库根目录）:
  python scripts/reproduce_figures.py
  python scripts/reproduce_figures.py --output results/figures_reproduced
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _release_paths import RELEASE_ROOT, p


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce audit figures from plot cache.")
    parser.add_argument(
        "--cache",
        default=str(p("results", "fairness_plot_cache.pkl")),
        help="Path to fairness_plot_cache.pkl",
    )
    parser.add_argument(
        "--output",
        default=str(p("results", "figures_reproduced")),
        help="Output directory for regenerated figures",
    )
    args = parser.parse_args()

    cache = Path(args.cache)
    if not cache.is_file():
        print(f"ERROR: cache not found: {cache}")
        return 1

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(p("scripts", "fairness_analysis.py")),
        "--figures-only",
        "--output",
        str(out),
        "--cache",
        str(cache),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(RELEASE_ROOT))
    if proc.returncode == 0:
        print(f"OK: figures written under {out}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
