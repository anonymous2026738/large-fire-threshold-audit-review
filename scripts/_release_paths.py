"""Release 仓库路径工具（避免硬编码本地绝对路径）。"""
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parent.parent


def p(*parts: str) -> Path:
    return RELEASE_ROOT.joinpath(*parts)
