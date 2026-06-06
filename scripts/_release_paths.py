"""Path helpers for the release repository."""
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parent.parent


def p(*parts: str) -> Path:
    return RELEASE_ROOT.joinpath(*parts)
