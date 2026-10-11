"""Load config.yaml once and resolve paths against the project root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _find_root() -> Path:
    """Project root: $COMPLIANCE_AI_ROOT, else the nearest parent folder holding config.yaml."""
    if os.environ.get("COMPLIANCE_AI_ROOT"):
        return Path(os.environ["COMPLIANCE_AI_ROOT"])
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists():
            return parent
    return here.parents[2]


ROOT = _find_root()


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str | Path) -> Path:
    """Turn a config path (relative to the project root) into an absolute path."""
    p = Path(rel_path)
    return p if p.is_absolute() else ROOT / p
