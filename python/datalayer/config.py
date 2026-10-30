"""The run configuration (config/config.json) and the factor list it points to (config/factors.json). Relative paths
in the configuration are relative to the configuration file's folder."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "config" / "config.json"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or DEFAULT).resolve()
    cfg = json.loads(path.read_text())
    cfg["db"] = str(path.parent / cfg["db"])
    cfg["factors"] = json.loads((path.parent / cfg["factors"]).read_text())["factors"]
    return cfg
