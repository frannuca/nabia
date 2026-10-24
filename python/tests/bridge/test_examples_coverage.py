"""The examples use every name the bridge exports (functions, classes, fields, enum values)."""

import pathlib
import re

import nabia_python_bridge as nb

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
MODULES = ["stats", "signal", "forecast", "optimization", "risk", "strategy", "regression", "frames"]


def unused_names() -> list[str]:
    text = "\n".join(p.read_text() for p in EXAMPLES.glob("0*.py"))
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    missing = []
    for modname in MODULES:
        mod = getattr(nb, modname)
        for name in mod.__all__:
            obj = getattr(mod, name)
            if name not in words:
                missing.append(f"{modname}.{name}")
            if not (isinstance(obj, type) and obj.__module__.startswith("nabia_python_bridge._core")):
                continue
            for member in vars(obj):
                if member in ("__add__", "__mul__"):
                    used = re.search(r"Kernel\([^)]*\)\s*[+*]|\bse \+|\bper \*", text) is not None
                elif member == "__call__":
                    used = re.search(r"\b(k|se|per|kernel)\(", text) is not None
                elif member.startswith("_") or member in ("name", "value"):
                    continue
                else:
                    used = member in words
                if not used:
                    missing.append(f"{modname}.{name}.{member}")
    return missing


def test_examples_cover_the_whole_api():
    assert unused_names() == []
