"""Helpers to expose the extension's submodules as plain Python modules."""

from __future__ import annotations

from typing import Any

import numpy as np

_SKIP = {"__init__", "__new__"}


def _field_names(cls: type) -> list[str]:
    names: list[str] = []
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, property) and not name.startswith("_") and name not in names:
                names.append(name)
    return names


def _short(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size <= 6:
            return np.array2string(value, precision=6, separator=", ")
        return f"array(shape={value.shape})"
    if isinstance(value, list) and len(value) > 6:
        return f"[{len(value)} items]"
    return repr(value)


def _repr(self: Any) -> str:
    parts = []
    for name in _field_names(type(self)):
        try:
            parts.append(f"{name}={_short(getattr(self, name))}")
        except Exception:  # e.g. a property that needs a fitted model
            continue
    return f"{type(self).__name__}({', '.join(parts)})"


# Plain data: parameters, options and results. Models and solvers keep the default repr.
_DATA_SUFFIXES = ("Params", "Options", "Result", "Fit", "Evaluation", "Summary", "Curve", "Computation",
                  "Metrics", "Constraints", "Constraint", "Forecast")


def export(module: Any, namespace: dict[str, Any]) -> list[str]:
    """Copies the public names of an extension submodule into `namespace`."""
    names = [n for n in dir(module) if not n.startswith("_")]
    for name in names:
        obj = getattr(module, name)
        if isinstance(obj, type) and name.endswith(_DATA_SUFFIXES) and "__repr__" not in vars(obj):
            obj.__repr__ = _repr
        namespace[name] = obj
    return names
