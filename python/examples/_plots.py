"""Plotting for the examples (matplotlib).

Every figure is saved as PNG in examples/figures/ and, at the end of the script,
shown in a window when the backend is interactive. Set NABIA_NO_SHOW=1 (or
MPLBACKEND=Agg) to only save them, for example on a headless machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("NABIA_NO_SHOW"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

FIGURES = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "font.size": 9, "axes.titlesize": 10, "figure.titlesize": 12,
                     "date.converter": "concise"})


def figure(nrows: int = 1, ncols: int = 1, title: str = "", size: tuple[float, float] | None = None, **kwargs):
    """plt.subplots with a suptitle and a size that grows with the grid.

    Returns one Axes for a 1x1 grid, a 1-d array for a single row or column, else a 2-d array.
    """
    size = size or (5.0 * ncols + 1.0, 3.4 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False, **kwargs)
    if title:
        fig.suptitle(title, fontweight="bold")
    if nrows * ncols == 1:
        return fig, axes[0, 0]
    return fig, axes.ravel() if 1 in (nrows, ncols) else axes


def percent(ax, axis: str = "y", decimals: int = 0) -> None:
    """Formats fractions on an axis as percentages."""
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def save(fig, name: str) -> Path:
    """Saves to figures/<script>_<name>.png."""
    FIGURES.mkdir(exist_ok=True)
    script = Path(sys.argv[0]).stem or "example"
    path = FIGURES / f"{script}_{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[plot] {path.relative_to(FIGURES.parent)}")
    return path


def show() -> None:
    """Shows every open figure when the backend is interactive."""
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        plt.show()
    plt.close("all")
