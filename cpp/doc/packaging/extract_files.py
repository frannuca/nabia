"""Copy the packaging files into the build tree for wheels_and_github_actions.tex.

Run by the documentation build (doc/CMakeLists.txt) before pdflatex, so the tutorial shows the
files as they are:

    python3 extract_files.py <cpp folder> <output directory>

Each file is split into the pieces the tutorial explains one at a time:
  wheels.yml      workflow-head.txt (name and triggers) and job-<name>.txt per job;
  pyproject.toml  pyproject-<table>.txt per [table] (dots become dashes);
  and whole files: bindings-cmake.txt, build-wheels-sh.txt, build-wheels-ps1.txt, setup-env-sh.txt.
A missing file or section stops the build, so the tutorial cannot silently fall out of date.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Sections the tutorial \lstinputlisting's; extraction fails if one disappears.
REQUIRED = [
    "workflow-head", "job-wheels", "job-collect", "job-release", "job-windows-script",
    "pyproject-build-system", "pyproject-project", "pyproject-project-optional-dependencies",
    "pyproject-tool-scikit-build",
    "pyproject-tool-scikit-build-cmake-define", "pyproject-tool-cibuildwheel",
    "pyproject-tool-cibuildwheel-linux", "pyproject-tool-cibuildwheel-macos",
    "bindings-cmake", "build-wheels-sh", "build-wheels-ps1", "setup-env-sh",
]


def split_workflow(text: str) -> dict[str, str]:
    """Head (everything before the first job) and one piece per job under `jobs:`."""
    lines = text.splitlines()
    out: dict[str, list[str]] = {"workflow-head": []}
    current = "workflow-head"
    in_jobs = False
    for line in lines:
        if line.rstrip() == "jobs:":
            in_jobs = True
            continue
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line) if in_jobs else None
        if m:
            current = f"job-{m.group(1)}"
            out[current] = []
        out[current].append(line)
    return {k: "\n".join(v).strip("\n") + "\n" for k, v in out.items()}


def split_toml(text: str) -> dict[str, str]:
    """One piece per [table], with the comment lines just above it."""
    out: dict[str, list[str]] = {}
    pending: list[str] = []
    current = None
    for line in text.splitlines():
        m = re.match(r"^\[([^\]]+)\]\s*$", line)
        if m:
            current = "pyproject-" + m.group(1).replace(".", "-")
            out[current] = pending + [line]
            pending = []
        elif current is None or (line.startswith("#") and not out[current][-1].strip()):
            pending.append(line)  # a comment block introducing the next table
        else:
            if pending:
                out[current].extend(pending)
                pending = []
            out[current].append(line)
    return {k: "\n".join(v).strip("\n") + "\n" for k, v in out.items()}


_ASCII = {"\u2013": "-", "\u2014": "--", "\u2026": "...", "\u2192": "->", "\u2190": "<-",
          "\u2265": ">=", "\u2264": "<=", "\u00d7": "x", "\u2018": "'", "\u2019": "'",
          "\u201c": '"', "\u201d": '"'}


def _ascii(text: str) -> str:
    """pdflatex listings need ASCII: replace the few typographic characters used in comments."""
    text = "".join(_ASCII.get(c, c) for c in text)
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        sys.exit(f"extract_files: non-ASCII characters to map: {bad}")
    return text


def main(cpp: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    root = cpp.parent
    pieces: dict[str, str] = {}
    pieces.update(split_workflow((root / ".github/workflows/wheels.yml").read_text()))
    pieces.update(split_toml((cpp / "pyproject.toml").read_text()))
    pieces["bindings-cmake"] = (cpp / "bindings/CMakeLists.txt").read_text()
    pieces["build-wheels-sh"] = (cpp / "scripts/build_wheels.sh").read_text()
    pieces["build-wheels-ps1"] = (cpp / "scripts/build_wheels.ps1").read_text()
    pieces["setup-env-sh"] = (root / "python/scripts/setup_env.sh").read_text()
    missing = [name for name in REQUIRED if name not in pieces]
    if missing:
        sys.exit(f"extract_files: sections not found: {missing}")
    for name, text in pieces.items():
        (out / f"{name}.txt").write_text(_ascii(text))
    print(f"extracted {len(pieces)} pieces into {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
