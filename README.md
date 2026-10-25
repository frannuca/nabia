# nabia

A platform for researching systematic, risk-budgeted portfolios: a C++20 numerical library, its Python bindings
(packaged as wheels for every platform), and a Python package whose centrepiece is **sentiment-modulated risk
budgeting**: LLM news sentiment and market indicators, combined per asset by an elastic net into a score that
modulates risk budgets (long-only) or decides the side of each position (long-short), backtested against plain ERC.

**Documentation: [`doc/nabia.pdf`](doc/nabia.pdf)**, a single document with the user guide, the architecture,
every method with its mathematics and proofs, the derivations behind the forecasters, optimizers and elastic net,
the bindings and wheel packaging (with a GitHub Actions tutorial), the Python package, and the sentiment pipeline,
with code excerpts and `file:line` pointers extracted from the source when it is built.

| Folder | Open in | Contents |
|---|---|---|
| [`cpp/`](cpp/README.md) | CLion | the C++ library, Catch2 tests, the command-line backtest, and the pybind11 bindings that build the `nabia_python_bridge` wheel |
| [`nabia_python/`](nabia_python/README.md) | PyCharm | the Python project: the `nabia` package (backtests, analytics, market data, the `nabia.sentiment_rb` pipeline), its tests, its `pyproject.toml` and `.venv` |
| [`doc/`](doc/) | | `nabia.tex` → `nabia.pdf`, its chapters and build script; the pipeline's specification |

```
cpp/  ──builds──▶  dist/nabia_python_bridge-*.whl  ──installed into──▶  nabia_python/.venv
```

## Quick start

```sh
# 1. C++: build and test (or open cpp/ in CLion)
cd cpp && cmake --preset debug && cmake --build --preset debug -j && ctest --preset debug && cd ..

# 2. the wheel for your Python version (needs uv); or download the CI's: nabia_python/scripts/fetch_wheels.sh
cpp/scripts/build_wheels.sh 3.14

# 3. the Python project: .venv with the wheel and nabia, then the tests
nabia_python/scripts/setup_env.sh
cd nabia_python && .venv/bin/python -m pytest

# 4. the sentiment risk-budgeting pipeline (offline: synthetic data, mocked LLM)
.venv/bin/python -m nabia.sentiment_rb                                             # long-only
.venv/bin/python -m nabia.sentiment_rb --config <(echo '{"budgets": {"mode": "long_short"}}')
.venv/bin/python -m nabia.sentiment_rb --ic-sweep 0 0.1 0.2 0.3 0.5                # alpha against news quality
```

On Windows: `cpp\scripts\build_wheels.ps1` and `nabia_python\scripts\setup_env.ps1`. Wheels for every platform:
`gh workflow run wheels.yml`; a version tag (`git tag v0.2.0 && git push origin v0.2.0`) publishes a GitHub
Release with all of them. Chapter 1 of the PDF has every command, including debugging the C++ code from Python.

## Building the documentation

```sh
doc/build.sh             # doc/nabia.pdf (pdflatex), excerpts extracted from the current code
doc/build.sh --figures   # first rerun the pipeline for the figures and result tables
```
