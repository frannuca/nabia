# nabia

A C++20 numerical library for systematic, risk-budgeted portfolios, its Python bindings (packaged as wheels for
every platform), and a small Python project that collects financial news and scores it with an LLM on 20
economic factors.

**Documentation: [`doc/nabia.pdf`](doc/nabia.pdf)** (the library, its methods and the wheel packaging).

| Folder | Open in | Contents |
|---|---|---|
| [`cpp/`](cpp/README.md) | CLion | the C++ library, Catch2 tests, the command-line backtest, the pybind11 bindings with their tests, and the scripts that build and fetch the `nabia_python_bridge` wheels |
| [`python/`](python/README.md) | PyCharm | the news collector (RSS, GDELT GKG), the LLM scoring (Ollama; OpenAI and Anthropic with an API key) and the SQLite data layer |
| [`data/`](data/) | | `market.db`, the SQLite database the news and their scores are stored in |
| [`doc/`](doc/) | | `nabia.tex` → `nabia.pdf`, its chapters and build script |

## Quick start

```sh
# 1. C++: build and test (or open cpp/ in CLion)
cd cpp && cmake --preset debug && cmake --build --preset debug -j && ctest --preset debug && cd ..

# 2. the wheel for your Python version, tested (needs uv) -> dist/; or download the CI's: cpp/scripts/fetch_wheels.sh
cpp/scripts/build_wheels.sh 3.14

# 3. news: collect into data/market.db, then score with the local Ollama model
python/run.sh
```

On Windows: `cpp\scripts\build_wheels.ps1`. Wheels for every platform: `gh workflow run wheels.yml`; a version tag
(`git tag v0.2.0 && git push origin v0.2.0`) publishes a GitHub Release with all of them.

## Building the documentation

```sh
doc/build.sh             # doc/nabia.pdf (pdflatex), excerpts extracted from the current code
doc/build.sh --figures   # first rerun the pipeline for the figures and result tables
```
