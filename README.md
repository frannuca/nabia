# nabia

A platform for researching and running **systematic trend strategies** across equity indices, bond
futures, commodities and crypto: a C++20 numerical library, a Python package on top of it, and a web
application with AI agents that decide the risk budgets.

The repository holds three independent projects. Each opens on its own in an IDE, has its own
environment, tests and documentation (`doc/`, LaTeX sources with their PDFs), and depends on the
others only through installed packages:

| Project | Open in | Contents | Documentation |
|---|---|---|---|
| [`cpp/`](cpp/README.md) | CLion | The C++ library (`include`, `src`), Catch2 tests, the command-line backtest, and the pybind11 bindings that build the `nabia_python_bridge` wheel | [`cpp/doc/nabia_cpp_reference.pdf`](cpp/doc/nabia_cpp_reference.pdf) and theory notes in `cpp/doc/` |
| [`python/`](python/README.md) | PyCharm | The `nabia` package (backtests, analytics, market data) on the wheel, examples with plots, and the bindings' tests | [`python/doc/nabia_python_reference.pdf`](python/doc/nabia_python_reference.pdf) |
| [`webui/`](webui/README.md) | PyCharm | `nabia_webui`: the agents, their MCP servers, and the web app (FastAPI back end, JavaScript front end), on the `nabia` package | [`webui/doc/nabia_webui_reference.pdf`](webui/doc/nabia_webui_reference.pdf) |

```
cpp/  ──builds──▶  dist/nabia_python_bridge-*.whl  ──installed by──▶  python/.venv
                                                    ──installed by──▶  webui/.venv  (+ python/ editable)
data/market.db   shared market database (python/ and webui/)
```

## Build and run

Commands run from the repository root unless a step says `cd`. The order matters: each project installs
what the previous one builds.

### 1. C++ project — `cpp/` (CLion)

Needs CMake ≥ 3.25 and a C++20 compiler (pdflatex for the PDFs). Eigen, iir1, Catch2, nlohmann/json and
pybind11 are found on the system or fetched.

```sh
cd cpp
cmake --preset debug                    # configure (once, or after editing CMakeLists)
cmake --build --preset debug -j         # library, tests, command-line app, examples, PDFs
ctest --preset debug                    # the test suite
cmake --build --preset debug --target nabia_docs   # only the PDFs of cpp/doc
```

Presets: `debug`, `release`, `releaseWithDebugInfo`, `asan`. Running the command-line app:

```sh
./build/debug/apps/rb_backtest/nabia_rb_backtest \
    --prices apps/rb_backtest/example/prices.csv \
    --config apps/rb_backtest/example/portfolio.json \
    --output /tmp/result.csv
```

In CLion: open `cpp/`. CLion reads `CMakePresets.json`; enable the `debug` preset (Settings → Build →
CMake), then pick the `nabia_tests` or `nabia_rb_backtest` target and Run or Debug.

### 2. The wheel — from C++ to Python

Needs [uv](https://docs.astral.sh/uv/). Builds one wheel per Python version into `dist/`, and tests each one:

```sh
cpp/scripts/build_wheels.sh 3.14        # the Python version of your .venv (no argument: 3.10–3.14)
NABIA_WHEEL_BUILD_TYPE=RelWithDebInfo cpp/scripts/build_wheels.sh 3.14   # with debug symbols
```

Rebuild the wheel whenever the C++ code changes, then run steps 3 and 4 again.

On Windows (Intel/AMD 64-bit), in PowerShell, with Visual Studio 2022 Build Tools ("Desktop
development with C++"), Git and Python from python.org installed:

```powershell
cpp\scripts\build_wheels.ps1                       # cibuildwheel: CPython 3.10–3.14, each tested -> dist\*-win_amd64.whl
cpp\scripts\build_wheels.ps1 -Versions 3.12,3.13    # only these versions
python\scripts\setup_env.ps1                        # then step 3 with python\.venv\Scripts\python.exe
webui\scripts\setup_env.ps1                         # and step 4 with webui\.venv\Scripts\python.exe
```

### 3. Python project — `python/` (PyCharm)

```sh
python/scripts/setup_env.sh             # python/.venv: the wheel + nabia (editable) + extras
cd python
.venv/bin/python -m pytest              # bridge tests and package tests
.venv/bin/python examples/01_factor_replication.py             # 01 … 06, each plots its results
NABIA_NO_SHOW=1 .venv/bin/python examples/05_forecasting.py    # save the figures only
.venv/bin/python -m nabia.marketdata --db sqlite:///../data/market.db --start 2010-01-01   # download data
doc/build.sh                            # python/doc/nabia_python_reference.pdf
```

In PyCharm: open `python/` and select `python/.venv/bin/python` as the interpreter (Settings → Python
Interpreter → Add → Existing). Examples and tests then run and debug directly.

### 4. Web UI — `webui/` (PyCharm)

```sh
webui/scripts/setup_env.sh              # webui/.venv: the wheel + python/ (editable) + nabia_webui
cd webui
.venv/bin/python -m pytest              # the web UI tests
.venv/bin/python -m nabia_webui.web --config ai_config.json      # http://127.0.0.1:8000
.venv/bin/python -m nabia_webui.ai daily --config ai_config.json # one daily decision from the command line
doc/build.sh                            # webui/doc/nabia_webui_reference.pdf
```

For the agents, start the model first: `ollama serve` for Ollama, or set `ANTHROPIC_API_KEY` for Claude.
The trend-rule policy needs neither.

In PyCharm: open `webui/` and select `webui/.venv/bin/python` as the interpreter. Run configuration for
the web app: module `nabia_webui.web`, parameters `--config ai_config.json`, working directory `webui/`
(the database path in `ai_config.json` is relative to it).

### Debugging C++ called from Python

1. Build the wheel with debug symbols (step 2 with `NABIA_WHEEL_BUILD_TYPE=RelWithDebInfo`), then run
   `python/scripts/setup_env.sh` again.
2. Start the script in PyCharm with a breakpoint before the C++ call.
3. In CLion: Run → Attach to Process, and pick that Python process. Breakpoints in `cpp/src` are hit.

Without rebuilding the wheel: `cd cpp && cmake --preset python-bridge && cmake --build --preset python-bridge`
builds the modules next to the package sources; run the script with `PYTHONPATH=<repo>/cpp/bindings`.

## Wheels for every version and platform (GitHub Actions)

`.github/workflows/wheels.yml` builds and tests the wheels for CPython 3.10–3.14 on Linux (x86_64,
aarch64), macOS (arm64, x86_64) and Windows (AMD64) with cibuildwheel (configured in
`cpp/pyproject.toml`):

```sh
gh workflow run wheels.yml                                  # start a build
gh run list --workflow wheels.yml                           # follow it
gh run download <run-id> -n all-wheels -D dist              # every wheel and the sdist, into dist/
```

On the run's page (Actions tab → wheels → the run), the **Artifacts** section has one zip per platform
and `all-wheels` with everything. Pushing a version tag also publishes a GitHub Release with all the
wheels attached:

```sh
git tag v0.1.0 && git push origin v0.1.0   # -> https://github.com/frannuca/nabia/releases
```

Locally, `cpp/scripts/build_wheels.sh` (macOS, Linux) and `cpp/scripts/build_wheels.ps1` (Windows)
build the wheels of the current machine. See section 14.5 of the C++ reference, and the tutorial
[`cpp/doc/packaging/wheels_and_github_actions.pdf`](cpp/doc/packaging/wheels_and_github_actions.pdf)
for how wheels, cibuildwheel and the workflow's jobs work and how to write your own.

## Documentation

Each project's `doc/` folder has its LaTeX sources and the built PDFs:

- `cpp/doc`: the C++ reference (architecture, every method stated mathematically with proofs, the
  command-line application, the bindings and the wheels) and theory notes on exponential smoothing, the
  Kalman filter, Gaussian processes, convex optimization and elastic net regression. Built by the
  `nabia_docs` CMake target.
- `python/doc`: the Python reference (the library from Python, backtests, analytics with their
  formulas, market data, the worked examples). Built by `python/doc/build.sh`.
- `webui/doc`: the web UI reference (agents and their complete prompts, every MCP tool, the agent loop,
  decisions to positions, news sentiment, the web app). Built by `webui/doc/build.sh`.

Code excerpts, prompts and tool descriptions are extracted from the sources at build time, so the
documents always show the current code.
