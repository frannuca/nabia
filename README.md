# nabia

A C++20 numerical library for signal processing and time-series forecasting —
built as a learning platform for modern C++ and CMake.

## Layout

```
nabia/
├── CMakeLists.txt          # root build: library target, options, subprojects
├── CMakePresets.json       # debug / release / asan build configurations
├── cmake/                  # reusable CMake modules (CompilerWarnings.cmake)
├── include/nabia/          # PUBLIC headers  (what users #include)
│   ├── stats/              #   descriptive statistics
│   ├── signal/             #   smoothing, differencing
│   └── forecast/           #   forecasting models
├── src/                    # PRIVATE implementation (.cpp)
├── tests/                  # Catch2 unit tests (fetched automatically)
└── examples/               # small programs using the library
```

## Build & test

```sh
cmake --preset debug            # configure  -> build/debug
cmake --build --preset debug    # compile
ctest --preset debug            # run tests
./build/debug/examples/forecast_demo
```

Other presets: `release` (optimised), `asan` (Address + UB sanitizers — use it
whenever you touch memory-sensitive code).

The build also compiles the LaTeX documentation when `pdflatex` is available
(turn it off with `-DNABIA_BUILD_DOCS=OFF`):
`build/debug/src/optimization/convex/docs/convex_optimization.pdf`.

## What's in it

| Module | Functions |
|---|---|
| `nabia::stats` | `mean`, `variance`, `stddev`, `autocorrelation`, `normal_pdf`, `normal_cdf`, `normal_quantile` |
| `nabia::signal` | `moving_average`, `exponential_moving_average`, `diff`; filters behind `IFilter<T>`: `ButterworthBandPass`, `EWMA` (wrapping iir1) |
| `nabia::forecast` | `SimpleExponentialSmoothing` |
| `optimization/convex` | `LBFGS`, `AugmentedLagrangian`, `SQP`, `LinearConstraint`, `BoxConstraints` |
| `optimization/nongradopt` | `NelderMead` (box-constrained), `DifferentialEvolution` (multithreaded population evaluation) |
| `risk` | `Volatility`, `ValueAtRisk`, `ExpectedShortfall` (Gaussian, with Euler risk contributions), `RiskBudgeting<Measure>` |

## Adding a new feature (the workflow)

1. Declare it in a header under `include/nabia/<module>/`.
2. Implement it in `src/<module>/` and add the `.cpp` to `add_library(nabia ...)`.
3. Write tests in `tests/` and add the file to `add_executable(nabia_tests ...)`.
4. `cmake --build --preset debug && ctest --preset debug`.

## Learning roadmap (suggested)

**Signals**
- [ ] Weighted / centered moving averages
- [ ] Convolution and FIR filters
- [ ] Discrete Fourier Transform (naive O(n²), then radix-2 FFT)
- [ ] Butterworth / IIR filters

**Forecasting**
- [ ] Holt's linear trend (double exponential smoothing)
- [ ] Holt-Winters (seasonal)
- [ ] Optimise `alpha` automatically (golden-section search on SSE)
- [ ] AR(p) via Yule-Walker, then ARIMA
- [ ] Error metrics: MAE, RMSE, MAPE; train/test split

**C++ topics along the way**
- [ ] Templates & concepts (`std::floating_point`) to support `float` and `double`
- [ ] A common `Forecaster` interface (virtual vs. CRTP vs. concepts)
- [ ] Ranges and `std::views`
- [ ] Benchmarks with Google Benchmark
- [ ] `constexpr` where possible

**CMake topics along the way**
- [ ] `install()` + `export()` so other projects can `find_package(nabia)`
- [ ] Shared vs. static library (`BUILD_SHARED_LIBS`, symbol visibility)
- [ ] Optional dependency (e.g. Eigen) behind an option
- [ ] CI with GitHub Actions
- [ ] Code coverage and `clang-tidy` integration
