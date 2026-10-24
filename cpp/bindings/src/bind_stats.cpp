#include "bridge.hpp"

#include <nabia/stats/descriptive.hpp>
#include <nabia/stats/normal.hpp>

namespace nabia_bridge {

void bind_stats(py::module_& m) {
    namespace st = nabia::stats;
    m.def("mean", [](const DoubleArray& x) { return st::mean(as_span(x)); }, py::arg("x"),
          "Arithmetic mean. Raises ValueError if x is empty.");
    m.def("variance", [](const DoubleArray& x, std::size_t ddof) { return st::variance(as_span(x), ddof); },
          py::arg("x"), py::arg("ddof") = 1,
          "Variance; ddof = delta degrees of freedom (0 = population, 1 = sample).");
    m.def("stddev", [](const DoubleArray& x, std::size_t ddof) { return st::stddev(as_span(x), ddof); },
          py::arg("x"), py::arg("ddof") = 1, "Standard deviation, sqrt(variance(x, ddof)).");
    m.def("autocorrelation", [](const DoubleArray& x, std::size_t lag) { return st::autocorrelation(as_span(x), lag); },
          py::arg("x"), py::arg("lag"), "Sample autocorrelation at `lag` (biased estimator, as in ACF plots).");

    // Vectorized: scalars give scalars, arrays give arrays.
    m.def("normal_pdf", py::vectorize(&st::normal_pdf), py::arg("x"), "Standard normal density.");
    m.def("normal_cdf", py::vectorize(&st::normal_cdf), py::arg("x"), "Standard normal CDF.");
    m.def("normal_quantile", py::vectorize(&st::normal_quantile), py::arg("p"),
          "Standard normal quantile (inverse CDF), p in (0, 1).");
}

} // namespace nabia_bridge
