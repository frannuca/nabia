#include "bridge.hpp"

#include <nabia/signal/ButterworthBandPass.h>
#include <nabia/signal/EWMA.h>
#include <nabia/signal/smoothing.hpp>

using Eigen::VectorXd;

namespace nabia_bridge {

namespace {

/// The members shared by every IFilter<Params>: process, reset, apply, parameters.
template <typename Filter, typename Params>
void bind_filter(py::class_<Filter>& c) {
    c.def(py::init([](const Params& p) { return std::make_unique<Filter>(std::make_unique<Params>(p)); }),
          py::arg("parameters"))
        .def(py::init([](const py::kwargs& kw) { return std::make_unique<Filter>(from_kwargs<Params>(kw)); }),
             "Construct from the parameter fields as keyword arguments.")
        .def("process", &Filter::process, py::arg("sample"), "Filters one sample, advancing the internal state.")
        .def("reset", &Filter::reset, "Clears the internal state.")
        .def(
            "apply",
            [](const Filter& f, VectorXd signal) {
                f.apply(signal);
                return signal;
            },
            py::arg("signal"), "Filters a whole series from a fresh state (the internal state is not used).")
        .def_property_readonly("parameters", [](const Filter& f) { return f.parameters(); });
}

} // namespace

void bind_signal(py::module_& m) {
    namespace sg = nabia::signal;
    m.def("moving_average", [](const DoubleArray& x, std::size_t window) { return to_array(sg::moving_average(as_span(x), window)); },
          py::arg("x"), py::arg("window"), "Trailing simple moving average (output has len(x) - window + 1 points).");
    m.def("exponential_moving_average",
          [](const DoubleArray& x, double alpha) { return to_array(sg::exponential_moving_average(as_span(x), alpha)); },
          py::arg("x"), py::arg("alpha"), "Exponential moving average with smoothing factor alpha in (0, 1].");
    m.def("diff", [](const DoubleArray& x, std::size_t order) { return to_array(sg::diff(as_span(x), order)); },
          py::arg("x"), py::arg("order") = 1, "Differences of the given order.");

    py::class_<EWMAParams>(m, "EWMAParams")
        .def(kwargs_init<EWMAParams>())
        .def_readwrite("alpha", &EWMAParams::alpha)
        .def_readwrite("seed_with_first_sample", &EWMAParams::seed_with_first_sample);
    py::class_<EWMA> ewma(m, "EWMA", "Exponentially weighted moving average: y[t] = alpha x[t] + (1 - alpha) y[t-1].");
    bind_filter<EWMA, EWMAParams>(ewma);
    ewma.def_static("alpha_from_span", &EWMA::alpha_from_span, py::arg("span"))
        .def_static("alpha_from_halflife", &EWMA::alpha_from_halflife, py::arg("halflife"));

    py::class_<ButterworthBandPassParams>(m, "ButterworthBandPassParams")
        .def(kwargs_init<ButterworthBandPassParams>())
        .def_readwrite("order", &ButterworthBandPassParams::order)
        .def_readwrite("sample_rate", &ButterworthBandPassParams::sample_rate)
        .def_readwrite("low_cutoff", &ButterworthBandPassParams::low_cutoff)
        .def_readwrite("high_cutoff", &ButterworthBandPassParams::high_cutoff);
    py::class_<ButterworthBandPass> bp(m, "ButterworthBandPass", "Causal Butterworth band-pass filter.");
    bind_filter<ButterworthBandPass, ButterworthBandPassParams>(bp);
    bp.def_property_readonly_static("max_order", [](const py::object&) { return ButterworthBandPass::max_order; });
}

} // namespace nabia_bridge
