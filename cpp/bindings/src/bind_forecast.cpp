#include "bridge.hpp"

#include <nabia/forecast/GaussianProcess.h>
#include <nabia/forecast/KalmanFilter.h>
#include <nabia/forecast/Kernels.h>
#include <nabia/forecast/exponential_smoothing.hpp>

using Eigen::Index;
using Eigen::VectorXd;
using nabia::forecast::SimpleExponentialSmoothing;
using nabia::forecast::SimpleExponentialSmoothingParams;

namespace nabia_bridge {

namespace {

/// The members shared by every IForecaster<Params>. fit() releases the GIL.
template <typename Model>
void bind_forecaster(py::class_<Model>& c) {
    c.def(
         "fit",
         [](Model& model, const VectorXd& y) {
             const py::gil_scoped_release release;
             model.fit(y);
         },
         py::arg("y"), "Fits the model to the series y.")
        .def("forecast", &Model::forecast, py::arg("horizon"), "Forecast mean and variance, 1..horizon steps ahead.")
        .def("fitted_values", &Model::fitted_values, "In-sample fit.")
        .def_property_readonly("is_fitted", &Model::is_fitted);
}

std::unique_ptr<IKernel> clone_or_null(const IKernel* k) { return k != nullptr ? k->clone() : nullptr; }

/// GaussianProcessParams owns its kernel: copies clone it.
GaussianProcessParams copy(const GaussianProcessParams& p) {
    GaussianProcessParams out;
    out.kernel = clone_or_null(p.kernel.get());
    out.noise_variance = p.noise_variance;
    out.normalize = p.normalize;
    out.optimize_hyperparameters = p.optimize_hyperparameters;
    out.optimize_noise = p.optimize_noise;
    out.min_noise_variance = p.min_noise_variance;
    out.max_iter = p.max_iter;
    return out;
}

} // namespace

void bind_forecast(py::module_& m) {
    py::class_<Forecast>(m, "Forecast")
        .def(py::init<>())
        .def_readwrite("mean", &Forecast::mean, "Point forecasts, h = 1..H steps ahead.")
        .def_readwrite("variance", &Forecast::variance, "Predictive variance of the observation (noise included).")
        .def_property_readonly("horizon", &Forecast::horizon)
        .def("stddev", &Forecast::stddev)
        .def("lower", &Forecast::lower, py::arg("level"), "Lower end of the central Gaussian interval, level in (0, 1).")
        .def("upper", &Forecast::upper, py::arg("level"), "Upper end of the central Gaussian interval, level in (0, 1).");

    // --- Kernels ---
    py::class_<IKernel>(m, "IKernel", "Covariance function over time for GaussianProcess.")
        .def("__call__", &IKernel::operator(), py::arg("t1"), py::arg("t2"))
        .def("matrix", &IKernel::matrix, py::arg("a"), py::arg("b"), "Gram matrix K_ij = k(a_i, b_j).")
        .def_property_readonly("num_hyperparameters", &IKernel::num_hyperparameters)
        .def("log_hyperparameters", &IKernel::log_hyperparameters)
        .def("set_log_hyperparameters", &IKernel::set_log_hyperparameters, py::arg("log_theta"))
        .def("gradient", &IKernel::gradient, py::arg("t1"), py::arg("t2"),
             "Gradient of k(t1, t2) with respect to the log-hyperparameters.")
        .def("clone", &IKernel::clone)
        .def("__add__", [](const IKernel& a, const IKernel& b) { return std::make_unique<SumKernel>(a.clone(), b.clone()); })
        .def("__mul__",
             [](const IKernel& a, const IKernel& b) { return std::make_unique<ProductKernel>(a.clone(), b.clone()); });
    py::class_<SquaredExponentialKernel, IKernel>(m, "SquaredExponentialKernel")
        .def(py::init<double, double>(), py::arg("variance"), py::arg("length_scale"))
        .def_property_readonly("variance", &SquaredExponentialKernel::variance)
        .def_property_readonly("length_scale", &SquaredExponentialKernel::length_scale);
    py::class_<Matern52Kernel, IKernel>(m, "Matern52Kernel")
        .def(py::init<double, double>(), py::arg("variance"), py::arg("length_scale"))
        .def_property_readonly("variance", &Matern52Kernel::variance)
        .def_property_readonly("length_scale", &Matern52Kernel::length_scale);
    py::class_<PeriodicKernel, IKernel>(m, "PeriodicKernel")
        .def(py::init<double, double, double>(), py::arg("variance"), py::arg("length_scale"), py::arg("period"))
        .def_property_readonly("variance", &PeriodicKernel::variance)
        .def_property_readonly("length_scale", &PeriodicKernel::length_scale)
        .def_property_readonly("period", &PeriodicKernel::period);
    py::class_<SumKernel, IKernel>(m, "SumKernel", "k1 + k2 (also: k1 + k2 on any two kernels).")
        .def(py::init([](const IKernel& a, const IKernel& b) { return std::make_unique<SumKernel>(a.clone(), b.clone()); }),
             py::arg("k1"), py::arg("k2"));
    py::class_<ProductKernel, IKernel>(m, "ProductKernel", "k1 * k2 (also: k1 * k2 on any two kernels).")
        .def(py::init([](const IKernel& a, const IKernel& b) { return std::make_unique<ProductKernel>(a.clone(), b.clone()); }),
             py::arg("k1"), py::arg("k2"));

    // --- Simple exponential smoothing ---
    py::class_<SimpleExponentialSmoothingParams>(m, "SimpleExponentialSmoothingParams")
        .def(kwargs_init<SimpleExponentialSmoothingParams>())
        .def_readwrite("alpha", &SimpleExponentialSmoothingParams::alpha);
    py::class_<SimpleExponentialSmoothing> ses(m, "SimpleExponentialSmoothing", "Level-only exponential smoothing.");
    ses.def(py::init<double>(), py::arg("alpha"))
        .def(py::init([](const SimpleExponentialSmoothingParams& p) {
                 return std::make_unique<SimpleExponentialSmoothing>(std::make_unique<SimpleExponentialSmoothingParams>(p));
             }),
             py::arg("parameters"))
        .def_property_readonly("sse", &SimpleExponentialSmoothing::sse)
        .def_property_readonly("alpha", &SimpleExponentialSmoothing::alpha);
    bind_forecaster(ses);

    // --- Kalman filter ---
    py::class_<KalmanFilterParams>(m, "KalmanFilterParams")
        .def(kwargs_init<KalmanFilterParams>())
        .def_readwrite("transition", &KalmanFilterParams::transition)
        .def_readwrite("observation", &KalmanFilterParams::observation)
        .def_readwrite("process_noise", &KalmanFilterParams::process_noise)
        .def_readwrite("observation_noise", &KalmanFilterParams::observation_noise)
        .def_readwrite("initial_state", &KalmanFilterParams::initial_state)
        .def_readwrite("initial_covariance", &KalmanFilterParams::initial_covariance)
        .def_static("local_level", &KalmanFilterParams::local_level, py::arg("level_variance"),
                    py::arg("observation_variance"), py::arg("initial_level") = 0.0, py::arg("initial_variance") = 1e7)
        .def_static("local_linear_trend", &KalmanFilterParams::local_linear_trend, py::arg("level_variance"),
                    py::arg("slope_variance"), py::arg("observation_variance"), py::arg("initial_variance") = 1e7);
    py::class_<KalmanFilter> kf(m, "KalmanFilter", "Linear Gaussian state-space model (NaN observations are missing).");
    kf.def(py::init([](const KalmanFilterParams& p) {
               return std::make_unique<KalmanFilter>(std::make_unique<KalmanFilterParams>(p));
           }),
           py::arg("parameters"))
        .def_property_readonly("state_dim", &KalmanFilter::state_dim)
        .def_property_readonly("parameters", [](const KalmanFilter& k) { return k.parameters(); })
        .def("filtered_states", &KalmanFilter::filtered_states, "x[t|t] as columns (n x T).")
        .def("filtered_covariances", &KalmanFilter::filtered_covariances, "P[t|t], one n x n matrix per time.")
        .def("innovations", &KalmanFilter::innovations)
        .def("innovation_variances", &KalmanFilter::innovation_variances)
        .def("log_likelihood", &KalmanFilter::log_likelihood)
        .def("smoothed_states", &KalmanFilter::smoothed_states, "x[t|T] as columns (n x T).");
    bind_forecaster(kf);

    // --- Gaussian process ---
    py::class_<GaussianProcessParams>(m, "GaussianProcessParams")
        .def(kwargs_init<GaussianProcessParams>())
        .def_property(
            "kernel", [](const GaussianProcessParams& p) { return clone_or_null(p.kernel.get()); },
            [](GaussianProcessParams& p, const IKernel* k) { p.kernel = clone_or_null(k); },
            "Prior covariance (stored as a copy).")
        .def_readwrite("noise_variance", &GaussianProcessParams::noise_variance)
        .def_readwrite("normalize", &GaussianProcessParams::normalize)
        .def_readwrite("optimize_hyperparameters", &GaussianProcessParams::optimize_hyperparameters)
        .def_readwrite("optimize_noise", &GaussianProcessParams::optimize_noise)
        .def_readwrite("min_noise_variance", &GaussianProcessParams::min_noise_variance)
        .def_readwrite("max_iter", &GaussianProcessParams::max_iter);
    py::class_<GaussianProcess> gp(m, "GaussianProcess", "Gaussian process regression on the time index.");
    gp.def(py::init([](const GaussianProcessParams& p) {
               return std::make_unique<GaussianProcess>(std::make_unique<GaussianProcessParams>(copy(p)));
           }),
           py::arg("parameters"))
        .def_property_readonly("parameters", [](const GaussianProcess& g) { return copy(g.parameters()); })
        .def("predict", &GaussianProcess::predict, py::arg("times"), "Predictions at arbitrary times.")
        .def("kernel", [](const GaussianProcess& g) { return g.kernel().clone(); }, "The fitted kernel (a copy).")
        .def("noise_variance", &GaussianProcess::noise_variance)
        .def("log_marginal_likelihood", &GaussianProcess::log_marginal_likelihood);
    bind_forecaster(gp);
}

} // namespace nabia_bridge
