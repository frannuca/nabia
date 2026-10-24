#include "bridge.hpp"

#include <nabia/regression/ElasticNet.h>
#include <nabia/regression/RegressionEngine.h>

#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace nabia_bridge {

void bind_regression(py::module_& m) {
    namespace rg = nabia::regression;

    // --- Elastic net ---
    py::class_<ElasticNetParams>(m, "ElasticNetParams")
        .def(kwargs_init<ElasticNetParams>())
        .def_readwrite("lambda_", &ElasticNetParams::lambda, "Penalty strength (>= 0); `lambda` is a Python keyword.")
        .def_readwrite("alpha", &ElasticNetParams::alpha, "Mix in [0, 1]: 1 = lasso, 0 = ridge.")
        .def_readwrite("fit_intercept", &ElasticNetParams::fit_intercept)
        .def_readwrite("standardize", &ElasticNetParams::standardize)
        .def_readwrite("tolerance", &ElasticNetParams::tolerance)
        .def_readwrite("max_iter", &ElasticNetParams::max_iter);
    py::class_<ElasticNetFit>(m, "ElasticNetFit")
        .def_readonly("lambda_", &ElasticNetFit::lambda)
        .def_readonly("alpha", &ElasticNetFit::alpha)
        .def_readonly("intercept", &ElasticNetFit::intercept)
        .def_readonly("coefficients", &ElasticNetFit::coefficients)
        .def_readonly("nonzero", &ElasticNetFit::nonzero)
        .def_readonly("degrees_of_freedom", &ElasticNetFit::degrees_of_freedom)
        .def_readonly("iterations", &ElasticNetFit::iterations)
        .def_readonly("converged", &ElasticNetFit::converged)
        .def("predict", &ElasticNetFit::predict, py::arg("X"));
    m.def("lambda_max", &rg::lambda_max, py::arg("X"), py::arg("y"), py::arg("alpha"), py::arg("fit_intercept") = true,
          py::arg("standardize") = true, "Smallest lambda at which every coefficient is zero.");
    m.def("lambda_grid", &rg::lambda_grid, py::arg("lambda_max"), py::arg("count"), py::arg("min_ratio"),
          "Log-spaced decreasing lambdas from lambda_max to lambda_max * min_ratio.");
    m.def(
        "elastic_net_path",
        [](const MatrixXd& X, const VectorXd& y, const ElasticNetParams& params, const std::vector<double>& lambdas) {
            const py::gil_scoped_release release;
            return rg::elastic_net_path(X, y, params, lambdas);
        },
        py::arg("X"), py::arg("y"), py::arg("parameters"), py::arg("lambdas"),
        "Warm-started fits along `lambdas` (params.lambda_ is ignored).");
    py::class_<ElasticNet>(m, "ElasticNet", "Elastic net linear regression by coordinate descent.")
        .def(py::init([](const ElasticNetParams& p) { return std::make_unique<ElasticNet>(std::make_unique<ElasticNetParams>(p)); }),
             py::arg("parameters"))
        .def(py::init([](const py::kwargs& kw) { return std::make_unique<ElasticNet>(from_kwargs<ElasticNetParams>(kw)); }))
        .def(
            "fit",
            [](ElasticNet& en, const MatrixXd& X, const VectorXd& y) {
                const py::gil_scoped_release release;
                en.fit(X, y);
            },
            py::arg("X"), py::arg("y"))
        .def("predict", &ElasticNet::predict, py::arg("X"))
        .def_property_readonly("result", [](const ElasticNet& en) { return en.result(); })
        .def_property_readonly("parameters", &ElasticNet::parameters);

    // --- Regression engine ---
    py::enum_<SubsetSearch>(m, "SubsetSearch")
        .value("Auto", SubsetSearch::Auto)
        .value("Exhaustive", SubsetSearch::Exhaustive)
        .value("Forward", SubsetSearch::Forward);
    py::enum_<ValidationScheme>(m, "ValidationScheme")
        .value("BlockedKFold", ValidationScheme::BlockedKFold)
        .value("WalkForward", ValidationScheme::WalkForward);
    py::enum_<InSampleCriterion>(m, "InSampleCriterion")
        .value("BIC", InSampleCriterion::BIC)
        .value("AIC", InSampleCriterion::AIC)
        .value("MSE", InSampleCriterion::MSE);

    using P = RegressionEngineParams;
    py::class_<P>(m, "RegressionEngineParams")
        .def(kwargs_init<P>())
        .def_readwrite("min_factors", &P::min_factors)
        .def_readwrite("max_factors", &P::max_factors, "0 = all factors.")
        .def_readwrite("search", &P::search)
        .def_readwrite("max_exhaustive_subsets", &P::max_exhaustive_subsets)
        .def_readwrite("train_fraction", &P::train_fraction, "1.0 = in-sample comparison of subsets.")
        .def_readwrite("in_sample_criterion", &P::in_sample_criterion)
        .def_readwrite("refit_on_full_sample", &P::refit_on_full_sample)
        .def_readwrite("alphas", &P::alphas)
        .def_readwrite("lambdas", &P::lambdas, "Fixed grid; empty = log grid from lambda_max per alpha.")
        .def_readwrite("n_lambdas", &P::n_lambdas)
        .def_readwrite("lambda_min_ratio", &P::lambda_min_ratio)
        .def_readwrite("validation", &P::validation)
        .def_readwrite("validation_folds", &P::validation_folds)
        .def_readwrite("validation_gap", &P::validation_gap)
        .def_readwrite("one_standard_error_rule", &P::one_standard_error_rule)
        .def_readwrite("bootstrap_samples", &P::bootstrap_samples)
        .def_readwrite("bootstrap_block_size", &P::bootstrap_block_size)
        .def_readwrite("confidence_level", &P::confidence_level)
        .def_readwrite("bootstrap_seed", &P::bootstrap_seed)
        .def_readwrite("fit_intercept", &P::fit_intercept)
        .def_readwrite("standardize", &P::standardize)
        .def_readwrite("tolerance", &P::tolerance)
        .def_readwrite("max_iter", &P::max_iter)
        .def_readwrite("num_threads", &P::num_threads);

    using S = SubsetEvaluation;
    py::class_<S>(m, "SubsetEvaluation")
        .def_readonly("factors", &S::factors)
        .def_readonly("alpha", &S::alpha)
        .def_readonly("lambda_", &S::lambda)
        .def_readonly("intercept", &S::intercept)
        .def_readonly("coefficients", &S::coefficients)
        .def_readonly("nonzero", &S::nonzero)
        .def_readonly("degrees_of_freedom", &S::degrees_of_freedom)
        .def_readonly("validation_mse", &S::validation_mse)
        .def_readonly("validation_se", &S::validation_se)
        .def_readonly("train_mse", &S::train_mse)
        .def_readonly("train_r2", &S::train_r2)
        .def_readonly("test_mse", &S::test_mse)
        .def_readonly("test_r2", &S::test_r2)
        .def_readonly("criterion", &S::criterion)
        .def_readonly("score", &S::score)
        .def_readonly("converged", &S::converged);
    py::class_<ValidationCurve>(m, "ValidationCurve")
        .def_readonly("alphas", &ValidationCurve::alphas)
        .def_readonly("lambdas", &ValidationCurve::lambdas)
        .def_readonly("mse", &ValidationCurve::mse)
        .def_readonly("se", &ValidationCurve::se);
    using B = BootstrapSummary;
    py::class_<B>(m, "BootstrapSummary")
        .def_readonly("samples", &B::samples)
        .def_readonly("mean", &B::mean)
        .def_readonly("std_error", &B::std_error)
        .def_readonly("lower", &B::lower)
        .def_readonly("upper", &B::upper)
        .def_readonly("selection_frequency", &B::selection_frequency)
        .def_readonly("intercept_mean", &B::intercept_mean)
        .def_readonly("intercept_std_error", &B::intercept_std_error)
        .def_readonly("test_mse_mean", &B::test_mse_mean)
        .def_readonly("test_mse_lower", &B::test_mse_lower)
        .def_readonly("test_mse_upper", &B::test_mse_upper)
        .def_readonly("coefficients", &B::coefficients);
    using C = RegressionComputation;
    py::class_<C>(m, "RegressionComputation")
        .def_readonly("factor_names", &C::factor_names)
        .def_readonly("factor_positions", &C::factor_positions)
        .def_readonly("alpha", &C::alpha)
        .def_readonly("lambda_", &C::lambda)
        .def_readonly("intercept", &C::intercept)
        .def_readonly("coefficients", &C::coefficients)
        .def_readonly("best", &C::best)
        .def_readonly("best_by_size", &C::best_by_size)
        .def_readonly("validation_curve", &C::validation_curve)
        .def_readonly("bootstrap", &C::bootstrap)
        .def_readonly("predictions", &C::predictions)
        .def_readonly("residuals", &C::residuals)
        .def_readonly("train_rows", &C::train_rows)
        .def_readonly("test_rows", &C::test_rows)
        .def_readonly("search_used", &C::search_used)
        .def_readonly("subsets_evaluated", &C::subsets_evaluated)
        .def_readonly("all_converged", &C::all_converged)
        .def_readonly("threads_used", &C::threads_used);

    py::class_<RegressionEngine>(m, "RegressionEngine",
                                 "Elastic net regression of a target on factors, choosing the factors, lambda and alpha.")
        .def(py::init([](const P& p) { return std::make_unique<RegressionEngine>(std::make_unique<P>(p)); }),
             py::arg("parameters"))
        .def(py::init([](const py::kwargs& kw) { return std::make_unique<RegressionEngine>(from_kwargs<P>(kw)); }))
        .def(
            "run",
            [](const RegressionEngine& engine, const VectorXd& target, const MatrixXd& factors,
               std::vector<std::string> names) {
                const py::gil_scoped_release release;
                return engine.run(target, factors, std::move(names));
            },
            py::arg("target"), py::arg("factors"), py::arg("names") = std::vector<std::string>{},
            "target: (T,), factors: (T x N), rows in time order; names default to x0, x1, ...")
        .def_property_readonly("parameters", &RegressionEngine::parameters);
}

} // namespace nabia_bridge
