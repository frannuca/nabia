#include <nabia/forecast/GaussianProcess.h>
#include <nabia/forecast/KalmanFilter.h>
#include <nabia/forecast/Kernels.h>
#include <nabia/forecast/exponential_smoothing.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

VectorXd noise(Eigen::Index n, double sd, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, sd);
    VectorXd e(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        e(i) = dist(rng);
    }
    return e;
}

std::unique_ptr<KalmanFilterParams> local_level(double q, double r) {
    return std::make_unique<KalmanFilterParams>(KalmanFilterParams::local_level(q, r));
}

std::unique_ptr<GaussianProcessParams> gp_params(std::unique_ptr<IKernel> kernel, double noise_var, bool optimize) {
    auto p = std::make_unique<GaussianProcessParams>();
    p->kernel = std::move(kernel);
    p->noise_variance = noise_var;
    p->optimize_hyperparameters = optimize;
    return p;
}

// Any forecaster through the base class.
template <typename T>
Forecast fit_and_forecast(IForecaster<T>& model, const VectorXd& y, Eigen::Index h) {
    model.fit(y);
    return model.forecast(h);
}

} // namespace

// --- Kalman filter ---------------------------------------------------------------

TEST_CASE("kalman: local level converges to SES with the steady-state gain", "[forecast]") {
    const double q = 0.5;
    const double r = 2.0;
    const VectorXd y = VectorXd::LinSpaced(300, 0.0, 30.0).array().sin() * 5.0 + noise(300, 1.0, 1).array();
    KalmanFilter kf(local_level(q, r));
    kf.fit(y);

    // Steady state: P solves P^2 - qP - qr = 0 and the gain is K = P / (P + r).
    const double P = 0.5 * (q + std::sqrt(q * q + 4.0 * q * r));
    const double K = P / (P + r);
    nabia::forecast::SimpleExponentialSmoothing ses(K);
    ses.fit(y.tail(100));
    // After the transient both recursions forget their start: compare the last levels.
    CHECK_THAT(kf.filtered_states()(0, 299), WithinAbs(ses.forecast(1).mean(0), 1e-9));
}

TEST_CASE("kalman: log-likelihood equals the joint Gaussian density", "[forecast]") {
    // Local level with prior N(a, P0): y ~ N(a 1, Sigma), Sigma_ij = P0 + q min(i, j) + r delta_ij.
    const double q = 0.3;
    const double r = 0.7;
    const double a = 1.5;
    const double P0 = 2.0;
    auto p = std::make_unique<KalmanFilterParams>(KalmanFilterParams::local_level(q, r, a, P0));
    KalmanFilter kf(std::move(p));
    const VectorXd y{{1.0, 2.5, 1.8, 3.1, 2.2}};
    kf.fit(y);

    const Eigen::Index n = y.size();
    MatrixXd sigma(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            sigma(i, j) = P0 + q * static_cast<double>(std::min(i, j)) + (i == j ? r : 0.0);
        }
    }
    const Eigen::LDLT<MatrixXd> ldlt(sigma);
    const VectorXd d = y.array() - a;
    const double expected = -0.5 * (d.dot(ldlt.solve(d)) + ldlt.vectorD().array().log().sum() +
                                    static_cast<double>(n) * std::log(2.0 * std::numbers::pi));
    CHECK_THAT(kf.log_likelihood(), WithinAbs(expected, 1e-10));
}

TEST_CASE("kalman: local linear trend extrapolates the slope", "[forecast]") {
    const Eigen::Index T = 200;
    const VectorXd t = VectorXd::LinSpaced(T, 0.0, static_cast<double>(T - 1));
    const VectorXd y = (2.0 + 0.5 * t.array()).matrix() + noise(T, 0.3, 2);
    KalmanFilter kf(std::make_unique<KalmanFilterParams>(KalmanFilterParams::local_linear_trend(1e-4, 1e-8, 0.09)));
    const Forecast fc = fit_and_forecast(kf, y, 10);
    for (Eigen::Index h = 0; h < 10; ++h) {
        CHECK_THAT(fc.mean(h), WithinAbs(2.0 + 0.5 * static_cast<double>(T + h), 0.15));
    }
    CHECK_THAT(kf.filtered_states()(1, T - 1), WithinAbs(0.5, 0.01));  // slope state
    CHECK((fc.variance.tail(9).array() > fc.variance.head(9).array()).all());
}

TEST_CASE("kalman: missing values, forecast variance and smoother", "[forecast]") {
    const double q = 0.1;
    const double r = 1.0;
    KalmanFilter kf(local_level(q, r));
    const VectorXd y{{1.0, 1.2, kNaN, kNaN, 0.9, 1.1}};
    kf.fit(y);
    CHECK(std::isnan(kf.innovations()(2)));
    // A missing step is predicted, not corrected: its variance keeps growing.
    CHECK(kf.filtered_covariances()[3](0, 0) > kf.filtered_covariances()[2](0, 0));
    CHECK(kf.filtered_covariances()[4](0, 0) < kf.filtered_covariances()[3](0, 0));

    // Local level forecast: mean = last level, variance = P[T|T] + h q + r.
    const Forecast fc = kf.forecast(3);
    const double PT = kf.filtered_covariances().back()(0, 0);
    for (Eigen::Index h = 0; h < 3; ++h) {
        CHECK(fc.mean(h) == kf.filtered_states()(0, 5));
        CHECK_THAT(fc.variance(h), WithinAbs(PT + static_cast<double>(h + 1) * q + r, 1e-12));
    }

    // With a constant level (q = 0), every smoothed state is the final filtered estimate.
    KalmanFilter constant(local_level(0.0, r));
    constant.fit(VectorXd{{1.0, 3.0, 2.0, 2.5}});
    const MatrixXd smoothed = constant.smoothed_states();
    CHECK(smoothed.isApproxToConstant(constant.filtered_states()(0, 3), 1e-9));
}

TEST_CASE("kalman: invalid parameters and usage", "[forecast]") {
    auto bad = std::make_unique<KalmanFilterParams>(KalmanFilterParams::local_linear_trend(1.0, 1.0, 1.0));
    bad->observation = Eigen::RowVectorXd::Ones(3);
    CHECK_THROWS_AS(KalmanFilter(std::move(bad)), std::invalid_argument);
    CHECK_THROWS_AS(KalmanFilter(local_level(1.0, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS(KalmanFilter(nullptr), std::invalid_argument);

    KalmanFilter kf(local_level(1.0, 1.0));
    CHECK_THROWS_AS(kf.forecast(1), std::logic_error);
    CHECK_THROWS_AS(kf.fit(VectorXd()), std::invalid_argument);
}

// --- Kernels -----------------------------------------------------------------------

TEST_CASE("kernels: gradients match finite differences", "[forecast]") {
    std::vector<std::unique_ptr<IKernel>> kernels;
    kernels.push_back(std::make_unique<SquaredExponentialKernel>(1.3, 2.0));
    kernels.push_back(std::make_unique<Matern52Kernel>(0.8, 1.5));
    kernels.push_back(std::make_unique<PeriodicKernel>(1.1, 0.9, 7.0));
    kernels.push_back(std::make_unique<SumKernel>(std::make_unique<SquaredExponentialKernel>(1.0, 3.0),
                                                  std::make_unique<PeriodicKernel>(0.5, 1.2, 5.0)));
    kernels.push_back(std::make_unique<ProductKernel>(std::make_unique<Matern52Kernel>(1.0, 4.0),
                                                      std::make_unique<PeriodicKernel>(1.0, 0.7, 6.0)));
    for (const auto& k : kernels) {
        const VectorXd theta = k->log_hyperparameters();
        const VectorXd g = k->gradient(1.7, 4.2);
        REQUIRE(g.size() == k->num_hyperparameters());
        for (Eigen::Index j = 0; j < theta.size(); ++j) {
            constexpr double h = 1e-6;
            auto up = k->clone();
            auto down = k->clone();
            VectorXd tp = theta;
            VectorXd tm = theta;
            tp(j) += h;
            tm(j) -= h;
            up->set_log_hyperparameters(tp);
            down->set_log_hyperparameters(tm);
            CHECK_THAT(g(j), WithinAbs(((*up)(1.7, 4.2) - (*down)(1.7, 4.2)) / (2.0 * h), 1e-7));
        }
        CHECK_THAT((*k)(2.0, 5.0), WithinAbs((*k)(5.0, 2.0), 1e-15));  // symmetric
    }
}

// --- Gaussian process -----------------------------------------------------------------

TEST_CASE("gaussian process: interpolates a smooth function", "[forecast]") {
    const Eigen::Index T = 40;
    const VectorXd t = VectorXd::LinSpaced(T, 0.0, static_cast<double>(T - 1));
    const VectorXd y = 3.0 + (t.array() / 5.0).sin();
    GaussianProcess gp(gp_params(std::make_unique<SquaredExponentialKernel>(1.0, 5.0), 1e-6, false));
    gp.fit(y);
    CHECK((gp.fitted_values() - y).lpNorm<Eigen::Infinity>() < 1e-3);

    const VectorXd mid = VectorXd::LinSpaced(10, 10.5, 19.5);
    const Forecast between = gp.predict(mid);
    CHECK((between.mean.array() - (3.0 + (mid.array() / 5.0).sin())).abs().maxCoeff() < 1e-3);
    CHECK(between.variance.maxCoeff() < 1e-3);
}

TEST_CASE("gaussian process: hyperparameter fit raises the marginal likelihood", "[forecast]") {
    const Eigen::Index T = 60;
    const VectorXd t = VectorXd::LinSpaced(T, 0.0, static_cast<double>(T - 1));
    const VectorXd y = (t.array() / 4.0).sin().matrix() + noise(T, 0.1, 3);

    GaussianProcess fixed(gp_params(std::make_unique<SquaredExponentialKernel>(1.0, 1.0), 0.5, false));
    GaussianProcess tuned(gp_params(std::make_unique<SquaredExponentialKernel>(1.0, 1.0), 0.5, true));
    fixed.fit(y);
    tuned.fit(y);
    CHECK(tuned.log_marginal_likelihood() > fixed.log_marginal_likelihood() + 10.0);
    const auto& se = dynamic_cast<const SquaredExponentialKernel&>(tuned.kernel());
    CHECK(se.length_scale() > 2.0);       // the data vary on a scale of ~4 steps, not 1
    CHECK(tuned.noise_variance() < 0.1);  // noise sd 0.1 vs signal sd ~0.7 (normalized units)
    // The initial kernel in the parameters is left untouched.
    CHECK(dynamic_cast<const SquaredExponentialKernel&>(*tuned.parameters().kernel).length_scale() == 1.0);
}

TEST_CASE("gaussian process: periodic kernel extrapolates seasonality", "[forecast]") {
    const Eigen::Index T = 72;
    auto season = [](const VectorXd& tt) -> VectorXd { return 10.0 + 2.0 * (2.0 * std::numbers::pi * tt.array() / 12.0).sin(); };
    const VectorXd t = VectorXd::LinSpaced(T, 0.0, static_cast<double>(T - 1));
    const VectorXd y = season(t) + noise(T, 0.05, 4);
    GaussianProcess gp(gp_params(std::make_unique<PeriodicKernel>(1.0, 1.0, 11.5), 0.01, true));
    const Forecast fc = fit_and_forecast(gp, y, 24);
    const VectorXd future = VectorXd::LinSpaced(24, static_cast<double>(T), static_cast<double>(T + 23));
    CHECK((fc.mean - season(future)).lpNorm<Eigen::Infinity>() < 0.1);
    CHECK_THAT(dynamic_cast<const PeriodicKernel&>(gp.kernel()).period(), WithinAbs(12.0, 0.05));
    // The true values fall inside the 95% interval.
    CHECK((season(future).array() > fc.lower(0.95).array()).all());
    CHECK((season(future).array() < fc.upper(0.95).array()).all());
}

TEST_CASE("gaussian process: far forecasts revert to the mean with the prior variance", "[forecast]") {
    const VectorXd y{{1.0, 2.0, 1.5, 3.0, 2.5, 2.0}};
    GaussianProcess gp(gp_params(std::make_unique<SquaredExponentialKernel>(1.0, 1.0), 0.1, false));
    gp.fit(y);
    const Forecast far = gp.predict(VectorXd{{1000.0}});
    const double sd = std::sqrt((y.array() - y.mean()).square().sum() / 5.0);
    CHECK_THAT(far.mean(0), WithinAbs(y.mean(), 1e-12));
    CHECK_THAT(far.variance(0), WithinRel((1.0 + 0.1) * sd * sd, 1e-12));
}

TEST_CASE("gaussian process: invalid input", "[forecast]") {
    CHECK_THROWS_AS(GaussianProcess(gp_params(nullptr, 0.1, false)), std::invalid_argument);
    CHECK_THROWS_AS(GaussianProcess(gp_params(std::make_unique<Matern52Kernel>(1.0, 1.0), 0.0, false)),
                    std::invalid_argument);
    CHECK_THROWS_AS(SquaredExponentialKernel(-1.0, 1.0), std::invalid_argument);

    GaussianProcess gp(gp_params(std::make_unique<Matern52Kernel>(1.0, 1.0), 0.1, false));
    CHECK_THROWS_AS(gp.forecast(3), std::logic_error);
    CHECK_THROWS_AS(gp.fit(VectorXd{{1.0}}), std::invalid_argument);
    CHECK_THROWS_AS(gp.fit(VectorXd{{1.0, kNaN, 2.0}}), std::invalid_argument);
}
