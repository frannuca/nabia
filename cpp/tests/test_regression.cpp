#include <nabia/dataframe/DataFrame.h>
#include <nabia/dataframe/TimeSeries.h>
#include <nabia/regression/ElasticNet.h>
#include <nabia/regression/RegressionEngine.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

MatrixXd gaussian(Index rows, Index cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    MatrixXd m(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            m(i, j) = dist(rng);
        }
    }
    return m;
}

/// y = 1 + 2 x0 - 1.5 x3 + noise, with `factors` candidate factors.
struct Synthetic {
    MatrixXd X;
    VectorXd y;
};

Synthetic synthetic(Index rows, Index factors, double noise_sd, unsigned seed) {
    Synthetic s;
    s.X = gaussian(rows, factors, seed);
    s.X.col(3) *= 5.0;  // different units: standardization must not change the selection
    s.y = 1.0 + 2.0 * s.X.col(0).array() - 0.3 * s.X.col(3).array();
    s.y += noise_sd * gaussian(rows, 1, seed + 1).col(0);
    return s;
}

std::unique_ptr<ElasticNetParams> en_params(double lambda, double alpha, bool standardize = true) {
    auto p = std::make_unique<ElasticNetParams>();
    p->lambda = lambda;
    p->alpha = alpha;
    p->standardize = standardize;
    p->tolerance = 1e-14;
    return p;
}

std::unique_ptr<RegressionEngineParams> engine_params() {
    auto p = std::make_unique<RegressionEngineParams>();
    p->alphas = {0.5, 1.0};
    p->n_lambdas = 20;
    p->num_threads = 1;
    return p;
}

} // namespace

// --- Elastic net solver -------------------------------------------------------------

TEST_CASE("elastic net: lambda = 0 is ordinary least squares", "[regression]") {
    const MatrixXd X = gaussian(80, 4, 1);
    const VectorXd y = 0.5 + (X * VectorXd::LinSpaced(4, -1.0, 2.0)).array() + 0.1 * gaussian(80, 1, 2).col(0).array();
    ElasticNet en(en_params(0.0, 1.0));
    en.fit(X, y);

    MatrixXd design(80, 5);
    design << VectorXd::Ones(80), X;
    const VectorXd ols = design.colPivHouseholderQr().solve(y);
    REQUIRE(en.result().converged);
    CHECK_THAT(en.result().intercept, WithinAbs(ols(0), 1e-6));
    for (Index j = 0; j < 4; ++j) {
        CHECK_THAT(en.result().coefficients(j), WithinAbs(ols(j + 1), 1e-6));
    }
    CHECK_THAT(en.result().degrees_of_freedom, WithinAbs(5.0, 1e-9));
}

TEST_CASE("elastic net: alpha = 0 matches the ridge closed form", "[regression]") {
    const MatrixXd X = gaussian(60, 3, 3);
    const VectorXd y = (X * VectorXd::LinSpaced(3, 1.0, 3.0)).array() + gaussian(60, 1, 4).col(0).array();
    const double lambda = 0.7;
    ElasticNet en(en_params(lambda, 0.0, false));
    en.fit(X, y);

    const MatrixXd Xc = X.rowwise() - X.colwise().mean();
    const VectorXd yc = y.array() - y.mean();
    const double n = 60.0;
    const MatrixXd gram = Xc.transpose() * Xc / n + lambda * MatrixXd::Identity(3, 3);
    const VectorXd ridge = gram.ldlt().solve(Xc.transpose() * yc / n);
    for (Index j = 0; j < 3; ++j) {
        CHECK_THAT(en.result().coefficients(j), WithinAbs(ridge(j), 1e-8));
    }
}

TEST_CASE("elastic net: solution satisfies the optimality conditions", "[regression]") {
    const MatrixXd X = gaussian(100, 8, 5);
    const VectorXd y = 2.0 * X.col(1) - X.col(6) + 0.5 * gaussian(100, 1, 6).col(0);
    const double lambda = 0.1;
    const double alpha = 0.6;
    ElasticNet en(en_params(lambda, alpha, false));
    en.fit(X, y);
    const ElasticNetFit& f = en.result();

    const MatrixXd Xc = X.rowwise() - X.colwise().mean();
    const VectorXd r = (y.array() - y.mean()).matrix() - Xc * f.coefficients;
    const VectorXd g = Xc.transpose() * r / 100.0;  // gradient of the loss part
    for (Index j = 0; j < 8; ++j) {
        const double b = f.coefficients(j);
        if (b != 0.0) {
            CHECK_THAT(g(j), WithinAbs(lambda * alpha * (b > 0 ? 1.0 : -1.0) + lambda * (1.0 - alpha) * b, 1e-6));
        } else {
            CHECK(std::abs(g(j)) <= lambda * alpha + 1e-9);
        }
    }
    CHECK_THAT(f.intercept, WithinAbs(y.mean() - X.colwise().mean().dot(f.coefficients), 1e-12));
}

TEST_CASE("elastic net: lambda_max zeroes every coefficient", "[regression]") {
    const Synthetic s = synthetic(120, 5, 0.5, 7);
    const double lmax = nabia::regression::lambda_max(s.X, s.y, 0.5);
    const std::vector<double> grid = nabia::regression::lambda_grid(lmax, 10, 1e-2);
    REQUIRE(grid.size() == 10);
    CHECK_THAT(grid.front(), WithinRel(lmax, 1e-12));
    CHECK_THAT(grid.back(), WithinRel(lmax * 1e-2, 1e-12));

    auto settings = en_params(0.0, 0.5);
    const std::vector<ElasticNetFit> path = nabia::regression::elastic_net_path(s.X, s.y, *settings, grid);
    CHECK(path.front().nonzero == 0);
    CHECK_THAT(path.front().intercept, WithinAbs(s.y.mean(), 1e-12));
    CHECK(path[1].nonzero > 0);
    CHECK(path.back().nonzero >= path[1].nonzero);
}

TEST_CASE("elastic net: constant factors get a zero coefficient", "[regression]") {
    MatrixXd X = gaussian(50, 3, 8);
    X.col(1).setConstant(4.2);
    const VectorXd y = X.col(0) + X.col(2);
    ElasticNet en(en_params(1e-3, 1.0));
    en.fit(X, y);
    CHECK(en.result().coefficients(1) == 0.0);
    CHECK_THAT(en.result().coefficients(0), WithinAbs(1.0, 1e-2));
}

TEST_CASE("elastic net: invalid input throws", "[regression]") {
    CHECK_THROWS_AS(ElasticNet(nullptr), std::invalid_argument);
    CHECK_THROWS_AS(ElasticNet(en_params(-1.0, 0.5)), std::invalid_argument);
    CHECK_THROWS_AS(ElasticNet(en_params(0.1, 1.5)), std::invalid_argument);
    ElasticNet en(en_params(0.1, 0.5));
    CHECK_THROWS_AS(en.result(), std::logic_error);
    CHECK_THROWS_AS(en.fit(MatrixXd::Zero(5, 2), VectorXd::Zero(4)), std::invalid_argument);
    MatrixXd X = MatrixXd::Zero(5, 2);
    X(0, 0) = std::nan("");
    CHECK_THROWS_AS(en.fit(X, VectorXd::Zero(5)), std::invalid_argument);
}

// --- Regression engine --------------------------------------------------------------

TEST_CASE("engine: exhaustive search finds the true factors out of sample", "[regression]") {
    const Synthetic s = synthetic(300, 8, 0.5, 11);
    auto p = engine_params();
    p->min_factors = 1;
    p->max_factors = 3;
    p->train_fraction = 0.7;
    p->search = SubsetSearch::Exhaustive;
    const RegressionComputation r = RegressionEngine(std::move(p)).run(s.y, s.X);

    CHECK(r.search_used == SubsetSearch::Exhaustive);
    CHECK(r.subsets_evaluated == 8 + 28 + 56);
    CHECK(r.train_rows == 210);
    CHECK(r.test_rows == 90);
    REQUIRE(r.best_by_size.size() == 3);
    CHECK(r.best_by_size[1].factors == std::vector<Index>{0, 3});
    CHECK(r.factor_positions.size() >= 2);
    CHECK(std::find(r.factor_positions.begin(), r.factor_positions.end(), 0) != r.factor_positions.end());
    CHECK(std::find(r.factor_positions.begin(), r.factor_positions.end(), 3) != r.factor_positions.end());
    CHECK(r.factor_names.front() == "x0");
    CHECK(r.best.test_r2 > 0.9);
    CHECK(std::isfinite(r.best.validation_mse));
    CHECK(r.validation_curve.mse.rows() == 2);
    CHECK(r.validation_curve.mse.cols() == 20);
    CHECK_THAT(r.validation_curve.mse.minCoeff(), WithinAbs(r.best.validation_mse, 1e-12));
    CHECK(r.predictions.size() == 300);
    CHECK_THAT(r.residuals.tail(90).squaredNorm() / 90.0, WithinRel(r.best.test_mse, 1e-10));
    CHECK(r.all_converged);
}

TEST_CASE("engine: forward search reaches the same subset", "[regression]") {
    const Synthetic s = synthetic(300, 10, 0.5, 13);
    auto p = engine_params();
    p->min_factors = 2;
    p->max_factors = 2;
    p->search = SubsetSearch::Forward;
    const RegressionComputation r = RegressionEngine(std::move(p)).run(s.y, s.X);
    CHECK(r.search_used == SubsetSearch::Forward);
    CHECK(r.subsets_evaluated == 10 + 9);
    CHECK(r.factor_positions == std::vector<Index>{0, 3});
    CHECK_THAT(r.coefficients(0), WithinAbs(2.0, 0.1));
    CHECK_THAT(r.coefficients(1), WithinAbs(-0.3, 0.05));
}

TEST_CASE("engine: auto search switches to forward above the subset limit", "[regression]") {
    const Synthetic s = synthetic(200, 12, 0.5, 17);
    auto p = engine_params();
    p->max_factors = 3;
    p->max_exhaustive_subsets = 100;  // 12 + 66 + 220 subsets would be needed
    CHECK(RegressionEngine(std::move(p)).run(s.y, s.X).search_used == SubsetSearch::Forward);
}

TEST_CASE("engine: in-sample mode with BIC rejects noise factors", "[regression]") {
    const Synthetic s = synthetic(400, 6, 0.5, 19);
    auto p = engine_params();
    p->min_factors = 1;
    p->max_factors = 4;
    p->train_fraction = 1.0;
    p->search = SubsetSearch::Exhaustive;
    const RegressionComputation r = RegressionEngine(std::move(p)).run(s.y, s.X);
    CHECK(r.test_rows == 0);
    CHECK(std::isnan(r.best.test_mse));
    CHECK(r.best.score == r.best.criterion);
    CHECK(r.factor_positions == std::vector<Index>{0, 3});
}

TEST_CASE("engine: fixed hyperparameters skip validation", "[regression]") {
    const Synthetic s = synthetic(100, 4, 0.5, 23);
    auto p = engine_params();
    p->alphas = {1.0};
    p->lambdas = {0.01};
    p->max_factors = 2;
    const RegressionComputation r = RegressionEngine(std::move(p)).run(s.y, s.X);
    CHECK(r.alpha == 1.0);
    CHECK(r.lambda == 0.01);
    CHECK(std::isnan(r.best.validation_mse));
    CHECK(r.validation_curve.mse.size() == 0);
}

TEST_CASE("engine: walk-forward validation and one-standard-error rule", "[regression]") {
    const Synthetic s = synthetic(250, 5, 1.0, 29);
    auto p = engine_params();
    p->validation = ValidationScheme::WalkForward;
    p->validation_folds = 3;
    p->validation_gap = 5;
    p->max_factors = 2;
    const RegressionComputation plain = RegressionEngine(std::make_unique<RegressionEngineParams>(*p)).run(s.y, s.X);
    p->one_standard_error_rule = true;
    const RegressionComputation sparse = RegressionEngine(std::move(p)).run(s.y, s.X);
    CHECK(sparse.best_by_size[1].lambda >= plain.best_by_size[1].lambda);
}

TEST_CASE("engine: bootstrap intervals cover the true coefficients", "[regression]") {
    const Synthetic s = synthetic(300, 5, 0.5, 31);
    auto p = engine_params();
    p->min_factors = 2;
    p->max_factors = 2;
    p->alphas = {1.0};
    p->bootstrap_samples = 200;
    p->bootstrap_block_size = 5;
    p->num_threads = 4;
    const RegressionComputation r = RegressionEngine(std::move(p)).run(s.y, s.X);
    REQUIRE(r.factor_positions == std::vector<Index>{0, 3});
    REQUIRE(r.bootstrap.has_value());
    const BootstrapSummary& b = *r.bootstrap;
    CHECK(b.samples == 200);
    CHECK(b.coefficients.rows() == 200);
    CHECK(b.lower(0) < 2.0);
    CHECK(b.upper(0) > 2.0);
    CHECK(b.lower(1) < -0.3);
    CHECK(b.upper(1) > -0.3);
    CHECK(b.selection_frequency(0) == 1.0);
    CHECK(b.std_error(0) > 0.0);
    CHECK(b.test_mse_lower <= b.test_mse_mean);
    CHECK(b.test_mse_mean <= b.test_mse_upper);
}

TEST_CASE("engine: results do not depend on the number of threads", "[regression]") {
    const Synthetic s = synthetic(200, 7, 0.8, 37);
    auto p = engine_params();
    p->max_factors = 3;
    p->bootstrap_samples = 30;
    p->bootstrap_block_size = 3;
    p->num_threads = 1;
    const RegressionComputation serial = RegressionEngine(std::make_unique<RegressionEngineParams>(*p)).run(s.y, s.X);
    p->num_threads = 4;
    const RegressionComputation parallel = RegressionEngine(std::move(p)).run(s.y, s.X);
    CHECK(parallel.threads_used > 1);
    CHECK(serial.factor_positions == parallel.factor_positions);
    CHECK(serial.coefficients == parallel.coefficients);
    CHECK(serial.bootstrap->coefficients == parallel.bootstrap->coefficients);
}

TEST_CASE("engine: refit on the full sample keeps out-of-sample predictions", "[regression]") {
    const Synthetic s = synthetic(200, 4, 0.5, 41);
    auto p = engine_params();
    p->max_factors = 2;
    p->refit_on_full_sample = true;
    const RegressionComputation r = RegressionEngine(std::move(p)).run(s.y, s.X);
    CHECK(r.coefficients != r.best.coefficients);
    const MatrixXd Xs = s.X(Eigen::placeholders::all, r.factor_positions);
    const VectorXd train_model = (Xs * r.best.coefficients).array() + r.best.intercept;
    CHECK(r.predictions.isApprox(train_model));
}

TEST_CASE("engine: DataFrame interface keeps names and index", "[regression]") {
    const Synthetic s = synthetic(100, 4, 0.5, 43);
    std::vector<int> index(100);
    std::iota(index.begin(), index.end(), 1000);
    const DataFrame<int> factors(index, {"mkt", "smb", "hml", "mom"}, s.X);
    const TimeSeries<int> target(index, s.y, "fund");
    auto p = engine_params();
    p->max_factors = 2;
    p->train_fraction = 0.8;
    const RegressionResult<int> r = RegressionEngine(std::move(p)).run(target, factors);
    CHECK(r.model.factor_names == std::vector<std::string>{"mkt", "mom"});
    CHECK(r.predictions.index() == index);
    REQUIRE(r.test_start.has_value());
    CHECK(*r.test_start == 1080);

    const TimeSeries<int> shifted(std::vector<int>(index.begin(), index.end() - 1), s.y.head(99));
    auto q = engine_params();
    CHECK_THROWS_AS(RegressionEngine(std::move(q)).run(shifted, factors), std::invalid_argument);
}

TEST_CASE("engine: invalid parameters and data throw", "[regression]") {
    const auto make = [](auto&& edit) {
        auto p = engine_params();
        edit(*p);
        return RegressionEngine(std::move(p));
    };
    CHECK_THROWS_AS(RegressionEngine(nullptr), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.min_factors = 0; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.min_factors = 3; p.max_factors = 2; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.train_fraction = 0.0; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.train_fraction = 1.2; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.alphas = {}; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.alphas = {1.5}; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.lambdas = {-0.1}; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.validation_folds = 1; }), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.bootstrap_block_size = 0; }), std::invalid_argument);

    const Synthetic s = synthetic(30, 4, 0.5, 47);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.max_factors = 5; }).run(s.y, s.X), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams&) {}).run(s.y.head(29), s.X), std::invalid_argument);
    CHECK_THROWS_AS(make([](RegressionEngineParams& p) { p.validation_folds = 25; }).run(s.y, s.X),
                    std::invalid_argument);
}
