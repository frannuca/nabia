#include <nabia/strategy/PerformanceMetrics.h>
#include <nabia/strategy/RiskBudgetingPortfolio.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std::chrono;

namespace {

// Returns whose sample covariance over any window of 4k consecutive rows is
// exactly diag(vol^2) (times a constant) with zero mean: asset i follows a
// column of a 4 x 4 Hadamard matrix, scaled by vol_i.
DataFrame<Date> hadamard_returns(Index rows, const VectorXd& vol) {
    MatrixXd h(4, 3);
    h << 1, 1, 1,
         -1, 1, -1,
         1, -1, -1,
         -1, -1, 1;
    MatrixXd r(rows, vol.size());
    for (Index t = 0; t < rows; ++t) {
        for (Index i = 0; i < vol.size(); ++i) {
            r(t, i) = vol(i) * h(t % 4, i);
        }
    }
    return {daily_dates(sys_days{2026y / January / 1}, rows), {"A", "B", "C"}, r};
}

DataFrame<Date> random_returns(Index rows, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    MatrixXd r(rows, 3);
    for (Index t = 0; t < rows; ++t) {
        const double common = z(rng);
        r(t, 0) = 0.0003 + 0.008 * (0.6 * common + 0.8 * z(rng));
        r(t, 1) = 0.0002 + 0.012 * (0.3 * common + 0.95 * z(rng));
        r(t, 2) = 0.0001 + 0.004 * z(rng);
    }
    return {daily_dates(sys_days{2024y / January / 1}, rows), {"EQ", "CMD", "BND"}, r};
}

std::unique_ptr<RiskBudgetingPortfolioParams> params(Index window, Index frequency) {
    auto p = std::make_unique<RiskBudgetingPortfolioParams>();
    p->covariance_window = window;
    p->rebalance_frequency = frequency;
    return p;
}

} // namespace

TEST_CASE("performance metrics on a known series", "[strategy]") {
    // equity: 1.1, 0.55, 0.66 -> drawdown 0, 0.5, 0.4
    const VectorXd r{{0.1, -0.5, 0.2}};
    const PerformanceMetrics m = compute_performance(r, 252);
    CHECK_THAT(m.total_return, WithinAbs(-0.34, 1e-14));
    CHECK_THAT(m.max_drawdown, WithinAbs(0.5, 1e-14));
    CHECK(m.max_drawdown_duration == 2);
    CHECK_THAT(m.hit_rate, WithinAbs(2.0 / 3.0, 1e-15));
    CHECK(m.best_period == 0.2);
    CHECK(m.worst_period == -0.5);
    CHECK_THAT(m.annualized_return, WithinAbs(std::pow(0.66, 252.0 / 3.0) - 1.0, 1e-12));
    CHECK_THROWS_AS(compute_performance(VectorXd(), 252), std::invalid_argument);
}

TEST_CASE("risk budgeting portfolio: equal risk weights are inverse volatility for uncorrelated assets",
          "[strategy]") {
    const VectorXd vol{{0.01, 0.02, 0.04}};
    const DataFrame<Date> returns = hadamard_returns(40, vol);
    const RiskBudgetingPortfolio engine(params(8, 4));
    const PortfolioResult<Date> res = engine.run(returns);

    const VectorXd inv = vol.cwiseInverse() / vol.cwiseInverse().sum();
    REQUIRE(res.all_converged);
    for (Index k = 0; k < res.target_weights.rows(); ++k) {
        CHECK((res.target_weights.values().row(k).transpose() - inv).lpNorm<Eigen::Infinity>() < 1e-7);
        CHECK((res.risk_contributions.values().row(k).array() - 1.0 / 3.0).abs().maxCoeff() < 1e-7);
    }
    // Time is preserved: results share the input index, trades happen at W-1, W-1+k, ...
    CHECK(res.weights.index() == returns.index());
    CHECK(res.target_weights.time(0) == returns.time(7));
    CHECK(res.target_weights.time(1) == returns.time(11));
    CHECK(res.target_weights.rows() == (40 - 8) / 4 + 1);
}

TEST_CASE("risk budgeting portfolio: no look-ahead", "[strategy]") {
    const DataFrame<Date> base = random_returns(300, 1);
    MatrixXd shocked = base.values();
    shocked.bottomRows(100).array() *= 3.0;  // change everything from row 200 on
    const DataFrame<Date> changed(base.index(), base.columns(), shocked);

    const RiskBudgetingPortfolio engine(params(60, 5));
    const PortfolioResult<Date> a = engine.run(base);
    const PortfolioResult<Date> b = engine.run(changed);
    // Weights held up to and including row 200 were decided with data up to row 199.
    CHECK(a.weights.values().topRows(201) == b.weights.values().topRows(201));
    CHECK(a.weights.values().bottomRows(99) != b.weights.values().bottomRows(99));
    // Nothing is held before the first trade at the close of row 59.
    CHECK(a.weights.values().topRows(60).isZero(0.0));
    CHECK_FALSE(a.weights.values().row(60).isZero(0.0));
}

TEST_CASE("risk budgeting portfolio: accounting identities", "[strategy]") {
    auto p = params(60, 10);
    p->transaction_cost = 0.001;
    p->initial_capital = 1e6;
    const DataFrame<Date> returns = random_returns(400, 2);
    const PortfolioResult<Date> res = RiskBudgetingPortfolio(std::move(p)).run(returns);
    const VectorXd& net = res.net_returns.values();

    // net = (1 + gross)(1 - cost) - 1, with cost = tc * turnover
    CHECK((res.costs.values() - 0.001 * res.turnover.values()).cwiseAbs().maxCoeff() < 1e-15);
    const VectorXd expected = (1.0 + res.gross_returns.values().array()) * (1.0 - res.costs.values().array()) - 1.0;
    CHECK((net - expected).cwiseAbs().maxCoeff() < 1e-15);
    // gross = sum_i w_i r_i with the weights held over the period
    const VectorXd gross = (res.weights.values().cwiseProduct(returns.values())).rowwise().sum();
    CHECK((gross - res.gross_returns.values()).cwiseAbs().maxCoeff() < 1e-15);
    // equity and PnL
    CHECK_THAT(res.equity.values()(399), WithinRel(1e6 * (1.0 + net.array()).prod(), 1e-12));
    CHECK_THAT(res.pnl.values().sum(), WithinRel(res.equity.values()(399) - 1e6, 1e-9));
    // the first trade buys the whole portfolio from cash: turnover = sum of weights = 1
    CHECK_THAT(res.turnover.values()(59), WithinAbs(1.0, 1e-12));
    CHECK_THAT(res.metrics.total_costs, WithinAbs(res.costs.values().sum(), 1e-15));
    CHECK_THAT(res.metrics.max_drawdown, WithinAbs(res.drawdown.values().maxCoeff(), 1e-15));
    CHECK(res.metrics.periods == 400 - 59);
    CHECK(res.metrics.rebalances == res.target_weights.rows());
}

TEST_CASE("risk budgeting portfolio: drift and constant weights between trades", "[strategy]") {
    const DataFrame<Date> returns = random_returns(120, 3);
    SECTION("drifting weights stay fully invested and follow returns") {
        const PortfolioResult<Date> res = RiskBudgetingPortfolio(params(40, 20)).run(returns);
        const MatrixXd& w = res.weights.values();
        for (Index t = 41; t < 120; ++t) {
            CHECK_THAT(w.row(t).sum(), WithinAbs(1.0, 1e-12));  // no leverage, no cash
        }
        // Between trades (row 41 -> 42): w' = w (1 + r) / (1 + w.r)
        const VectorXd r = returns.values().row(41).transpose();
        const VectorXd drifted = w.row(41).transpose().cwiseProduct(VectorXd::Ones(3) + r) / (1.0 + w.row(41).dot(r));
        CHECK((w.row(42).transpose() - drifted).lpNorm<Eigen::Infinity>() < 1e-14);
    }
    SECTION("constant weights between trades") {
        auto p = params(40, 20);
        p->drift_weights = false;
        const PortfolioResult<Date> res = RiskBudgetingPortfolio(std::move(p)).run(returns);
        const MatrixXd& w = res.weights.values();
        for (Index t = 41; t <= 59; ++t) {  // trades at rows 39 and 59
            CHECK(w.row(t) == w.row(40));
        }
        CHECK(w.row(60) != w.row(59));
    }
}

TEST_CASE("risk budgeting portfolio: transaction costs reduce returns", "[strategy]") {
    const DataFrame<Date> returns = random_returns(250, 4);
    auto cheap = params(50, 1);
    auto costly = params(50, 1);
    costly->transaction_cost = 0.002;
    const PortfolioResult<Date> a = RiskBudgetingPortfolio(std::move(cheap)).run(returns);
    const PortfolioResult<Date> b = RiskBudgetingPortfolio(std::move(costly)).run(returns);
    CHECK(a.weights.values() == b.weights.values());  // costs do not change the targets
    CHECK(b.equity.values()(249) < a.equity.values()(249));
    CHECK(b.metrics.total_costs > 0.0);
    CHECK(a.metrics.total_costs == 0.0);
    // Daily rebalancing trades more than monthly rebalancing.
    const PortfolioResult<Date> monthly = RiskBudgetingPortfolio(params(50, 21)).run(returns);
    CHECK(a.metrics.annualized_turnover > monthly.metrics.annualized_turnover);
}

TEST_CASE("risk budgeting portfolio: volatility targeting", "[strategy]") {
    // Exact covariance -> exact predicted volatility of the ERC portfolio.
    const VectorXd vol{{0.01, 0.02, 0.04}};
    const DataFrame<Date> returns = hadamard_returns(40, vol);
    const RiskBudgetingPortfolio unlevered(params(8, 4));
    const PortfolioResult<Date> base = unlevered.run(returns);
    const VectorXd w = base.target_weights.values().row(0).transpose();
    // Sample covariance of the window: vol^2 * 8/7 on the diagonal (unbiased, 8 rows of +-vol).
    const double predicted = std::sqrt(w.cwiseProduct(vol).squaredNorm() * 8.0 / 7.0 * 252.0);

    auto p = params(8, 4);
    p->target_volatility = 0.5 * predicted;
    const PortfolioResult<Date> half = RiskBudgetingPortfolio(std::move(p)).run(returns);
    CHECK_THAT(half.leverage.values()(0), WithinAbs(0.5, 1e-9));
    CHECK_THAT(half.target_weights.values().row(0).sum(), WithinAbs(0.5, 1e-9));

    auto q = params(8, 4);
    q->target_volatility = 10.0 * predicted;
    q->max_leverage = 3.0;
    const PortfolioResult<Date> capped = RiskBudgetingPortfolio(std::move(q)).run(returns);
    CHECK_THAT(capped.leverage.values()(0), WithinAbs(3.0, 1e-12));
}

TEST_CASE("risk budgeting portfolio: other risk measures and estimators", "[strategy]") {
    const DataFrame<Date> returns = random_returns(200, 5);
    const PortfolioResult<Date> by_vol = RiskBudgetingPortfolio(params(60, 20)).run(returns);

    auto es = params(60, 20);
    es->risk_measure = PortfolioRiskMeasure::ExpectedShortfall;
    es->confidence_level = 0.975;
    const PortfolioResult<Date> by_es = RiskBudgetingPortfolio(std::move(es)).run(returns);
    // Zero-mean Gaussian ES is a multiple of volatility: same weights.
    CHECK((by_es.target_weights.values() - by_vol.target_weights.values()).cwiseAbs().maxCoeff() < 1e-6);

    auto var = params(60, 20);
    var->risk_measure = PortfolioRiskMeasure::ValueAtRisk;
    var->use_expected_returns = true;
    var->budgets = VectorXd{{0.5, 0.25, 0.25}};
    const PortfolioResult<Date> by_var = RiskBudgetingPortfolio(std::move(var)).run(returns);
    REQUIRE(by_var.all_converged);
    CHECK((by_var.risk_contributions.values().rowwise() - Eigen::RowVector3d(0.5, 0.25, 0.25)).cwiseAbs().maxCoeff() <
          1e-6);

    auto ewma = params(60, 20);
    ewma->covariance_estimator = CovarianceEstimator::Exponential;
    ewma->ewma_halflife = 10.0;
    const PortfolioResult<Date> by_ewma = RiskBudgetingPortfolio(std::move(ewma)).run(returns);
    CHECK(by_ewma.all_converged);
    CHECK(by_ewma.target_weights.values() != by_vol.target_weights.values());
}

TEST_CASE("risk budgeting portfolio: integer index and invalid input", "[strategy]") {
    const DataFrame<Date> dated = random_returns(80, 6);
    const DataFrame<Step> bars(integer_index(80), dated.columns(), dated.values());
    const PortfolioResult<Step> res = RiskBudgetingPortfolio(params(30, 5)).run(bars);
    CHECK(res.target_weights.time(0) == 29);

    CHECK_THROWS_AS(RiskBudgetingPortfolio(params(1, 1)), std::invalid_argument);
    CHECK_THROWS_AS(RiskBudgetingPortfolio(params(10, 0)), std::invalid_argument);
    auto bad_cost = params(10, 1);
    bad_cost->transaction_cost = -0.1;
    CHECK_THROWS_AS(RiskBudgetingPortfolio(std::move(bad_cost)), std::invalid_argument);

    const RiskBudgetingPortfolio engine(params(100, 1));
    CHECK_THROWS_AS(engine.run(bars), std::invalid_argument);  // fewer rows than the window
    MatrixXd holes = dated.values();
    holes(5, 1) = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(RiskBudgetingPortfolio(params(30, 1)).run(DataFrame<Date>(dated.index(), dated.columns(), holes)),
                    std::invalid_argument);
    auto wrong_budgets = params(30, 1);
    wrong_budgets->budgets = VectorXd::Ones(2);
    CHECK_THROWS_AS(RiskBudgetingPortfolio(std::move(wrong_budgets)).run(dated), std::invalid_argument);
}

TEST_CASE("risk budgeting portfolio: multithreaded solves give identical results", "[strategy]") {
    const DataFrame<Date> returns = random_returns(500, 7);
    auto run = [&](int threads) {
        auto p = params(60, 2);
        p->num_threads = threads;
        p->transaction_cost = 0.0005;
        p->target_volatility = 0.10;
        return RiskBudgetingPortfolio(std::move(p)).run(returns);
    };
    const PortfolioResult<Date> serial = run(1);
    CHECK(serial.threads_used == 1);
    for (int threads : {2, 3, 8}) {
        const PortfolioResult<Date> parallel = run(threads);
        CHECK(parallel.threads_used == threads);
        CHECK(parallel.target_weights.values() == serial.target_weights.values());
        CHECK(parallel.weights.values() == serial.weights.values());
        CHECK(parallel.equity.values() == serial.equity.values());
    }
    // 0 = hardware concurrency, capped by the number of tasks (blocks of rebalance dates).
    CHECK(run(0).threads_used >= 1);
    auto few = params(60, 200);  // 3 rebalances -> a single block
    few->num_threads = 8;
    CHECK(RiskBudgetingPortfolio(std::move(few)).run(returns).threads_used == 1);

    auto negative = params(60, 1);
    negative->num_threads = -1;
    CHECK_THROWS_AS(RiskBudgetingPortfolio(std::move(negative)), std::invalid_argument);
}

TEST_CASE("risk budgeting portfolio: solver errors reach the caller from worker threads", "[strategy]") {
    // From row 300 on the expected returns are so large that VaR is negative for
    // every long-only portfolio: risk budgeting is undefined there and throws.
    MatrixXd r = random_returns(400, 8).values();
    r.bottomRows(100).array() += 0.5;
    const DataFrame<Date> returns(daily_dates(sys_days{2024y / January / 1}, 400), {"EQ", "CMD", "BND"}, r);
    auto p = params(60, 1);
    p->risk_measure = PortfolioRiskMeasure::ValueAtRisk;
    p->use_expected_returns = true;
    p->num_threads = 4;
    CHECK_THROWS_AS(RiskBudgetingPortfolio(std::move(p)).run(returns), std::invalid_argument);
}

TEST_CASE("risk budgeting portfolio: time-varying budget schedule", "[strategy]") {
    const DataFrame<Date> returns = random_returns(300, 9);  // rows from 2024-01-01, daily
    const Eigen::RowVector3d first{0.6, 0.3, 0.1};
    const Eigen::RowVector3d second{0.2, 0.2, 0.6};
    // Decided on row 100's date, and on a date between rows 199 and 200 (12:00 not representable
    // with Date, so use a date missing from the index: drop row 200's date from the returns).
    MatrixXd schedule_values(2, 3);
    schedule_values << 6.0, 3.0, 1.0,   // normalized by the engine
                       0.2, 0.2, 0.6;
    const Date d1 = returns.time(100);
    const Date d2 = returns.time(200);
    const DataFrame<Date> schedule({d1, d2}, returns.columns(), schedule_values);
    std::vector<Date> kept;
    std::vector<Index> rows;
    for (Index t = 0; t < 300; ++t) {
        if (t != 200) {
            kept.push_back(returns.time(t));
            rows.push_back(t);
        }
    }
    MatrixXd values(299, 3);
    for (Index k = 0; k < 299; ++k) {
        values.row(k) = returns.values().row(rows[static_cast<std::size_t>(k)]);
    }
    const DataFrame<Date> gappy(kept, returns.columns(), values);  // d2 is not in the index

    auto run = [&](int threads) {
        auto p = params(60, 7);
        p->num_threads = threads;
        p->budgets = VectorXd{{1.0, 1.0, 1.0}};
        return RiskBudgetingPortfolio(std::move(p)).run(gappy, schedule);
    };
    const PortfolioResult<Date> res = run(1);
    REQUIRE(res.all_converged);
    for (Index k = 0; k < res.risk_contributions.rows(); ++k) {
        const Date t = res.risk_contributions.time(k);
        const Eigen::RowVector3d expected = t < d1 ? Eigen::RowVector3d::Constant(1.0 / 3.0) : (t < d2 ? first : second);
        CHECK((res.risk_contributions.values().row(k) - expected).cwiseAbs().maxCoeff() < 1e-6);
    }
    // Some rebalances fall in each regime.
    const auto count = [&](auto pred) {
        Index n = 0;
        for (Index k = 0; k < res.risk_contributions.rows(); ++k) {
            n += pred(res.risk_contributions.time(k)) ? 1 : 0;
        }
        return n;
    };
    CHECK(count([&](Date t) { return t < d1; }) > 0);
    CHECK(count([&](Date t) { return t >= d1 && t < d2; }) > 0);
    CHECK(count([&](Date t) { return t >= d2; }) > 0);
    CHECK(run(4).target_weights.values() == res.target_weights.values());

    MatrixXd bad = schedule_values;
    bad(1, 2) = 0.0;
    CHECK_THROWS_AS(RiskBudgetingPortfolio(params(60, 7)).run(gappy, DataFrame<Date>({d1, d2}, returns.columns(), bad)),
                    std::invalid_argument);
    CHECK_THROWS_AS(RiskBudgetingPortfolio(params(60, 7)).run(gappy, schedule.select({"CMD", "EQ", "BND"})),
                    std::invalid_argument);
}
