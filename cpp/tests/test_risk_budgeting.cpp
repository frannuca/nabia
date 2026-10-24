#include <nabia/risk/RiskBudgeting.h>
#include <nabia/risk/measures/ExpectedShortfall.h>
#include <nabia/risk/measures/ValueAtRisk.h>
#include <nabia/risk/measures/Volatility.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// Only IRiskMeasure subclasses are accepted as the template argument.
template <typename M>
concept Budgetable = requires { typename RiskBudgeting<M>; };
static_assert(Budgetable<Volatility>);
static_assert(Budgetable<ExpectedShortfall>);
static_assert(!Budgetable<int>);
static_assert(!Budgetable<MatrixXd>);

constexpr double kTol = 1e-7;

MatrixXd covariance() {
    VectorXd vol(4);
    vol << 0.10, 0.15, 0.25, 0.30;
    MatrixXd corr(4, 4);
    corr << 1.0, 0.6, 0.3, 0.1,
            0.6, 1.0, 0.4, 0.2,
            0.3, 0.4, 1.0, 0.5,
            0.1, 0.2, 0.5, 1.0;
    return vol.asDiagonal() * corr * vol.asDiagonal();
}

std::unique_ptr<RiskBudgetingParams> make_params(VectorXd budgets = VectorXd()) {
    auto p = std::make_unique<RiskBudgetingParams>();
    p->budgets = std::move(budgets);
    return p;
}

template <typename Measure>
void check_budgets(const RiskBudgeting<Measure>& rb, const VectorXd& x, const VectorXd& b) {
    const RiskBudgetingResult& r = rb.result();
    REQUIRE(r.converged);
    CHECK(x == r.weights);
    CHECK_THAT(x.sum(), WithinAbs(1.0, 1e-14));
    CHECK((x.array() > 0.0).all());
    CHECK((r.relative_contributions - b).lpNorm<Eigen::Infinity>() < kTol);
    CHECK_THAT(r.risk_contributions.sum(), WithinAbs(r.risk, 1e-12));
    CHECK_THAT(r.risk, WithinAbs(rb.measure().compute_risk(x), 1e-14));
}

} // namespace

TEST_CASE("risk budgeting: uncorrelated assets get inverse-volatility weights", "[risk]") {
    VectorXd vol(3);
    vol << 0.1, 0.2, 0.4;
    const MatrixXd cov = vol.cwiseAbs2().asDiagonal();
    RiskBudgeting<Volatility> rb(make_params(), Volatility(cov));
    const VectorXd x = rb.optimize(std::nullopt);
    const VectorXd inv = vol.cwiseInverse();
    CHECK((x - inv / inv.sum()).lpNorm<Eigen::Infinity>() < kTol);
    check_budgets(rb, x, VectorXd::Constant(3, 1.0 / 3.0));
}

TEST_CASE("risk budgeting: two-asset ERC is inverse volatility for any correlation", "[risk]") {
    MatrixXd cov(2, 2);
    cov << 0.04, -0.7 * 0.2 * 0.5, -0.7 * 0.2 * 0.5, 0.25;  // vols 20%, 50%, rho = -0.7
    RiskBudgeting<Volatility> rb(make_params(), Volatility(cov));
    const VectorXd x = rb.optimize(std::nullopt);
    CHECK_THAT(x(0), WithinAbs(0.5 / 0.7, kTol));
    CHECK_THAT(x(1), WithinAbs(0.2 / 0.7, kTol));
}

TEST_CASE("risk budgeting: custom budgets with each risk measure", "[risk]") {
    VectorXd budgets(4);
    budgets << 4.0, 3.0, 2.0, 1.0;  // normalized to 0.4, 0.3, 0.2, 0.1
    const VectorXd b = budgets / budgets.sum();
    VectorXd mu(4);
    mu << 0.02, 0.03, 0.05, 0.06;

    SECTION("volatility") {
        RiskBudgeting<Volatility> rb(make_params(budgets), Volatility(covariance()));
        check_budgets(rb, rb.optimize(std::nullopt), b);
    }
    SECTION("value at risk with expected returns") {
        RiskBudgeting<ValueAtRisk> rb(make_params(budgets), ValueAtRisk(covariance(), 0.99, mu));
        check_budgets(rb, rb.optimize(std::nullopt), b);
    }
    SECTION("expected shortfall with expected returns") {
        RiskBudgeting<ExpectedShortfall> rb(make_params(budgets), ExpectedShortfall(covariance(), 0.975, mu));
        check_budgets(rb, rb.optimize(std::nullopt), b);
    }
    SECTION("zero-mean VaR and ES give the volatility portfolio") {
        RiskBudgeting<Volatility> by_vol(make_params(budgets), Volatility(covariance()));
        RiskBudgeting<ExpectedShortfall> by_es(make_params(budgets), ExpectedShortfall(covariance(), 0.975));
        const VectorXd x_vol = by_vol.optimize(std::nullopt);
        CHECK((by_es.optimize(std::nullopt) - x_vol).lpNorm<Eigen::Infinity>() < kTol);
    }
}

TEST_CASE("risk budgeting: initial guess does not change the solution", "[risk]") {
    RiskBudgeting<Volatility> rb(make_params(), Volatility(covariance()));
    const VectorXd from_default = rb.optimize(std::nullopt);
    VectorXd guess(4);
    guess << 10.0, 1.0, 0.1, 5.0;  // any positive scale
    const VectorXd from_guess = rb.optimize(guess);
    REQUIRE(rb.result().converged);
    CHECK((from_guess - from_default).lpNorm<Eigen::Infinity>() < kTol);
}

TEST_CASE("risk budgeting: invalid input", "[risk]") {
    VectorXd negative(4);
    negative << 0.5, 0.5, -0.1, 0.1;
    RiskBudgeting<Volatility> bad_budget(make_params(negative), Volatility(covariance()));
    CHECK_THROWS_AS(bad_budget.optimize(std::nullopt), std::invalid_argument);

    RiskBudgeting<Volatility> wrong_size(make_params(VectorXd::Ones(3)), Volatility(covariance()));
    CHECK_THROWS_AS(wrong_size.optimize(std::nullopt), std::invalid_argument);

    RiskBudgeting<Volatility> rb(make_params(), Volatility(covariance()));
    CHECK_THROWS_AS(rb.optimize(VectorXd::Zero(4)), std::invalid_argument);
    CHECK_THROWS_AS(rb.optimize(VectorXd::Ones(3)), std::invalid_argument);

    // Expected returns large enough that every long-only portfolio has negative VaR.
    RiskBudgeting<ValueAtRisk> negative_risk(make_params(), ValueAtRisk(covariance(), 0.6, VectorXd::Constant(4, 1.0)));
    CHECK_THROWS_AS(negative_risk.optimize(std::nullopt), std::invalid_argument);
}
