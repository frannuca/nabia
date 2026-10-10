#include <nabia/risk/measures/ExpectedShortfall.h>
#include <nabia/risk/measures/ValueAtRisk.h>
#include <nabia/risk/measures/Volatility.h>
#include <nabia/stats/normal.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// Three assets: vols 10%, 20%, 30%; correlations 0.5, 0.2, -0.3.
MatrixXd covariance() {
    VectorXd vol(3);
    vol << 0.1, 0.2, 0.3;
    MatrixXd corr(3, 3);
    corr << 1.0, 0.5, 0.2,
            0.5, 1.0, -0.3,
            0.2, -0.3, 1.0;
    return vol.asDiagonal() * corr * vol.asDiagonal();
}

VectorXd weights() {
    VectorXd x(3);
    x << 0.5, 0.3, 0.2;
    return x;
}

VectorXd mean_returns() {
    VectorXd mu(3);
    mu << 0.01, 0.02, -0.005;
    return mu;
}

// Central finite-difference gradient of the risk measure.
VectorXd numerical_gradient(const IRiskMeasure& m, const VectorXd& x) {
    constexpr double h = 1e-6;
    VectorXd g(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        VectorXd up = x;
        VectorXd down = x;
        up(i) += h;
        down(i) -= h;
        g(i) = (m.compute_risk(up) - m.compute_risk(down)) / (2.0 * h);
    }
    return g;
}

// Properties every measure must satisfy: Euler decomposition, correct gradient,
// positive homogeneity.
void check_decomposition(const IRiskMeasure& m, const VectorXd& x) {
    const double risk = m.compute_risk(x);
    CHECK_THAT(m.compute_risk_contributions(x).sum(), WithinAbs(risk, 1e-12));
    CHECK((m.compute_marginal_risk_contributions(x) - numerical_gradient(m, x)).lpNorm<Eigen::Infinity>() < 1e-7);
    CHECK_THAT(m.compute_risk(2.5 * x), WithinRel(2.5 * risk, 1e-12));
}

} // namespace

TEST_CASE("normal distribution helpers", "[stats]") {
    using namespace nabia::stats;
    CHECK_THAT(normal_cdf(0.0), WithinAbs(0.5, 1e-15));
    CHECK_THAT(normal_cdf(1.0), WithinAbs(0.8413447460685429, 1e-15));
    CHECK_THAT(normal_pdf(0.0), WithinAbs(0.3989422804014327, 1e-15));
    CHECK_THAT(normal_quantile(0.95), WithinAbs(1.6448536269514715, 1e-13));
    CHECK_THAT(normal_quantile(0.975), WithinAbs(1.9599639845400534, 1e-13));
    CHECK_THAT(normal_quantile(0.01), WithinAbs(-2.3263478740408408, 1e-13));
    CHECK_THAT(normal_quantile(1e-10), WithinAbs(-6.361340902404057, 1e-11));
    for (double p : {1e-8, 0.001, 0.02425, 0.3, 0.5, 0.7, 0.97575, 0.999}) {
        CHECK_THAT(normal_cdf(normal_quantile(p)), WithinRel(p, 1e-12));
    }
    CHECK_THROWS_AS(normal_quantile(0.0), std::invalid_argument);
    CHECK_THROWS_AS(normal_quantile(1.0), std::invalid_argument);
}

TEST_CASE("volatility", "[risk]") {
    const Volatility vol(covariance());
    const VectorXd x = weights();
    const double expected = std::sqrt(x.dot(covariance() * x));
    CHECK_THAT(vol.compute_risk(x), WithinAbs(expected, 1e-15));
    check_decomposition(vol, x);

    SECTION("single asset is its own volatility") {
        VectorXd e1 = VectorXd::Zero(3);
        e1(1) = 1.0;
        CHECK_THAT(vol.compute_risk(e1), WithinAbs(0.2, 1e-15));
    }
    SECTION("zero portfolio has zero risk and zero gradient") {
        CHECK(vol.compute_risk(VectorXd::Zero(3)) == 0.0);
        CHECK(vol.compute_marginal_risk_contributions(VectorXd::Zero(3)).isZero());
    }
}

TEST_CASE("value at risk", "[risk]") {
    const VectorXd x = weights();
    const double sigma = std::sqrt(x.dot(covariance() * x));

    SECTION("zero mean: VaR = z * sigma") {
        const ValueAtRisk var(covariance(), 0.99);
        CHECK_THAT(var.compute_risk(x), WithinAbs(2.3263478740408408 * sigma, 1e-13));
        check_decomposition(var, x);
    }
    SECTION("with expected returns: VaR = -mu'x + z * sigma") {
        const ValueAtRisk var(covariance(), 0.95, mean_returns());
        CHECK_THAT(var.compute_risk(x), WithinAbs(-mean_returns().dot(x) + 1.6448536269514715 * sigma, 1e-13));
        check_decomposition(var, x);
    }
    SECTION("matches a single normal asset") {
        MatrixXd s(1, 1);
        s << 0.04;  // sigma = 20%
        const ValueAtRisk var(s, 0.99);
        CHECK_THAT(var.compute_risk(VectorXd::Ones(1)), WithinAbs(0.2 * 2.3263478740408408, 1e-13));
    }
}

TEST_CASE("expected shortfall", "[risk]") {
    const VectorXd x = weights();
    const double sigma = std::sqrt(x.dot(covariance() * x));

    SECTION("zero mean: ES = phi(z) / (1 - alpha) * sigma") {
        const ExpectedShortfall es(covariance(), 0.975);
        CHECK_THAT(es.compute_risk(x), WithinAbs(2.3378027922014164 * sigma, 1e-13));
        check_decomposition(es, x);
    }
    SECTION("with expected returns") {
        const ExpectedShortfall es(covariance(), 0.99, mean_returns());
        CHECK_THAT(es.compute_risk(x), WithinAbs(-mean_returns().dot(x) + 2.665214220345806 * sigma, 1e-13));
        check_decomposition(es, x);
    }
    SECTION("ES >= VaR at the same level") {
        for (double alpha : {0.9, 0.95, 0.99, 0.999}) {
            const ExpectedShortfall es(covariance(), alpha, mean_returns());
            const ValueAtRisk var(covariance(), alpha, mean_returns());
            CHECK(es.compute_risk(x) > var.compute_risk(x));
        }
    }
    SECTION("ES_97.5 is close to VaR_99 for normal returns (Basel FRTB calibration)") {
        CHECK_THAT(ExpectedShortfall(covariance(), 0.975).compute_risk(x),
                   WithinRel(ValueAtRisk(covariance(), 0.99).compute_risk(x), 0.01));
    }
}

TEST_CASE("risk measures: invalid input", "[risk]") {
    MatrixXd not_square(2, 3);
    not_square.setOnes();
    CHECK_THROWS_AS(Volatility(not_square), std::invalid_argument);

    MatrixXd asymmetric(2, 2);
    asymmetric << 1.0, 0.5, 0.0, 1.0;
    CHECK_THROWS_AS(Volatility(asymmetric), std::invalid_argument);

    MatrixXd indefinite(2, 2);
    indefinite << 1.0, 2.0, 2.0, 1.0;  // eigenvalues 3 and -1
    CHECK_THROWS_AS(Volatility(indefinite), std::invalid_argument);

    CHECK_THROWS_AS(ValueAtRisk(covariance(), 1.0), std::invalid_argument);
    CHECK_THROWS_AS(ExpectedShortfall(covariance(), 0.0), std::invalid_argument);
    CHECK_THROWS_AS(ValueAtRisk(covariance(), 0.99, VectorXd::Zero(2)), std::invalid_argument);

    const Volatility vol(covariance());
    CHECK_THROWS_AS(vol.compute_risk(VectorXd::Ones(2)), std::invalid_argument);
    CHECK_THROWS_AS(vol.compute_risk_contributions(VectorXd::Ones(4)), std::invalid_argument);
}
