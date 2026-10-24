#include <nabia/forecast/exponential_smoothing.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Eigen::VectorXd;
using nabia::forecast::SimpleExponentialSmoothing;

TEST_CASE("SES rejects invalid alpha", "[forecast]") {
    CHECK_THROWS_AS(SimpleExponentialSmoothing(0.0), std::invalid_argument);
    CHECK_THROWS_AS(SimpleExponentialSmoothing(1.1), std::invalid_argument);
}

TEST_CASE("SES forecast before fit throws", "[forecast]") {
    const SimpleExponentialSmoothing model(0.5);
    CHECK_FALSE(model.is_fitted());
    CHECK_THROWS_AS(model.forecast(3), std::logic_error);
}

TEST_CASE("SES fit and forecast", "[forecast]") {
    SimpleExponentialSmoothing model(0.5);
    model.fit(VectorXd{{10.0, 20.0, 30.0}});

    // levels: 10 -> 15 -> 22.5 ; fitted = level before each observation
    CHECK(model.fitted_values() == VectorXd{{10.0, 10.0, 15.0}});
    CHECK_THAT(model.sse(), WithinAbs(0.0 + 100.0 + 225.0, 1e-12));
    const Forecast fc = model.forecast(2);
    CHECK(fc.mean == VectorXd{{22.5, 22.5}});
    // ETS(A,N,N) variance: sigma^2 (1 + (h-1) alpha^2), sigma^2 = SSE / T = 325 / 3
    CHECK_THAT(fc.variance(0), WithinAbs(325.0 / 3.0, 1e-12));
    CHECK_THAT(fc.variance(1), WithinAbs(325.0 / 3.0 * 1.25, 1e-12));
    CHECK((fc.lower(0.95).array() < fc.mean.array()).all());
    CHECK((fc.upper(0.95).array() > fc.mean.array()).all());
}

TEST_CASE("SES with alpha = 1 is a naive forecast", "[forecast]") {
    SimpleExponentialSmoothing model(1.0);
    model.fit(VectorXd{{3.0, 7.0, 4.0}});
    CHECK(model.forecast(1).mean == VectorXd{{4.0}});
}
