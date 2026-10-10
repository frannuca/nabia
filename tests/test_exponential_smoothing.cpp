#include <nabia/forecast/exponential_smoothing.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <stdexcept>
#include <vector>

using Catch::Matchers::Approx;
using Catch::Matchers::WithinAbs;
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
    const std::vector<double> y{10.0, 20.0, 30.0};
    model.fit(y);

    // levels: 10 -> 15 -> 22.5 ; fitted = level before each observation
    CHECK_THAT(model.fitted(), Approx(std::vector<double>{10.0, 10.0, 15.0}));
    CHECK_THAT(model.sse(), WithinAbs(0.0 + 100.0 + 225.0, 1e-12));
    CHECK_THAT(model.forecast(2), Approx(std::vector<double>{22.5, 22.5}));
}

TEST_CASE("SES with alpha = 1 is a naive forecast", "[forecast]") {
    SimpleExponentialSmoothing model(1.0);
    const std::vector<double> y{3.0, 7.0, 4.0};
    model.fit(y);
    CHECK_THAT(model.forecast(1), Approx(std::vector<double>{4.0}));
}
