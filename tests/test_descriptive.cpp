#include <nabia/stats/descriptive.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace nabia::stats;

TEST_CASE("mean of a simple series", "[stats]") {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    CHECK_THAT(mean(x), WithinAbs(2.5, 1e-12));
}

TEST_CASE("mean of empty input throws", "[stats]") {
    const std::vector<double> x;
    CHECK_THROWS_AS(mean(x), std::invalid_argument);
}

TEST_CASE("sample vs population variance", "[stats]") {
    const std::vector<double> x{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    CHECK_THAT(variance(x, 0), WithinAbs(4.0, 1e-12));
    CHECK_THAT(stddev(x, 0), WithinAbs(2.0, 1e-12));
    CHECK_THAT(variance(x, 1), WithinAbs(32.0 / 7.0, 1e-12));
}

TEST_CASE("autocorrelation", "[stats]") {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    CHECK_THAT(autocorrelation(x, 0), WithinAbs(1.0, 1e-12));
    // mean = 3, denom = 10, numer(lag 1) = (-1*-2)+(0*-1)+(1*0)+(2*1) = 4
    CHECK_THAT(autocorrelation(x, 1), WithinAbs(0.4, 1e-12));
    CHECK_THROWS_AS(autocorrelation(x, 5), std::invalid_argument);
}
