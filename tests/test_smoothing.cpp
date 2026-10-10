#include <nabia/signal/smoothing.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <stdexcept>
#include <vector>

using Catch::Matchers::Approx;
using namespace nabia::signal;

TEST_CASE("moving average", "[signal]") {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    CHECK_THAT(moving_average(x, 3), Approx(std::vector<double>{2.0, 3.0, 4.0}));
    CHECK_THAT(moving_average(x, 1), Approx(x));
    CHECK_THROWS_AS(moving_average(x, 0), std::invalid_argument);
    CHECK_THROWS_AS(moving_average(x, 6), std::invalid_argument);
}

TEST_CASE("exponential moving average", "[signal]") {
    const std::vector<double> x{10.0, 20.0, 30.0};
    // y0 = 10, y1 = 0.5*20 + 0.5*10 = 15, y2 = 0.5*30 + 0.5*15 = 22.5
    CHECK_THAT(exponential_moving_average(x, 0.5), Approx(std::vector<double>{10.0, 15.0, 22.5}));
    CHECK_THROWS_AS(exponential_moving_average(x, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(exponential_moving_average(x, 1.5), std::invalid_argument);
}

TEST_CASE("diff", "[signal]") {
    const std::vector<double> x{1.0, 4.0, 9.0, 16.0};
    CHECK_THAT(diff(x), Approx(std::vector<double>{3.0, 5.0, 7.0}));
    CHECK_THAT(diff(x, 2), Approx(std::vector<double>{2.0, 2.0}));
    CHECK(diff(x, 10).empty());
}
