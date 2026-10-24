#include <nabia/dataframe/DataFrame.h>
#include <nabia/dataframe/TimeSeries.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std::chrono;

namespace {

DataFrame<Date> prices() {
    MatrixXd v(4, 2);
    v << 100.0, 50.0,
         110.0, 50.0,
          99.0, 55.0,
          99.0, 44.0;
    return {daily_dates(sys_days{2026y / January / 1}, 4), {"SPX", "BUND"}, v};
}

} // namespace

TEST_CASE("dataframe: construction and validation", "[dataframe]") {
    const DataFrame<Date> df = prices();
    CHECK(df.rows() == 4);
    CHECK(df.cols() == 2);
    CHECK(df.time(1) == sys_days{2026y / January / 2});

    CHECK_THROWS_AS(DataFrame<Step>({0, 2, 1}, {"a"}, MatrixXd::Zero(3, 1)), std::invalid_argument);  // unsorted
    CHECK_THROWS_AS(DataFrame<Step>({0, 0, 1}, {"a"}, MatrixXd::Zero(3, 1)), std::invalid_argument);  // duplicate time
    CHECK_THROWS_AS(DataFrame<Step>({0, 1}, {"a"}, MatrixXd::Zero(3, 1)), std::invalid_argument);     // shape
    CHECK_THROWS_AS(DataFrame<Step>({0, 1}, {"a", "a"}, MatrixXd::Zero(2, 2)), std::invalid_argument); // names
    CHECK_THROWS_AS(TimeSeries<Step>({0, 1}, VectorXd::Zero(3)), std::invalid_argument);
}

TEST_CASE("dataframe: columns, lookup and slicing by time", "[dataframe]") {
    const DataFrame<Date> df = prices();
    const TimeSeries<Date> bund = df.column("BUND");
    CHECK(bund.name() == "BUND");
    CHECK(bund.at(sys_days{2026y / January / 3}) == 55.0);
    CHECK_THROWS_AS(bund.at(sys_days{2025y / December / 31}), std::out_of_range);
    CHECK_THROWS_AS(df.column("NOPE"), std::out_of_range);
    CHECK(df.row_at(sys_days{2026y / January / 2}) == Eigen::RowVector2d(110.0, 50.0));

    const DataFrame<Date> mid = df.between(sys_days{2026y / January / 2}, sys_days{2026y / January / 3});
    CHECK(mid.rows() == 2);
    CHECK(mid.time(0) == sys_days{2026y / January / 2});
    CHECK(df.between(sys_days{2027y / January / 1}, sys_days{2027y / February / 1}).empty());
    CHECK(df.head(1).values() == df.values().topRows(1));
    CHECK(df.tail(10).rows() == 4);
    CHECK_THROWS_AS(df.slice(2, 1), std::out_of_range);

    const DataFrame<Date> swapped = df.select({"BUND", "SPX"});
    CHECK(swapped.columns() == std::vector<std::string>{"BUND", "SPX"});
    CHECK(swapped.values().col(0) == df.values().col(1));
}

TEST_CASE("dataframe: returns", "[dataframe]") {
    const DataFrame<Date> r = prices().pct_change();
    REQUIRE(r.rows() == 3);
    CHECK(r.time(0) == sys_days{2026y / January / 2});  // return over (Jan 1, Jan 2]
    CHECK_THAT(r.values()(0, 0), WithinAbs(0.10, 1e-15));
    CHECK_THAT(r.values()(1, 0), WithinAbs(-0.10, 1e-15));
    CHECK_THAT(r.values()(2, 1), WithinAbs(-0.20, 1e-15));
    const DataFrame<Date> lr = prices().log_returns();
    CHECK_THAT(lr.values()(1, 1), WithinAbs(std::log(1.1), 1e-15));
}

TEST_CASE("dataframe: dropping rows with missing values", "[dataframe]") {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double inf = std::numeric_limits<double>::infinity();
    MatrixXd v(5, 2);
    v << 1.0, 10.0,
         nan, 11.0,
         3.0, 12.0,
         4.0, inf,
         5.0, 14.0;
    const DataFrame<Step> df(integer_index(5), {"a", "b"}, v);
    const DataFrame<Step> clean = df.drop_missing();
    CHECK(clean.index() == std::vector<Step>{0, 2, 4});
    CHECK(clean.values() == (MatrixXd(3, 2) << 1.0, 10.0, 3.0, 12.0, 5.0, 14.0).finished());
    CHECK(clean.columns() == df.columns());
    CHECK(DataFrame<Step>(integer_index(1), {"a"}, MatrixXd::Constant(1, 1, nan)).drop_missing().empty());
}

TEST_CASE("dataframe: integer and timestamp indices", "[dataframe]") {
    const DataFrame<Step> bars(integer_index(3, 10), {"x"}, MatrixXd::Constant(3, 1, 2.0));
    CHECK(bars.position(11) == 1);
    CHECK_FALSE(bars.position(13).has_value());

    const Timestamp t0 = sys_days{2026y / March / 2} + 9h + 30min;
    const TimeSeries<Timestamp> ticks({t0, t0 + 1s, t0 + 2s}, VectorXd{{1.0, 2.0, 3.0}}, "price");
    CHECK(ticks.between(t0 + 500ms, t0 + 5s).size() == 2);
    CHECK(ticks.tail(1)[0] == 3.0);
}
