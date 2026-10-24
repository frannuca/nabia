#include "config.hpp"
#include "csv_io.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using namespace std::chrono;
using nabia::app::BacktestConfig;

namespace {

/// A file in the temporary directory, removed when the test ends.
class TempFile {
public:
    TempFile(const std::string& name, const std::string& contents)
        : path_(std::filesystem::temp_directory_path() / ("nabia_" + name)) {
        std::ofstream(path_) << contents;
    }
    explicit TempFile(const std::string& name) : path_(std::filesystem::temp_directory_path() / ("nabia_" + name)) {}
    ~TempFile() { std::filesystem::remove(path_); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        lines.push_back(line);
    }
    return lines;
}

std::string error_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST_CASE("app: dates", "[app]") {
    CHECK(nabia::app::parse_date("2024-02-29") == sys_days{2024y / February / 29});
    CHECK(nabia::app::format_date(sys_days{2024y / March / 5}) == "2024-03-05");
    CHECK_THROWS_AS(nabia::app::parse_date("2023-02-29"), std::invalid_argument);  // not a leap year
    CHECK_THROWS_AS(nabia::app::parse_date("2024/01/02"), std::invalid_argument);
    CHECK_THROWS_AS(nabia::app::parse_date("24-01-02"), std::invalid_argument);
}

TEST_CASE("app: reading a prices CSV", "[app]") {
    const TempFile file("prices_ok.csv",
                        "date,\"SPX\", BUND \r\n"
                        "2024-01-02,100,50\r\n"
                        "\n"
                        "2024-01-03,110.5,49.5\r\n");
    const DataFrame<Date> prices = nabia::app::read_prices_csv(file.path());
    CHECK(prices.columns() == std::vector<std::string>{"SPX", "BUND"});
    CHECK(prices.rows() == 2);
    CHECK(prices.time(1) == sys_days{2024y / January / 3});
    CHECK(prices.values()(1, 0) == 110.5);
    CHECK(prices.values()(1, 1) == 49.5);

    SECTION("missing prices are read as NaN and unaligned dates can be dropped") {
        // A has no price on Jan 3 and Jan 5, B none on Jan 4 (other market holidays).
        const TempFile gaps("prices_gaps.csv",
                            "date,A,B\n"
                            "2024-01-02,100,50\n"
                            "2024-01-03,,51\n"
                            "2024-01-04,102,NA\n"
                            "2024-01-05,#N/A,null\n"
                            "2024-01-08,104,53\n");
        const DataFrame<Date> raw = nabia::app::read_prices_csv(gaps.path());
        REQUIRE(raw.rows() == 5);
        CHECK(std::isnan(raw.values()(1, 0)));
        CHECK(std::isnan(raw.values()(2, 1)));
        CHECK(raw.values()(2, 0) == 102.0);

        const DataFrame<Date> aligned = raw.drop_missing();
        REQUIRE(aligned.rows() == 2);
        CHECK(aligned.time(0) == sys_days{2024y / January / 2});
        CHECK(aligned.time(1) == sys_days{2024y / January / 8});
        // Returns then span the gap between consecutive aligned dates.
        CHECK(aligned.pct_change().values().isApprox(Eigen::RowVector2d(0.04, 0.06), 1e-12));
    }
    SECTION("errors name the file and line") {
        const TempFile garbage("prices_garbage.csv", "date,A,B\n2024-01-02,1,2\n2024-01-03,abc,2\n");
        CHECK_THAT(error_of([&] { (void)nabia::app::read_prices_csv(garbage.path()); }),
                   ContainsSubstring(":3:") && ContainsSubstring("price of A"));
        const TempFile unsorted("prices_unsorted.csv", "date,A\n2024-01-03,1\n2024-01-02,1\n");
        CHECK_THAT(error_of([&] { (void)nabia::app::read_prices_csv(unsorted.path()); }),
                   ContainsSubstring("strictly increasing"));
        const TempFile short_row("prices_short.csv", "date,A,B\n2024-01-02,1\n");
        CHECK_THAT(error_of([&] { (void)nabia::app::read_prices_csv(short_row.path()); }),
                   ContainsSubstring("expected 3 fields"));
        const TempFile negative("prices_negative.csv", "date,A\n2024-01-02,-1\n");
        CHECK_THROWS_AS(nabia::app::read_prices_csv(negative.path()), std::runtime_error);
        CHECK_THROWS_AS(nabia::app::read_prices_csv("/no/such/file.csv"), std::runtime_error);
    }
}

TEST_CASE("app: parsing the JSON config", "[app]") {
    const BacktestConfig c = nabia::app::parse_config(R"({
        "start": "2020-01-01", "covariance_window": 90, "covariance_estimator": "exponential",
        "ewma_halflife": 30.5, "rebalance_frequency": 5, "risk_measure": "expected_shortfall",
        "confidence_level": 0.975, "budgets": {"A": 2, "B": 1}, "target_volatility": 0.12,
        "max_leverage": 1.5, "transaction_cost": 0.001, "drift_weights": false,
        "initial_capital": 1e6, "periods_per_year": 260, "num_threads": 4 })");
    const RiskBudgetingPortfolioParams& p = c.params;
    CHECK(c.start == sys_days{2020y / January / 1});
    CHECK_FALSE(c.end.has_value());
    CHECK(p.covariance_window == 90);
    CHECK(p.covariance_estimator == CovarianceEstimator::Exponential);
    CHECK(p.ewma_halflife == 30.5);
    CHECK(p.rebalance_frequency == 5);
    CHECK(p.risk_measure == PortfolioRiskMeasure::ExpectedShortfall);
    CHECK(p.confidence_level == 0.975);
    CHECK(p.target_volatility == 0.12);
    CHECK(p.max_leverage == 1.5);
    CHECK(p.transaction_cost == 0.001);
    CHECK_FALSE(p.drift_weights);
    CHECK(p.initial_capital == 1e6);
    CHECK(p.periods_per_year == 260);
    CHECK(p.num_threads == 4);
    CHECK(nabia::app::resolve_budgets(c, {"B", "A"}) == Eigen::Vector2d(1.0, 2.0));  // by name, any order

    SECTION("defaults") {
        const BacktestConfig d = nabia::app::parse_config("{}");
        CHECK(d.params.covariance_window == RiskBudgetingPortfolioParams{}.covariance_window);
        CHECK(nabia::app::resolve_budgets(d, {"A", "B"}).size() == 0);  // equal risk contribution
        CHECK_FALSE(nabia::app::parse_config(R"({"target_volatility": null})").params.target_volatility);
    }
    SECTION("budget lists and mismatches") {
        const BacktestConfig list = nabia::app::parse_config(R"({"budgets": [0.5, 0.25, 0.25]})");
        CHECK(nabia::app::resolve_budgets(list, {"A", "B", "C"}) == Eigen::Vector3d(0.5, 0.25, 0.25));
        CHECK_THROWS_AS(nabia::app::resolve_budgets(list, {"A", "B"}), std::invalid_argument);
        CHECK_THROWS_AS(nabia::app::resolve_budgets(c, {"A", "C"}), std::invalid_argument);       // no budget for C
        CHECK_THROWS_AS(nabia::app::resolve_budgets(c, {"A", "B", "C"}), std::invalid_argument);  // C missing
    }
    SECTION("invalid configs name the key") {
        CHECK_THAT(error_of([] { (void)nabia::app::parse_config(R"({"covarianse_window": 60})"); }),
                   ContainsSubstring("covarianse_window") && ContainsSubstring("unknown key"));
        CHECK_THAT(error_of([] { (void)nabia::app::parse_config(R"({"rebalance_frequency": 2.5})"); }),
                   ContainsSubstring("expected an integer"));
        CHECK_THAT(error_of([] { (void)nabia::app::parse_config(R"({"risk_measure": "cvar"})"); }),
                   ContainsSubstring("risk_measure"));
        CHECK_THAT(error_of([] { (void)nabia::app::parse_config(R"({"covariance_window": 1})"); }),
                   ContainsSubstring("covariance_window must be >= 2"));
        CHECK_THAT(error_of([] { (void)nabia::app::parse_config(R"({"start": "2024-01-02", "end": "2023-01-01"})"); }),
                   ContainsSubstring("before"));
        CHECK_THAT(error_of([] { (void)nabia::app::parse_config("{ not json"); }), ContainsSubstring("not valid JSON"));
        CHECK_THROWS_AS(nabia::app::parse_config("[1, 2]"), std::invalid_argument);
    }
}

TEST_CASE("app: writing the backtest CSV", "[app]") {
    Eigen::MatrixXd prices(70, 2);
    for (Eigen::Index t = 0; t < 70; ++t) {
        prices(t, 0) = 100.0 + static_cast<double>(t % 7) - 0.5 * static_cast<double>(t % 3);
        prices(t, 1) = 50.0 + 0.3 * static_cast<double>(t % 5);
    }
    const DataFrame<Date> px(daily_dates(sys_days{2024y / January / 1}, 70), {"A", "B"}, prices);
    auto params = std::make_unique<RiskBudgetingPortfolioParams>();
    params->covariance_window = 20;
    params->rebalance_frequency = 10;
    params->transaction_cost = 0.001;
    const PortfolioResult<Date> result = RiskBudgetingPortfolio(std::move(params)).run(px.pct_change());

    const TempFile out("result.csv");
    const TempFile metrics("metrics.csv");
    nabia::app::write_backtest_csv(out.path(), result);
    nabia::app::write_metrics_csv(metrics.path(), result.metrics);

    const std::vector<std::string> lines = read_lines(out.path());
    REQUIRE(lines.size() == 70);  // header + 69 return dates
    CHECK(lines[0] ==
          "date,weight_A,weight_B,rebalance,leverage,risk_contribution_A,risk_contribution_B,"
          "turnover,transaction_cost,gross_return,net_return,pnl,equity,drawdown");
    CHECK(lines[1].rfind("2024-01-02,0,0,0,,,,0,0,0,0,0,1,0", 0) == 0);  // nothing held, no trade
    // First rebalance at the close of return row 19 (2024-01-21): leverage and contributions filled.
    CHECK(lines[20].rfind("2024-01-21,0,0,1,1,", 0) == 0);
    CHECK(lines[21].rfind("2024-01-22,0,0,0,,,,", 0) != 0);  // weights now held

    // Values round-trip exactly (shortest representation).
    std::stringstream row(lines[69]);
    std::vector<std::string> fields;
    for (std::string f; std::getline(row, f, ',');) {
        fields.push_back(f);
    }
    REQUIRE(fields.size() == 14);
    CHECK(std::stod(fields[12]) == result.equity[68]);
    CHECK(std::stod(fields[1]) == result.weights.values()(68, 0));

    const std::vector<std::string> m = read_lines(metrics.path());
    CHECK(m[0] == "metric,value");
    CHECK(m.size() == 17);
    CHECK_THAT(std::stod(m[2].substr(m[2].find(',') + 1)), WithinAbs(result.metrics.total_return, 0.0));
}
