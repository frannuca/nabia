// nabia_rb_backtest: risk budgeting portfolio backtest from a prices CSV and a JSON config.
//
//   nabia_rb_backtest --prices prices.csv --config portfolio.json --output result.csv [--metrics metrics.csv]

#include "config.hpp"
#include "csv_io.hpp"

#include <nabia/strategy/RiskBudgetingPortfolio.h>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kUsage = R"(Usage:
  nabia_rb_backtest --prices <prices.csv> --config <portfolio.json> --output <result.csv> [--metrics <metrics.csv>]

Runs a walk-forward risk budgeting backtest.

  --prices   CSV with a date column (YYYY-MM-DD) and one price column per asset
  --config   JSON portfolio parameters (covariance_window, rebalance_frequency,
             budgets, transaction_cost, ...); see apps/rb_backtest/README.md
  --output   CSV written with one row per date: weights, risk contributions,
             turnover, costs, returns, PnL, equity and drawdown
  --metrics  optional CSV with the performance metrics
  --help     show this message
)";

struct Arguments {
    std::filesystem::path prices;
    std::filesystem::path config;
    std::filesystem::path output;
    std::optional<std::filesystem::path> metrics;
};

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    Arguments a;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            return std::nullopt;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value after " + std::string(flag));
        }
        const std::string value = argv[++i];
        if (flag == "--prices") {
            a.prices = value;
        } else if (flag == "--config") {
            a.config = value;
        } else if (flag == "--output") {
            a.output = value;
        } else if (flag == "--metrics") {
            a.metrics = value;
        } else {
            throw std::invalid_argument("unknown option " + std::string(flag));
        }
    }
    if (a.prices.empty() || a.config.empty() || a.output.empty()) {
        throw std::invalid_argument("--prices, --config and --output are required");
    }
    return a;
}

void print_summary(const PortfolioResult<Date>& r, const std::vector<std::string>& assets) {
    const PerformanceMetrics& m = r.metrics;
    const Eigen::Index first = r.weights.rows() - m.periods;  // first active row
    std::printf("Assets               %zu\n", assets.size());
    std::printf("Period               %s .. %s (%ld active periods)\n",
                nabia::app::format_date(r.weights.time(first)).c_str(),
                nabia::app::format_date(r.weights.time(r.weights.rows() - 1)).c_str(), static_cast<long>(m.periods));
    std::printf("Rebalances           %ld (%d threads)%s\n", static_cast<long>(m.rebalances), r.threads_used,
                r.all_converged ? "" : "  WARNING: some solves did not converge");
    std::printf("Total return         %10.2f %%\n", 100.0 * m.total_return);
    std::printf("Annualized return    %10.2f %%\n", 100.0 * m.annualized_return);
    std::printf("Annualized vol       %10.2f %%\n", 100.0 * m.annualized_volatility);
    std::printf("Sharpe / Sortino     %10.2f / %.2f\n", m.sharpe_ratio, m.sortino_ratio);
    std::printf("Max drawdown         %10.2f %% (%ld periods)\n", 100.0 * m.max_drawdown,
                static_cast<long>(m.max_drawdown_duration));
    std::printf("Annualized turnover  %10.2f\n", m.annualized_turnover);
    std::printf("Total costs          %10.4f %% of equity\n", 100.0 * m.total_costs);
    std::printf("Final equity         %14.2f\n", r.equity[r.equity.size() - 1]);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::optional<Arguments> args = parse_arguments(argc, argv);
        if (!args) {
            std::cout << kUsage;
            return 0;
        }
        nabia::app::BacktestConfig config = nabia::app::load_config(args->config);
        DataFrame<Date> prices = nabia::app::read_prices_csv(args->prices);
        // Keep only the dates where every asset has a price.
        const Eigen::Index read_dates = prices.rows();
        prices = prices.drop_missing();
        const Eigen::Index dropped_dates = read_dates - prices.rows();
        if (prices.rows() < 2) {
            throw std::invalid_argument("fewer than 2 dates have a price for every asset");
        }
        if (config.start || config.end) {
            const Date from = config.start.value_or(prices.time(0));
            const Date to = config.end.value_or(prices.time(prices.rows() - 1));
            prices = prices.between(from, to);
        }
        if (prices.rows() < 2) {
            throw std::invalid_argument("need at least 2 price dates in the selected period");
        }
        const DataFrame<Date> returns = prices.pct_change();

        auto params = std::make_unique<RiskBudgetingPortfolioParams>(config.params);
        params->budgets = nabia::app::resolve_budgets(config, returns.columns());
        const PortfolioResult<Date> result = RiskBudgetingPortfolio(std::move(params)).run(returns);

        nabia::app::write_backtest_csv(args->output, result);
        if (args->metrics) {
            nabia::app::write_metrics_csv(*args->metrics, result.metrics);
        }
        if (dropped_dates > 0) {
            std::printf("Dropped dates        %ld of %ld (missing prices for some assets)\n",
                        static_cast<long>(dropped_dates), static_cast<long>(read_dates));
        }
        print_summary(result, returns.columns());
        std::printf("Results written to   %s\n", args->output.string().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "nabia_rb_backtest: " << e.what() << "\n\n" << kUsage;
        return 1;
    }
}
