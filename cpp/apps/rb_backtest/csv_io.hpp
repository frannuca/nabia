// CSV input/output of the risk budgeting backtest application.
#pragma once

#include <nabia/dataframe/DataFrame.h>
#include <nabia/strategy/PerformanceMetrics.h>
#include <nabia/strategy/RiskBudgetingPortfolio.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace nabia::app {

/// Parses an ISO date "YYYY-MM-DD". Throws std::invalid_argument otherwise.
[[nodiscard]] Date parse_date(std::string_view text);

/// Formats a date as "YYYY-MM-DD".
[[nodiscard]] std::string format_date(Date date);

/// Reads asset prices from a CSV file:
///
///     date,SPX,BUND,GOLD
///     2024-01-02,4742.83,135.12,2063.4
///     ...
///
/// The first column holds ISO dates in increasing order, the others one price
/// series per asset (the header gives the names). Fields may be quoted;
/// Windows line endings and blank lines are accepted. A price is either a
/// positive number or missing: an empty field or NA, N/A, #N/A, NaN, null
/// (any case), read as NaN. Use DataFrame::drop_missing() to keep only the
/// dates where every asset has a price. Throws std::runtime_error with the file
/// name and line number on any other problem.
[[nodiscard]] DataFrame<Date> read_prices_csv(const std::filesystem::path& path);

/// Writes one row per period of the backtest:
///   date, weight_<asset>..., rebalance, leverage, risk_contribution_<asset>...,
///   turnover, transaction_cost, gross_return, net_return, pnl, equity, drawdown
/// leverage and risk contributions are filled on rebalance rows only.
/// Throws std::runtime_error if the file cannot be written.
void write_backtest_csv(const std::filesystem::path& path, const PortfolioResult<Date>& result);

/// Writes the performance metrics as "metric,value" rows.
void write_metrics_csv(const std::filesystem::path& path, const PerformanceMetrics& metrics);

} // namespace nabia::app
