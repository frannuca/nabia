// JSON configuration of the risk budgeting backtest application.
#pragma once

#include <nabia/dataframe/TimeIndex.h>
#include <nabia/strategy/RiskBudgetingPortfolio.h>

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nabia::app {

/// Everything the JSON file can set. Example (all keys optional):
///
///     {
///       "start": "2020-01-01", "end": "2024-12-31",
///       "covariance_window": 120, "covariance_estimator": "exponential", "ewma_halflife": 40,
///       "rebalance_frequency": 21,
///       "risk_measure": "volatility", "confidence_level": 0.99, "use_expected_returns": false,
///       "budgets": {"EQUITY": 0.4, "BONDS": 0.6},
///       "target_volatility": 0.10, "max_leverage": 2.0,
///       "transaction_cost": 0.0005, "drift_weights": true,
///       "initial_capital": 1000000, "periods_per_year": 252,
///       "solver_tolerance": 1e-8, "solver_max_iter": 5000, "num_threads": 0
///     }
///
/// risk_measure: "volatility" | "value_at_risk" | "expected_shortfall".
/// covariance_estimator: "sample" | "exponential".
/// budgets: an object keyed by asset name (every asset, positive values) or an
/// array in column order; omitted = equal risk contribution.
/// target_volatility: a number, or null / omitted for none.
struct BacktestConfig {
    RiskBudgetingPortfolioParams params;  // budgets left empty here; see named_budgets / budget_list
    std::optional<Date> start;            // first price date used (inclusive)
    std::optional<Date> end;              // last price date used (inclusive)
    std::map<std::string, double> named_budgets;
    std::vector<double> budget_list;
};

/// Parses the JSON text. Throws std::invalid_argument naming the offending key
/// on syntax errors, wrong types, invalid values or unknown keys.
[[nodiscard]] BacktestConfig parse_config(std::string_view json_text);

/// Reads and parses a JSON file. Throws std::runtime_error if it cannot be read.
[[nodiscard]] BacktestConfig load_config(const std::filesystem::path& path);

/// Budgets in the order of `assets` (empty vector = equal risk contribution).
/// Throws std::invalid_argument if named budgets miss an asset or name an
/// unknown one, or if a budget list has the wrong length.
[[nodiscard]] Eigen::VectorXd resolve_budgets(const BacktestConfig& config, const std::vector<std::string>& assets);

} // namespace nabia::app
