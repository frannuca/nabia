#include "config.hpp"

#include "csv_io.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nabia::app {

namespace {

using nlohmann::json;

[[noreturn]] void bad(const std::string& key, const std::string& message) {
    throw std::invalid_argument("config key '" + key + "': " + message);
}

double number(const std::string& key, const json& v) {
    if (!v.is_number()) {
        bad(key, "expected a number");
    }
    return v.get<double>();
}

long long integer(const std::string& key, const json& v) {
    if (!v.is_number_integer()) {
        bad(key, "expected an integer");
    }
    return v.get<long long>();
}

int small_integer(const std::string& key, const json& v) {
    const long long x = integer(key, v);
    if (x < -2147483647LL || x > 2147483647LL) {
        bad(key, "out of range");
    }
    return static_cast<int>(x);
}

bool boolean(const std::string& key, const json& v) {
    if (!v.is_boolean()) {
        bad(key, "expected true or false");
    }
    return v.get<bool>();
}

std::string text(const std::string& key, const json& v) {
    if (!v.is_string()) {
        bad(key, "expected a string");
    }
    return v.get<std::string>();
}

Date date(const std::string& key, const json& v) {
    try {
        return parse_date(text(key, v));
    } catch (const std::invalid_argument& e) {
        bad(key, e.what());
    }
}

} // namespace

BacktestConfig parse_config(std::string_view json_text) {
    json root;
    try {
        root = json::parse(json_text);
    } catch (const json::parse_error& e) {
        throw std::invalid_argument(std::string("config is not valid JSON: ") + e.what());
    }
    if (!root.is_object()) {
        throw std::invalid_argument("config must be a JSON object");
    }

    BacktestConfig c;
    RiskBudgetingPortfolioParams& p = c.params;
    for (const auto& [key, v] : root.items()) {
        if (key == "start") {
            c.start = date(key, v);
        } else if (key == "end") {
            c.end = date(key, v);
        } else if (key == "covariance_window") {
            p.covariance_window = integer(key, v);
        } else if (key == "covariance_estimator") {
            const std::string s = text(key, v);
            if (s == "sample") {
                p.covariance_estimator = CovarianceEstimator::Sample;
            } else if (s == "exponential") {
                p.covariance_estimator = CovarianceEstimator::Exponential;
            } else {
                bad(key, "expected \"sample\" or \"exponential\", got \"" + s + "\"");
            }
        } else if (key == "ewma_halflife") {
            p.ewma_halflife = number(key, v);
        } else if (key == "rebalance_frequency") {
            p.rebalance_frequency = integer(key, v);
        } else if (key == "risk_measure") {
            const std::string s = text(key, v);
            if (s == "volatility") {
                p.risk_measure = PortfolioRiskMeasure::Volatility;
            } else if (s == "value_at_risk") {
                p.risk_measure = PortfolioRiskMeasure::ValueAtRisk;
            } else if (s == "expected_shortfall") {
                p.risk_measure = PortfolioRiskMeasure::ExpectedShortfall;
            } else {
                bad(key, "expected \"volatility\", \"value_at_risk\" or \"expected_shortfall\", got \"" + s + "\"");
            }
        } else if (key == "confidence_level") {
            p.confidence_level = number(key, v);
        } else if (key == "use_expected_returns") {
            p.use_expected_returns = boolean(key, v);
        } else if (key == "budgets") {
            if (v.is_object()) {
                for (const auto& [asset, b] : v.items()) {
                    c.named_budgets[asset] = number(key + "." + asset, b);
                }
            } else if (v.is_array()) {
                for (std::size_t i = 0; i < v.size(); ++i) {
                    c.budget_list.push_back(number(key + "[" + std::to_string(i) + "]", v[i]));
                }
            } else {
                bad(key, "expected an object {asset: budget} or an array");
            }
        } else if (key == "target_volatility") {
            if (v.is_null()) {
                p.target_volatility.reset();
            } else {
                p.target_volatility = number(key, v);
            }
        } else if (key == "max_leverage") {
            p.max_leverage = number(key, v);
        } else if (key == "transaction_cost") {
            p.transaction_cost = number(key, v);
        } else if (key == "drift_weights") {
            p.drift_weights = boolean(key, v);
        } else if (key == "initial_capital") {
            p.initial_capital = number(key, v);
        } else if (key == "periods_per_year") {
            p.periods_per_year = small_integer(key, v);
        } else if (key == "solver_tolerance") {
            p.solver_tolerance = number(key, v);
        } else if (key == "solver_max_iter") {
            p.solver_max_iter = small_integer(key, v);
        } else if (key == "num_threads") {
            p.num_threads = small_integer(key, v);
        } else {
            bad(key, "unknown key");
        }
    }
    if (c.start && c.end && *c.end < *c.start) {
        throw std::invalid_argument("config: 'end' is before 'start'");
    }
    try {
        nabia::strategy::detail::validate(p);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("config: ") + e.what());
    }
    return c;
}

BacktestConfig load_config(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open config file " + path.string());
    }
    std::ostringstream text;
    text << in.rdbuf();
    try {
        return parse_config(text.str());
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(path.string() + ": " + e.what());
    }
}

Eigen::VectorXd resolve_budgets(const BacktestConfig& config, const std::vector<std::string>& assets) {
    const auto n = static_cast<Eigen::Index>(assets.size());
    if (!config.named_budgets.empty()) {
        Eigen::VectorXd b(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            const auto it = config.named_budgets.find(assets[static_cast<std::size_t>(i)]);
            if (it == config.named_budgets.end()) {
                throw std::invalid_argument("config budgets: no budget for asset '" + assets[static_cast<std::size_t>(i)] + "'");
            }
            b(i) = it->second;
        }
        for (const auto& [name, value] : config.named_budgets) {
            if (std::find(assets.begin(), assets.end(), name) == assets.end()) {
                throw std::invalid_argument("config budgets: asset '" + name + "' is not in the prices file");
            }
        }
        return b;
    }
    if (!config.budget_list.empty()) {
        if (config.budget_list.size() != assets.size()) {
            throw std::invalid_argument("config budgets: " + std::to_string(config.budget_list.size()) +
                                        " values for " + std::to_string(assets.size()) + " assets");
        }
        return Eigen::Map<const Eigen::VectorXd>(config.budget_list.data(), n);
    }
    return Eigen::VectorXd();
}

} // namespace nabia::app
