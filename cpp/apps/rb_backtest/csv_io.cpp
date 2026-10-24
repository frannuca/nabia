#include "csv_io.hpp"

#include <charconv>
#include <chrono>
#include <concepts>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace nabia::app {

namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
        s.remove_suffix(1);
    }
    return s;
}

/// Splits a CSV line on commas; double-quoted fields may contain commas and "" escapes.
std::vector<std::string> split_csv(std::string_view line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                field += c;
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            fields.emplace_back(trim(field));
            field.clear();
        } else {
            field += c;
        }
    }
    fields.emplace_back(trim(field));
    return fields;
}

/// Empty field or a common "not available" marker.
bool is_missing(std::string_view field) {
    std::string lower;
    for (const char c : field) {
        lower += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    return lower.empty() || lower == "na" || lower == "n/a" || lower == "#n/a" || lower == "nan" || lower == "null";
}

template <std::integral T>
bool parse_number(std::string_view text, T& out) {
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc() && end == text.data() + text.size();
}

// The floating-point overloads of std::from_chars / std::to_chars need macOS 26 / 13.3 in
// Apple's libc++, and this code is also compiled into the Python wheel, which targets older
// macOS. strtod and snprintf give the same results here (the C locale is never changed).
bool parse_number(std::string_view text, double& out) {
    if (text.empty() || text.front() == ' ' || text.front() == '+') {
        return false;
    }
    const std::string buffer(text);
    char* end = nullptr;
    out = std::strtod(buffer.c_str(), &end);
    return end == buffer.c_str() + buffer.size();
}

/// Shortest text that reads back as the same double: plain decimal notation
/// for ordinary magnitudes (1000000, 0.0025), scientific only for very large
/// or very small values. Negative zero is written as 0.
std::string format_number(double x) {
    if (x == 0.0) {
        return "0";
    }
    if (!std::isfinite(x)) {
        return std::isnan(x) ? "nan" : (x > 0 ? "inf" : "-inf");
    }
    // Fewest significant digits that round-trip.
    char buffer[400];
    int digits = 1;
    for (; digits < 17; ++digits) {
        std::snprintf(buffer, sizeof(buffer), "%.*e", digits - 1, x);
        if (std::strtod(buffer, nullptr) == x) {
            break;
        }
    }
    std::snprintf(buffer, sizeof(buffer), "%.*e", digits - 1, x);
    const std::string scientific(buffer);
    // Decimal exponent of the rounded value, read back from the scientific text.
    const int exponent = std::atoi(std::strchr(buffer, 'e') + 1);
    std::snprintf(buffer, sizeof(buffer), "%.*f", std::max(0, digits - 1 - exponent), x);
    const double magnitude = std::abs(x);
    // Outside this range std::chars_format::general (fixed only for exponents -4 to 5) is scientific.
    return magnitude >= 1e-5 && magnitude < 1e15 ? std::string(buffer) : scientific;
}

[[noreturn]] void fail(const std::filesystem::path& path, std::size_t line, const std::string& message) {
    throw std::runtime_error(path.string() + ":" + std::to_string(line) + ": " + message);
}

std::ofstream open_output(const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
    return out;
}

} // namespace

Date parse_date(std::string_view text) {
    text = trim(text);
    int y = 0;
    unsigned m = 0;
    unsigned d = 0;
    const bool shape = text.size() == 10 && text[4] == '-' && text[7] == '-';
    if (!shape || !parse_number(text.substr(0, 4), y) || !parse_number(text.substr(5, 2), m) ||
        !parse_number(text.substr(8, 2), d)) {
        throw std::invalid_argument("invalid date '" + std::string(text) + "' (expected YYYY-MM-DD)");
    }
    const std::chrono::year_month_day ymd{std::chrono::year{y}, std::chrono::month{m}, std::chrono::day{d}};
    if (!ymd.ok()) {
        throw std::invalid_argument("invalid date '" + std::string(text) + "'");
    }
    return std::chrono::sys_days{ymd};
}

std::string format_date(Date date) {
    const std::chrono::year_month_day ymd{date};
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02u-%02u", static_cast<int>(ymd.year()),
                  static_cast<unsigned>(ymd.month()), static_cast<unsigned>(ymd.day()));
    return buffer;
}

DataFrame<Date> read_prices_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open prices file " + path.string());
    }
    std::string line;
    std::size_t line_no = 0;
    std::vector<std::string> header;
    while (header.empty() && std::getline(in, line)) {
        ++line_no;
        if (!trim(line).empty()) {
            header = split_csv(line);
        }
    }
    if (header.size() < 2) {
        fail(path, line_no, "header must be 'date,<asset>,...' with at least one asset");
    }
    const std::vector<std::string> assets(header.begin() + 1, header.end());
    const auto n = static_cast<Eigen::Index>(assets.size());

    std::vector<Date> dates;
    std::vector<double> values;  // row-major
    while (std::getline(in, line)) {
        ++line_no;
        if (trim(line).empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv(line);
        if (fields.size() != header.size()) {
            fail(path, line_no, "expected " + std::to_string(header.size()) + " fields, found " +
                                    std::to_string(fields.size()));
        }
        try {
            dates.push_back(parse_date(fields[0]));
        } catch (const std::invalid_argument& e) {
            fail(path, line_no, e.what());
        }
        if (dates.size() > 1 && !(dates[dates.size() - 2] < dates.back())) {
            fail(path, line_no, "dates must be strictly increasing");
        }
        for (std::size_t j = 1; j < fields.size(); ++j) {
            double price = 0.0;
            if (is_missing(fields[j])) {
                price = std::numeric_limits<double>::quiet_NaN();
            } else if (!parse_number(std::string_view(fields[j]), price) || !(price > 0.0)) {
                fail(path, line_no,
                     "price of " + assets[j - 1] + " must be a positive number or missing, got '" + fields[j] + "'");
            }
            values.push_back(price);
        }
    }
    if (dates.empty()) {
        fail(path, line_no, "no price rows");
    }
    const auto rows = static_cast<Eigen::Index>(dates.size());
    Eigen::MatrixXd prices = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        values.data(), rows, n);
    return {std::move(dates), assets, std::move(prices)};
}

void write_backtest_csv(const std::filesystem::path& path, const PortfolioResult<Date>& result) {
    std::ofstream out = open_output(path);
    const std::vector<std::string>& assets = result.weights.columns();
    out << "date";
    for (const std::string& a : assets) {
        out << ",weight_" << a;
    }
    out << ",rebalance,leverage";
    for (const std::string& a : assets) {
        out << ",risk_contribution_" << a;
    }
    out << ",turnover,transaction_cost,gross_return,net_return,pnl,equity,drawdown\n";

    const Eigen::Index n = result.weights.cols();
    Eigen::Index next = 0;  // next rebalance row in target_weights / risk_contributions
    for (Eigen::Index t = 0; t < result.weights.rows(); ++t) {
        const Date date = result.weights.time(t);
        out << format_date(date);
        for (Eigen::Index j = 0; j < n; ++j) {
            out << ',' << format_number(result.weights.values()(t, j));
        }
        const bool rebalance = next < result.target_weights.rows() && result.target_weights.time(next) == date;
        out << ',' << (rebalance ? 1 : 0) << ',';
        if (rebalance) {
            out << format_number(result.leverage[next]);
        }
        for (Eigen::Index j = 0; j < n; ++j) {
            out << ',';
            if (rebalance) {
                out << format_number(result.risk_contributions.values()(next, j));
            }
        }
        out << ',' << format_number(result.turnover[t]) << ',' << format_number(result.costs[t]) << ','
            << format_number(result.gross_returns[t]) << ',' << format_number(result.net_returns[t]) << ','
            << format_number(result.pnl[t]) << ',' << format_number(result.equity[t]) << ','
            << format_number(result.drawdown[t]) << '\n';
        if (rebalance) {
            ++next;
        }
    }
    if (!out) {
        throw std::runtime_error("error while writing " + path.string());
    }
}

void write_metrics_csv(const std::filesystem::path& path, const PerformanceMetrics& m) {
    std::ofstream out = open_output(path);
    out << "metric,value\n"
        << "periods," << m.periods << '\n'
        << "total_return," << format_number(m.total_return) << '\n'
        << "annualized_return," << format_number(m.annualized_return) << '\n'
        << "annualized_volatility," << format_number(m.annualized_volatility) << '\n'
        << "sharpe_ratio," << format_number(m.sharpe_ratio) << '\n'
        << "sortino_ratio," << format_number(m.sortino_ratio) << '\n'
        << "max_drawdown," << format_number(m.max_drawdown) << '\n'
        << "max_drawdown_duration," << m.max_drawdown_duration << '\n'
        << "calmar_ratio," << format_number(m.calmar_ratio) << '\n'
        << "hit_rate," << format_number(m.hit_rate) << '\n'
        << "best_period," << format_number(m.best_period) << '\n'
        << "worst_period," << format_number(m.worst_period) << '\n'
        << "rebalances," << m.rebalances << '\n'
        << "average_turnover," << format_number(m.average_turnover) << '\n'
        << "annualized_turnover," << format_number(m.annualized_turnover) << '\n'
        << "total_costs," << format_number(m.total_costs) << '\n';
    if (!out) {
        throw std::runtime_error("error while writing " + path.string());
    }
}

} // namespace nabia::app
