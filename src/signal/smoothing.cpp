#include "nabia/signal/smoothing.hpp"

#include <stdexcept>

namespace nabia::signal {

std::vector<double> moving_average(std::span<const double> x, std::size_t window) {
    if (window == 0 || window > x.size()) {
        throw std::invalid_argument("moving_average: window must be in [1, x.size()]");
    }

    std::vector<double> out;
    out.reserve(x.size() - window + 1);

    // Rolling sum: O(n) instead of O(n * window).
    double sum = 0.0;
    for (std::size_t i = 0; i < window; ++i) {
        sum += x[i];
    }
    const auto w = static_cast<double>(window);
    out.push_back(sum / w);

    for (std::size_t i = window; i < x.size(); ++i) {
        sum += x[i] - x[i - window];
        out.push_back(sum / w);
    }
    return out;
}

std::vector<double> exponential_moving_average(std::span<const double> x, double alpha) {
    if (!(alpha > 0.0 && alpha <= 1.0)) {
        throw std::invalid_argument("exponential_moving_average: alpha must be in (0, 1]");
    }

    std::vector<double> out;
    out.reserve(x.size());
    for (double v : x) {
        out.push_back(out.empty() ? v : alpha * v + (1.0 - alpha) * out.back());
    }
    return out;
}

std::vector<double> diff(std::span<const double> x, std::size_t order) {
    std::vector<double> current(x.begin(), x.end());
    for (std::size_t k = 0; k < order && !current.empty(); ++k) {
        for (std::size_t i = 0; i + 1 < current.size(); ++i) {
            current[i] = current[i + 1] - current[i];
        }
        current.pop_back();
    }
    return current;
}

} // namespace nabia::signal
