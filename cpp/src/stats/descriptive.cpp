#include "nabia/stats/descriptive.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nabia::stats {

double mean(std::span<const double> x) {
    if (x.empty()) {
        throw std::invalid_argument("mean: input is empty");
    }
    return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

double variance(std::span<const double> x, std::size_t ddof) {
    if (x.size() <= ddof) {
        throw std::invalid_argument("variance: not enough data points for given ddof");
    }
    const double m = mean(x);
    double sum_sq = 0.0;
    for (double v : x) {
        const double d = v - m;
        sum_sq += d * d;
    }
    return sum_sq / static_cast<double>(x.size() - ddof);
}

double stddev(std::span<const double> x, std::size_t ddof) {
    return std::sqrt(variance(x, ddof));
}

double autocorrelation(std::span<const double> x, std::size_t lag) {
    if (lag >= x.size()) {
        throw std::invalid_argument("autocorrelation: lag must be smaller than series length");
    }
    const double m = mean(x);

    double denom = 0.0;
    for (double v : x) {
        denom += (v - m) * (v - m);
    }
    if (denom == 0.0) {
        throw std::invalid_argument("autocorrelation: series is constant");
    }

    double numer = 0.0;
    for (std::size_t t = lag; t < x.size(); ++t) {
        numer += (x[t] - m) * (x[t - lag] - m);
    }
    return numer / denom;
}

} // namespace nabia::stats
