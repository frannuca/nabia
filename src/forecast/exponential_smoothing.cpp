#include "nabia/forecast/exponential_smoothing.hpp"

#include <stdexcept>

namespace nabia::forecast {

SimpleExponentialSmoothing::SimpleExponentialSmoothing(double alpha) : alpha_(alpha) {
    if (!(alpha > 0.0 && alpha <= 1.0)) {
        throw std::invalid_argument("SimpleExponentialSmoothing: alpha must be in (0, 1]");
    }
}

void SimpleExponentialSmoothing::fit(std::span<const double> y) {
    if (y.empty()) {
        throw std::invalid_argument("SimpleExponentialSmoothing::fit: series is empty");
    }

    fitted_.clear();
    fitted_.reserve(y.size());
    sse_ = 0.0;

    // Initialise the level with the first observation.
    level_ = y[0];
    for (double obs : y) {
        fitted_.push_back(level_);
        const double err = obs - level_;
        sse_ += err * err;
        level_ += alpha_ * err; // equivalent to alpha*y + (1-alpha)*level
    }
    is_fitted_ = true;
}

std::vector<double> SimpleExponentialSmoothing::forecast(std::size_t horizon) const {
    if (!is_fitted_) {
        throw std::logic_error("SimpleExponentialSmoothing::forecast: model not fitted");
    }
    return std::vector<double>(horizon, level_);
}

} // namespace nabia::forecast
