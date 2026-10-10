#include "nabia/forecast/exponential_smoothing.hpp"

#include <stdexcept>
#include <utility>

namespace nabia::forecast {

SimpleExponentialSmoothing::SimpleExponentialSmoothing(std::unique_ptr<SimpleExponentialSmoothingParams> parameters)
    : IForecaster(std::move(parameters)) {
    const double alpha = parameters_->alpha;
    if (!(alpha > 0.0 && alpha <= 1.0)) {
        throw std::invalid_argument("SimpleExponentialSmoothing: alpha must be in (0, 1]");
    }
}

SimpleExponentialSmoothing::SimpleExponentialSmoothing(double alpha)
    : SimpleExponentialSmoothing(std::make_unique<SimpleExponentialSmoothingParams>(SimpleExponentialSmoothingParams{alpha})) {}

void SimpleExponentialSmoothing::fit(const Eigen::VectorXd& y) {
    if (y.size() == 0 || !y.allFinite()) {
        throw std::invalid_argument("SimpleExponentialSmoothing::fit: series is empty or not finite");
    }
    const double alpha = parameters_->alpha;
    in_sample_.resize(y.size());
    sse_ = 0.0;

    // Initialise the level with the first observation.
    level_ = y(0);
    for (Eigen::Index t = 0; t < y.size(); ++t) {
        in_sample_(t) = level_;
        const double err = y(t) - level_;
        sse_ += err * err;
        level_ += alpha * err; // equivalent to alpha*y + (1-alpha)*level
    }
    fitted_ = true;
}

Forecast SimpleExponentialSmoothing::forecast(Eigen::Index horizon) const {
    require_fitted("SimpleExponentialSmoothing::forecast");
    require_horizon(horizon, "SimpleExponentialSmoothing::forecast");
    const double alpha = parameters_->alpha;
    const double sigma2 = sse_ / static_cast<double>(in_sample_.size());
    Forecast out{Eigen::VectorXd::Constant(horizon, level_), Eigen::VectorXd(horizon)};
    for (Eigen::Index h = 0; h < horizon; ++h) {
        out.variance(h) = sigma2 * (1.0 + static_cast<double>(h) * alpha * alpha);
    }
    return out;
}

Eigen::VectorXd SimpleExponentialSmoothing::fitted_values() const {
    require_fitted("SimpleExponentialSmoothing::fitted_values");
    return in_sample_;
}

} // namespace nabia::forecast
