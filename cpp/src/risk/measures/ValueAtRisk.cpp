//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/risk/measures/ValueAtRisk.h"

#include "nabia/stats/normal.hpp"

#include <stdexcept>
#include <utility>

ValueAtRisk::ValueAtRisk(Eigen::MatrixXd covariance_matrix, double confidence_level, Eigen::VectorXd expected_returns)
    : IRiskMeasure(std::move(covariance_matrix)),
      confidence_level_(confidence_level),
      z_(0.0),
      expected_returns_(std::move(expected_returns)) {
    if (!(confidence_level_ > 0.0 && confidence_level_ < 1.0)) {
        throw std::invalid_argument("ValueAtRisk: confidence level must be in (0, 1)");
    }
    z_ = nabia::stats::normal_quantile(confidence_level_);
    if (expected_returns_.size() == 0) {
        expected_returns_ = Eigen::VectorXd::Zero(dim());
    }
    check_dimension(expected_returns_, "expected returns");
}

double ValueAtRisk::compute_risk(const Eigen::VectorXd& x) const {
    return -expected_returns_.dot(x) + z_ * portfolio_volatility(x);
}

Eigen::VectorXd ValueAtRisk::compute_marginal_risk_contributions(const Eigen::VectorXd& x) const {
    return -expected_returns_ + z_ * portfolio_volatility_gradient(x);
}
