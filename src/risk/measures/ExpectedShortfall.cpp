//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/risk/measures/ExpectedShortfall.h"

#include "nabia/stats/normal.hpp"

#include <stdexcept>
#include <utility>

ExpectedShortfall::ExpectedShortfall(Eigen::MatrixXd covariance_matrix, double confidence_level,
                                     Eigen::VectorXd expected_returns)
    : IRiskMeasure(std::move(covariance_matrix)),
      confidence_level_(confidence_level),
      k_(0.0),
      expected_returns_(std::move(expected_returns)) {
    if (!(confidence_level_ > 0.0 && confidence_level_ < 1.0)) {
        throw std::invalid_argument("ExpectedShortfall: confidence level must be in (0, 1)");
    }
    k_ = nabia::stats::normal_pdf(nabia::stats::normal_quantile(confidence_level_)) / (1.0 - confidence_level_);
    if (expected_returns_.size() == 0) {
        expected_returns_ = Eigen::VectorXd::Zero(dim());
    }
    check_dimension(expected_returns_, "expected returns");
}

double ExpectedShortfall::compute_risk(const Eigen::VectorXd& x) const {
    return -expected_returns_.dot(x) + k_ * portfolio_volatility(x);
}

Eigen::VectorXd ExpectedShortfall::compute_marginal_risk_contributions(const Eigen::VectorXd& x) const {
    return -expected_returns_ + k_ * portfolio_volatility_gradient(x);
}
