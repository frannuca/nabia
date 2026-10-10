//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/risk/IRiskMeasure.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

IRiskMeasure::IRiskMeasure(Eigen::MatrixXd covariance_matrix) : covariance_matrix_(std::move(covariance_matrix)) {
    const Eigen::MatrixXd& s = covariance_matrix_;
    if (s.rows() == 0 || s.rows() != s.cols()) {
        throw std::invalid_argument("IRiskMeasure: covariance matrix must be square and non-empty");
    }
    if (!s.allFinite()) {
        throw std::invalid_argument("IRiskMeasure: covariance matrix has non-finite entries");
    }
    const double scale = std::max(1.0, s.cwiseAbs().maxCoeff());
    if (((s - s.transpose()).cwiseAbs().array() > 1e-12 * scale).any()) {
        throw std::invalid_argument("IRiskMeasure: covariance matrix must be symmetric");
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(s, Eigen::EigenvaluesOnly);
    if (eig.eigenvalues().minCoeff() < -1e-10 * scale) {
        throw std::invalid_argument("IRiskMeasure: covariance matrix must be positive semi-definite");
    }
}

Eigen::VectorXd IRiskMeasure::compute_risk_contributions(const Eigen::VectorXd& x) const {
    return x.cwiseProduct(compute_marginal_risk_contributions(x));
}

void IRiskMeasure::check_dimension(const Eigen::VectorXd& v, const char* what) const {
    if (v.size() != dim()) {
        throw std::invalid_argument(std::string("IRiskMeasure: ") + what + " must have one entry per asset");
    }
}

double IRiskMeasure::portfolio_volatility(const Eigen::VectorXd& x) const {
    check_dimension(x, "portfolio");
    // max(0, .) guards against tiny negative values from round-off.
    return std::sqrt(std::max(0.0, x.dot(covariance_matrix_ * x)));
}

Eigen::VectorXd IRiskMeasure::portfolio_volatility_gradient(const Eigen::VectorXd& x) const {
    check_dimension(x, "portfolio");
    Eigen::VectorXd sigma_x = covariance_matrix_ * x;
    const double sigma = std::sqrt(std::max(0.0, x.dot(sigma_x)));
    if (sigma == 0.0) {
        return Eigen::VectorXd::Zero(dim());
    }
    return sigma_x / sigma;
}
