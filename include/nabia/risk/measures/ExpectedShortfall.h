//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_EXPECTEDSHORTFALL_H
#define NABIA_EXPECTEDSHORTFALL_H
#include "nabia/risk/IRiskMeasure.h"

/// Parametric (Gaussian) Expected Shortfall, also called CVaR: the expected loss
/// given that the loss exceeds VaR_alpha.
///
/// With asset returns r ~ N(mu, Sigma) and portfolio loss L = -r'x:
///     ES_alpha(x) = E[L | L >= VaR_alpha] = -mu'x + phi(z_alpha) / (1 - alpha) * sqrt(x' Sigma x),
/// where phi is the standard normal density and z_alpha = Phi^-1(alpha).
/// ES_alpha >= VaR_alpha always, and unlike VaR it is a coherent (subadditive)
/// risk measure.
class ExpectedShortfall: public IRiskMeasure {
public:
    /// `confidence_level` in (0, 1), e.g. 0.975. `expected_returns` empty means
    /// mu = 0; otherwise it needs one entry per asset.
    ExpectedShortfall(Eigen::MatrixXd covariance_matrix, double confidence_level,
                      Eigen::VectorXd expected_returns = Eigen::VectorXd());

    [[nodiscard]] double compute_risk(const Eigen::VectorXd& x) const override;
    /// -mu + phi(z_alpha) / (1 - alpha) * Sigma x / sigma(x)
    [[nodiscard]] Eigen::VectorXd compute_marginal_risk_contributions(const Eigen::VectorXd& x) const override;

    [[nodiscard]] double confidence_level() const { return confidence_level_; }

private:
    double confidence_level_;
    double k_;                         // phi(z_alpha) / (1 - alpha)
    Eigen::VectorXd expected_returns_; // mu, zero if not given
};

#endif // NABIA_EXPECTEDSHORTFALL_H
