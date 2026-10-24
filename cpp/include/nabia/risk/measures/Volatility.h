//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_VOLATILITY_H
#define NABIA_VOLATILITY_H
#include "nabia/risk/IRiskMeasure.h"

/// Portfolio volatility  sigma(x) = sqrt(x' Sigma x).
/// Marginal contributions are Sigma x / sigma(x); the risk contributions
/// x_i (Sigma x)_i / sigma(x) sum to sigma(x).
class Volatility: public IRiskMeasure {
public:
    using IRiskMeasure::IRiskMeasure;

    [[nodiscard]] double compute_risk(const Eigen::VectorXd& x) const override;
    [[nodiscard]] Eigen::VectorXd compute_marginal_risk_contributions(const Eigen::VectorXd& x) const override;
};

#endif // NABIA_VOLATILITY_H
