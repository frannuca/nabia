//
// Created by Francisco Nunez on 24.10.2026.
//

#ifndef NABIA_ELASTICNET_H
#define NABIA_ELASTICNET_H
#include <memory>
#include <vector>
#include <Eigen/Dense>

struct ElasticNetParams {
    double lambda = 1e-2;        // overall penalty strength (>= 0)
    double alpha = 1.0;          // mix in [0, 1]: 1 = lasso, 0 = ridge
    bool fit_intercept = true;   // unpenalized intercept (the data are centred)
    bool standardize = true;     // penalize coefficients of unit-variance factors (results are in original units)
    double tolerance = 1e-7;     // stop when max_j v_j (delta b_j)^2 < tolerance * var(y)
    int max_iter = 100000;       // coordinate descent sweeps over the factors
};

/// A fitted elastic net, in the original units of the factors.
struct ElasticNetFit {
    double lambda = 0.0;
    double alpha = 1.0;
    double intercept = 0.0;
    Eigen::VectorXd coefficients;     // one per factor (0 for constant factors)
    Eigen::Index nonzero = 0;         // factors with a non-zero coefficient
    double degrees_of_freedom = 0.0;  // effective df (Zou, Hastie & Tibshirani), intercept included
    int iterations = 0;               // coordinate descent sweeps
    bool converged = false;

    /// intercept + X * coefficients. Throws std::invalid_argument on a column count mismatch.
    [[nodiscard]] Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
};

namespace nabia::regression {

/// Smallest lambda at which every coefficient is zero, for the given alpha
/// (alpha is floored at 1e-3, as in glmnet, so that ridge gets a finite path).
[[nodiscard]] double lambda_max(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double alpha,
                                bool fit_intercept = true, bool standardize = true);

/// `count` lambdas log-spaced from `lambda_max` down to lambda_max * min_ratio
/// (decreasing). All zeros if lambda_max is 0. Throws std::invalid_argument if
/// count < 1 or min_ratio is not in (0, 1].
[[nodiscard]] std::vector<double> lambda_grid(double lambda_max, Eigen::Index count, double min_ratio);

/// Fits along a sequence of lambdas (best given in decreasing order), each fit
/// warm-started from the previous one. Uses params.alpha and the solver
/// settings; params.lambda is ignored. Throws std::invalid_argument on invalid
/// settings, mismatched or non-finite data, or a negative lambda.
[[nodiscard]] std::vector<ElasticNetFit> elastic_net_path(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                                          const ElasticNetParams& params,
                                                          const std::vector<double>& lambdas);

void validate(const ElasticNetParams& params);

} // namespace nabia::regression

/// Elastic net linear regression solved by cyclic coordinate descent.
///
/// Minimizes over (b0, b)
///     1/(2n) ||y - b0 - X b||^2 + lambda * (alpha ||b||_1 + (1 - alpha)/2 ||b||_2^2),
/// the glmnet parametrization. With standardize, the penalty applies to the
/// coefficients of the standardized factors (mean 0, population variance 1),
/// so the factors' units do not change which ones are selected; the
/// coefficients are reported in original units. Each sweep updates every
/// coefficient by soft thresholding; after a full sweep the solver iterates over
/// the non-zero coefficients only until they settle, then checks with a new
/// full sweep. Constant factors get a zero coefficient.
class ElasticNet {
public:
    /// Throws std::invalid_argument on null or invalid parameters.
    explicit ElasticNet(std::unique_ptr<ElasticNetParams> parameters);

    /// X is n x p (rows = observations). Throws std::invalid_argument if the
    /// sizes differ, there are no rows, or values are not finite.
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    /// Throws std::logic_error before fit().
    [[nodiscard]] Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
    /// Throws std::logic_error before fit().
    [[nodiscard]] const ElasticNetFit& result() const;
    [[nodiscard]] const ElasticNetParams& parameters() const { return *parameters_; }

private:
    std::unique_ptr<ElasticNetParams> parameters_;
    ElasticNetFit fit_;
    bool fitted_ = false;
};

#endif // NABIA_ELASTICNET_H
