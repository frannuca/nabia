//
// Created by Francisco Nunez on 24.10.2026.
//

#ifndef NABIA_REGRESSIONENGINE_H
#define NABIA_REGRESSIONENGINE_H
#include "nabia/dataframe/DataFrame.h"
#include "nabia/dataframe/TimeSeries.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

/// How the subsets of factors are searched.
enum class SubsetSearch {
    Auto,        // Exhaustive if it needs at most max_exhaustive_subsets fits, else Forward
    Exhaustive,  // every subset with min_factors..max_factors factors
    Forward,     // greedy: add, one at a time, the factor that improves the score most
};

/// How the training rows are split to choose lambda and alpha. Folds are
/// contiguous blocks of time (rows are never shuffled).
enum class ValidationScheme {
    BlockedKFold,  // validate on each of K blocks, train on the others
    WalkForward,   // K + 1 blocks; fold k trains on blocks 0..k-1 and validates on block k
};

/// How subsets are compared when there is no test set (train_fraction = 1).
enum class InSampleCriterion {
    BIC,  // n log(MSE) + df log(n), with the elastic net's effective df
    AIC,  // n log(MSE) + 2 df
    MSE,  // in-sample mean squared error (favours the largest subsets)
};

struct RegressionEngineParams {
    // --- Factor selection ---
    Eigen::Index min_factors = 1;            // smallest subset size (>= 1)
    Eigen::Index max_factors = 0;            // largest subset size; 0 = all factors
    SubsetSearch search = SubsetSearch::Auto;
    std::int64_t max_exhaustive_subsets = 10000;  // Auto: limit on the subsets for Exhaustive
    // --- Sample split ---
    /// The first train_fraction of the rows (in time order) fit the models and
    /// choose lambda and alpha; the remaining rows score the subsets out of
    /// sample (extrapolation). 1.0 = in-sample: subsets are compared by
    /// in_sample_criterion instead.
    double train_fraction = 0.7;
    InSampleCriterion in_sample_criterion = InSampleCriterion::BIC;
    /// After selection, refit the chosen factors (same lambda, alpha) on all
    /// rows. The reported coefficients are then the refit; the predictions and
    /// scores are still those of the model fit on the training rows.
    bool refit_on_full_sample = false;
    // --- Hyperparameters ---
    std::vector<double> alphas = {0.0, 0.25, 0.5, 0.75, 1.0};  // in [0, 1]
    std::vector<double> lambdas;             // fixed grid (>= 0); empty = log grid from lambda_max per alpha
    Eigen::Index n_lambdas = 50;             // size of the automatic grid
    double lambda_min_ratio = 1e-3;          // smallest / largest lambda of the automatic grid
    // --- Validation (choice of lambda and alpha on the training rows) ---
    ValidationScheme validation = ValidationScheme::BlockedKFold;
    Eigen::Index validation_folds = 5;       // K (>= 2 for BlockedKFold, >= 1 for WalkForward)
    Eigen::Index validation_gap = 0;         // rows left out between training and validation blocks (autocorrelation)
    bool one_standard_error_rule = false;    // largest lambda within one standard error of the best (sparser models)
    // --- Bootstrap (of the selected model, on the training rows) ---
    Eigen::Index bootstrap_samples = 0;      // resamples refit with the selected lambda and alpha; 0 = off
    Eigen::Index bootstrap_block_size = 1;   // moving block bootstrap; 1 = iid rows
    double confidence_level = 0.95;          // percentile intervals
    std::uint64_t bootstrap_seed = 42;
    // --- Solver ---
    bool fit_intercept = true;
    bool standardize = true;
    double tolerance = 1e-7;
    int max_iter = 100000;
    /// Threads evaluating subsets (and bootstrap samples) in parallel, the
    /// calling thread included. 0 = hardware concurrency, 1 = serial. Results
    /// are identical for every thread count.
    int num_threads = 0;
};

/// One subset of factors with its tuned hyperparameters and scores.
struct SubsetEvaluation {
    std::vector<Eigen::Index> factors;   // column positions in the input, ascending
    double alpha = 0.0;
    double lambda = 0.0;
    double intercept = 0.0;              // fit on the training rows
    Eigen::VectorXd coefficients;        // one per entry of factors, fit on the training rows
    Eigen::Index nonzero = 0;
    double degrees_of_freedom = 0.0;
    double validation_mse = 0.0;         // cross-validated MSE at (alpha, lambda); NaN if nothing was tuned
    double validation_se = 0.0;          // its standard error across folds; NaN if nothing was tuned
    double train_mse = 0.0;
    double train_r2 = 0.0;
    double test_mse = 0.0;               // out of sample; NaN without test rows
    double test_r2 = 0.0;                // 1 - SSE / sum (y - mean(y_train))^2 on the test rows; NaN without test rows
    double criterion = 0.0;              // in_sample_criterion on the training rows
    double score = 0.0;                  // minimized: test_mse, or criterion without test rows
    bool converged = true;               // every fit behind this evaluation converged
};

/// Cross-validation error of the selected subset over the hyperparameter grid.
struct ValidationCurve {
    std::vector<double> alphas;          // A
    Eigen::MatrixXd lambdas;             // A x L, row a = lambda grid of alphas[a] (decreasing)
    Eigen::MatrixXd mse;                 // A x L, mean validation MSE over the folds
    Eigen::MatrixXd se;                  // A x L, its standard error
};

/// Bootstrap distribution of the selected model's coefficients.
struct BootstrapSummary {
    Eigen::Index samples = 0;
    Eigen::VectorXd mean;                 // per selected factor
    Eigen::VectorXd std_error;            // std of the bootstrap coefficients (ddof = 1)
    Eigen::VectorXd lower;                // percentile interval at confidence_level
    Eigen::VectorXd upper;
    Eigen::VectorXd selection_frequency;  // fraction of samples where the coefficient is non-zero
    double intercept_mean = 0.0;
    double intercept_std_error = 0.0;
    double test_mse_mean = 0.0;           // out-of-sample MSE of the bootstrap models; NaN without test rows
    double test_mse_lower = 0.0;
    double test_mse_upper = 0.0;
    Eigen::MatrixXd coefficients;         // samples x selected factors, the raw draws
};

/// Index-free output of the engine.
struct RegressionComputation {
    std::vector<std::string> factor_names;    // the selected factors
    std::vector<Eigen::Index> factor_positions;
    double alpha = 0.0;
    double lambda = 0.0;
    double intercept = 0.0;                   // the final model (refit on all rows if refit_on_full_sample)
    Eigen::VectorXd coefficients;
    SubsetEvaluation best;                    // the selected subset, fit on the training rows
    std::vector<SubsetEvaluation> best_by_size;  // best subset of each size min_factors..max_factors
    ValidationCurve validation_curve;         // of the selected subset; empty if nothing was tuned
    std::optional<BootstrapSummary> bootstrap;
    Eigen::VectorXd predictions;              // T, model fit on the training rows (test rows are out of sample)
    Eigen::VectorXd residuals;                // T, y - predictions
    Eigen::Index train_rows = 0;
    Eigen::Index test_rows = 0;
    SubsetSearch search_used = SubsetSearch::Exhaustive;
    std::int64_t subsets_evaluated = 0;
    bool all_converged = true;
    int threads_used = 1;
};

/// Result with the time index of the input.
template <TimeIndex Index>
struct RegressionResult {
    RegressionComputation model;
    TimeSeries<Index> predictions;   // in sample up to the end of the training rows, out of sample after
    TimeSeries<Index> residuals;
    std::optional<Index> test_start; // first out-of-sample time, if any
};

namespace nabia::regression::detail {
/// `X` is T x N (rows = time), `names` has N entries (empty = "x0", "x1", ...).
RegressionComputation run_regression_engine(const RegressionEngineParams& params, const Eigen::VectorXd& y,
                                            const Eigen::MatrixXd& X, std::vector<std::string> names);
void validate(const RegressionEngineParams& params);
} // namespace nabia::regression::detail

/// Elastic net regression of a target series on factors, with the choice of
/// factors, hyperparameters and their uncertainty.
///
/// 1. Split: the first train_fraction of the rows are the training set, the
///    rest the test set (time order kept, so the test set is an extrapolation).
/// 2. For every candidate subset of min_factors..max_factors factors (all of
///    them, or a forward stepwise path; see SubsetSearch):
///    a. choose (alpha, lambda) on the grid by time-series cross-validation on
///       the training rows (skipped if the grid has a single point),
///    b. fit the elastic net on all training rows at (alpha, lambda),
///    c. score it by the test MSE (out of sample) or, with train_fraction = 1,
///       by the in-sample criterion.
/// 3. Keep the subset with the lowest score. Optionally bootstrap the training
///    rows (moving blocks), refit at the chosen (alpha, lambda), and report
///    coefficient standard errors, percentile intervals, selection
///    frequencies and the spread of the test MSE.
///
/// Note that choosing the factors on the test set makes its MSE an optimistic
/// estimate of the error on new data; keep a later period aside for a final check.
class RegressionEngine {
public:
    /// Throws std::invalid_argument on null or invalid parameters.
    explicit RegressionEngine(std::unique_ptr<RegressionEngineParams> parameters)
        : parameters_(std::move(parameters)) {
        if (!parameters_) {
            throw std::invalid_argument("RegressionEngine: parameters are null");
        }
        nabia::regression::detail::validate(*parameters_);
    }

    /// Throws std::invalid_argument if the sizes differ, values are not finite,
    /// subset sizes exceed the factors, or there are too few rows for the folds.
    [[nodiscard]] RegressionComputation run(const Eigen::VectorXd& target, const Eigen::MatrixXd& factors,
                                            std::vector<std::string> names = {}) const {
        return nabia::regression::detail::run_regression_engine(*parameters_, target, factors, std::move(names));
    }

    /// The target and the factors must share the same index (align them first,
    /// e.g. with DataFrame::drop_missing). Throws std::invalid_argument otherwise.
    template <TimeIndex Index>
    [[nodiscard]] RegressionResult<Index> run(const TimeSeries<Index>& target, const DataFrame<Index>& factors) const {
        if (target.index() != factors.index()) {
            throw std::invalid_argument("RegressionEngine: target and factors must have the same index");
        }
        RegressionResult<Index> r;
        r.model = run(target.values(), factors.values(), factors.columns());
        r.predictions = TimeSeries<Index>(target.index(), r.model.predictions, "predictions");
        r.residuals = TimeSeries<Index>(target.index(), r.model.residuals, "residuals");
        if (r.model.test_rows > 0) {
            r.test_start = target.time(r.model.train_rows);
        }
        return r;
    }

    [[nodiscard]] const RegressionEngineParams& parameters() const { return *parameters_; }

private:
    std::unique_ptr<RegressionEngineParams> parameters_;
};

#endif // NABIA_REGRESSIONENGINE_H
