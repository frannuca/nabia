//
// Created by Francisco Nunez on 24.10.2026.
//

#include "nabia/regression/RegressionEngine.h"

#include "nabia/regression/ElasticNet.h"

#include "../common/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::placeholders::all;

namespace nabia::regression::detail {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Fold {
    std::vector<Index> train;
    std::vector<Index> validation;
};

/// Training and test data plus everything shared by the subset evaluations.
struct Problem {
    const RegressionEngineParams& p;
    MatrixXd X_train;
    VectorXd y_train;
    MatrixXd X_test;
    VectorXd y_test;
    std::vector<Fold> folds;
    bool tune = false;  // more than one (alpha, lambda) on the grid
};

ElasticNetParams solver_settings(const RegressionEngineParams& p, double alpha, double lambda = 0.0) {
    ElasticNetParams s;
    s.lambda = lambda;
    s.alpha = alpha;
    s.fit_intercept = p.fit_intercept;
    s.standardize = p.standardize;
    s.tolerance = p.tolerance;
    s.max_iter = p.max_iter;
    return s;
}

double mse(const VectorXd& y, const VectorXd& prediction) {
    return (y - prediction).squaredNorm() / static_cast<double>(y.size());
}

std::vector<Index> range(Index begin, Index end) {
    std::vector<Index> rows(static_cast<std::size_t>(std::max<Index>(0, end - begin)));
    std::iota(rows.begin(), rows.end(), begin);
    return rows;
}

/// Contiguous validation blocks over the n training rows (see ValidationScheme).
std::vector<Fold> make_folds(const RegressionEngineParams& p, Index n) {
    const Index K = p.validation_folds;
    const Index blocks = p.validation == ValidationScheme::BlockedKFold ? K : K + 1;
    const auto bound = [&](Index i) { return i * n / blocks; };
    std::vector<Fold> folds;
    const Index first = p.validation == ValidationScheme::BlockedKFold ? 0 : 1;
    for (Index k = first; k < blocks; ++k) {
        const Index begin = bound(k);
        const Index end = bound(k + 1);
        Fold f;
        f.validation = range(begin, end);
        f.train = range(0, begin - p.validation_gap);
        if (p.validation == ValidationScheme::BlockedKFold) {
            const std::vector<Index> after = range(end + p.validation_gap, n);
            f.train.insert(f.train.end(), after.begin(), after.end());
        }
        if (f.validation.empty() || f.train.size() < 2) {
            throw std::invalid_argument("RegressionEngine: " + std::to_string(n) +
                                        " training rows are too few for the validation folds and gap");
        }
        folds.push_back(std::move(f));
    }
    return folds;
}

std::vector<double> lambdas_for(const Problem& pb, const MatrixXd& X, double alpha) {
    if (!pb.p.lambdas.empty()) {
        std::vector<double> grid = pb.p.lambdas;
        std::sort(grid.begin(), grid.end(), std::greater<>());  // warm starts work best from large to small
        return grid;
    }
    return lambda_grid(lambda_max(X, pb.y_train, alpha, pb.p.fit_intercept, pb.p.standardize), pb.p.n_lambdas,
                       pb.p.lambda_min_ratio);
}

double in_sample_criterion(InSampleCriterion c, double train_mse, double df, Index n) {
    const auto dn = static_cast<double>(n);
    const double log_mse = std::log(std::max(train_mse, std::numeric_limits<double>::min()));
    switch (c) {
    case InSampleCriterion::BIC:
        return dn * log_mse + df * std::log(dn);
    case InSampleCriterion::AIC:
        return dn * log_mse + 2.0 * df;
    case InSampleCriterion::MSE:
        return train_mse;
    }
    throw std::invalid_argument("RegressionEngine: unknown in-sample criterion");
}

double r_squared(const VectorXd& y, const VectorXd& prediction, double reference_mean) {
    const double sst = (y.array() - reference_mean).square().sum();
    return sst > 0.0 ? 1.0 - (y - prediction).squaredNorm() / sst : 0.0;
}

/// Tunes (alpha, lambda) for one subset, fits it on the training rows and
/// scores it. Fills `curve` with the validation errors if given.
SubsetEvaluation evaluate(const Problem& pb, const std::vector<Index>& subset, ValidationCurve* curve) {
    const RegressionEngineParams& p = pb.p;
    const MatrixXd X = pb.X_train(all, subset);
    const auto A = static_cast<Index>(p.alphas.size());
    std::vector<std::vector<double>> grids;
    grids.reserve(p.alphas.size());
    for (double alpha : p.alphas) {
        grids.push_back(lambdas_for(pb, X, alpha));
    }
    const auto L = static_cast<Index>(grids.front().size());

    SubsetEvaluation e;
    e.factors = subset;
    e.validation_mse = kNaN;
    e.validation_se = kNaN;
    Index best_a = 0;
    Index best_l = 0;
    if (pb.tune) {
        MatrixXd sum = MatrixXd::Zero(A, L);
        MatrixXd sum_sq = MatrixXd::Zero(A, L);
        for (const Fold& fold : pb.folds) {
            const MatrixXd Xf = X(fold.train, all);
            const VectorXd yf = pb.y_train(fold.train);
            const MatrixXd Xv = X(fold.validation, all);
            const VectorXd yv = pb.y_train(fold.validation);
            for (Index a = 0; a < A; ++a) {
                const auto ai = static_cast<std::size_t>(a);
                const std::vector<ElasticNetFit> path =
                    elastic_net_path(Xf, yf, solver_settings(p, p.alphas[ai]), grids[ai]);
                for (Index l = 0; l < L; ++l) {
                    const ElasticNetFit& fit = path[static_cast<std::size_t>(l)];
                    const double m = mse(yv, fit.predict(Xv));
                    sum(a, l) += m;
                    sum_sq(a, l) += m * m;
                    e.converged = e.converged && fit.converged;
                }
            }
        }
        const auto K = static_cast<double>(pb.folds.size());
        const MatrixXd mean = sum / K;
        MatrixXd se = MatrixXd::Zero(A, L);
        if (pb.folds.size() > 1) {
            const MatrixXd var = ((sum_sq / K - mean.cwiseProduct(mean)) * (K / (K - 1.0))).cwiseMax(0.0);
            se = (var / K).cwiseSqrt();
        }
        for (Index a = 0; a < A; ++a) {
            for (Index l = 0; l < L; ++l) {
                if (mean(a, l) < mean(best_a, best_l)) {
                    best_a = a;
                    best_l = l;
                }
            }
        }
        if (p.one_standard_error_rule) {
            // Lambdas decrease along the row: the first one within the band is the largest.
            const double threshold = mean(best_a, best_l) + se(best_a, best_l);
            for (Index l = 0; l <= best_l; ++l) {
                if (mean(best_a, l) <= threshold) {
                    best_l = l;
                    break;
                }
            }
        }
        e.validation_mse = mean(best_a, best_l);
        e.validation_se = se(best_a, best_l);
        if (curve != nullptr) {
            curve->alphas = p.alphas;
            curve->lambdas.resize(A, L);
            for (Index a = 0; a < A; ++a) {
                curve->lambdas.row(a) = Eigen::Map<const Eigen::RowVectorXd>(grids[static_cast<std::size_t>(a)].data(), L);
            }
            curve->mse = mean;
            curve->se = se;
        }
    }

    // Fit on all training rows, warm-starting along the grid down to the chosen lambda.
    const auto ai = static_cast<std::size_t>(best_a);
    const std::vector<double> path_lambdas(grids[ai].begin(), grids[ai].begin() + best_l + 1);
    const ElasticNetFit fit = elastic_net_path(X, pb.y_train, solver_settings(p, p.alphas[ai]), path_lambdas).back();
    e.alpha = p.alphas[ai];
    e.lambda = fit.lambda;
    e.intercept = fit.intercept;
    e.coefficients = fit.coefficients;
    e.nonzero = fit.nonzero;
    e.degrees_of_freedom = fit.degrees_of_freedom;
    e.converged = e.converged && fit.converged;

    const double y_mean = pb.y_train.mean();
    const VectorXd train_prediction = fit.predict(X);
    e.train_mse = mse(pb.y_train, train_prediction);
    e.train_r2 = r_squared(pb.y_train, train_prediction, y_mean);
    e.criterion = in_sample_criterion(p.in_sample_criterion, e.train_mse, e.degrees_of_freedom, pb.y_train.size());
    if (pb.y_test.size() > 0) {
        const VectorXd test_prediction = fit.predict(pb.X_test(all, subset));
        e.test_mse = mse(pb.y_test, test_prediction);
        e.test_r2 = r_squared(pb.y_test, test_prediction, y_mean);
        e.score = e.test_mse;
    } else {
        e.test_mse = kNaN;
        e.test_r2 = kNaN;
        e.score = e.criterion;
    }
    return e;
}

/// Position of the lowest score (the first one on ties, which keeps results deterministic).
std::size_t argmin_score(const std::vector<SubsetEvaluation>& evals) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < evals.size(); ++i) {
        if (evals[i].score < evals[best].score) {
            best = i;
        }
    }
    return best;
}

struct SearchOutcome {
    std::vector<SubsetEvaluation> best_by_size;
    std::int64_t evaluated = 0;
    bool all_converged = true;
    int threads_used = 1;
};

std::vector<SubsetEvaluation> evaluate_all(const Problem& pb, const std::vector<std::vector<Index>>& subsets,
                                           SearchOutcome& out) {
    std::vector<SubsetEvaluation> evals(subsets.size());
    const auto count = static_cast<std::int64_t>(subsets.size());
    const int threads = nabia::detail::resolve_threads(pb.p.num_threads, count);
    nabia::detail::parallel_for(count, threads, [&](std::int64_t i) {
        const auto k = static_cast<std::size_t>(i);
        evals[k] = evaluate(pb, subsets[k], nullptr);
    });
    out.evaluated += count;
    out.threads_used = std::max(out.threads_used, threads);
    for (const SubsetEvaluation& e : evals) {
        out.all_converged = out.all_converged && e.converged;
    }
    return evals;
}

/// Number of subsets of sizes kmin..kmax of n factors, saturated just above `cap`.
std::int64_t count_subsets(Index n, Index kmin, Index kmax, std::int64_t cap) {
    double total = 0.0;
    for (Index k = kmin; k <= kmax; ++k) {
        double c = 1.0;
        for (Index i = 1; i <= k; ++i) {
            c = c * static_cast<double>(n - k + i) / static_cast<double>(i);
        }
        total += c;
        if (total > static_cast<double>(cap)) {
            return cap + 1;
        }
    }
    return static_cast<std::int64_t>(std::llround(total));
}

SearchOutcome exhaustive_search(const Problem& pb, Index n, Index kmin, Index kmax) {
    SearchOutcome out;
    for (Index k = kmin; k <= kmax; ++k) {
        std::vector<std::vector<Index>> subsets;
        std::vector<Index> c = range(0, k);
        while (true) {
            subsets.push_back(c);
            // Next combination in lexicographic order.
            Index i = k - 1;
            while (i >= 0 && c[static_cast<std::size_t>(i)] == n - k + i) {
                --i;
            }
            if (i < 0) {
                break;
            }
            ++c[static_cast<std::size_t>(i)];
            for (Index j = i + 1; j < k; ++j) {
                c[static_cast<std::size_t>(j)] = c[static_cast<std::size_t>(j - 1)] + 1;
            }
        }
        std::vector<SubsetEvaluation> evals = evaluate_all(pb, subsets, out);
        out.best_by_size.push_back(std::move(evals[argmin_score(evals)]));
    }
    return out;
}

SearchOutcome forward_search(const Problem& pb, Index n, Index kmin, Index kmax) {
    SearchOutcome out;
    std::vector<Index> selected;
    std::vector<Index> remaining = range(0, n);
    for (Index k = 1; k <= kmax; ++k) {
        std::vector<std::vector<Index>> candidates;
        candidates.reserve(remaining.size());
        for (Index factor : remaining) {
            std::vector<Index> s = selected;
            s.insert(std::upper_bound(s.begin(), s.end(), factor), factor);
            candidates.push_back(std::move(s));
        }
        std::vector<SubsetEvaluation> evals = evaluate_all(pb, candidates, out);
        const std::size_t best = argmin_score(evals);
        selected = candidates[best];
        remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(best));
        if (k >= kmin) {
            out.best_by_size.push_back(std::move(evals[best]));
        }
    }
    return out;
}

double percentile(std::vector<double> x, double q) {
    std::sort(x.begin(), x.end());
    const double pos = q * static_cast<double>(x.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, x.size() - 1);
    const double t = pos - static_cast<double>(lo);
    return (1.0 - t) * x[lo] + t * x[hi];
}

std::vector<double> to_vector(const VectorXd& v) { return {v.data(), v.data() + v.size()}; }

/// Moving block bootstrap of the training rows, refit at the selected (alpha, lambda).
BootstrapSummary bootstrap(const Problem& pb, const SubsetEvaluation& best, bool& all_converged, int& threads_used) {
    const RegressionEngineParams& p = pb.p;
    const Index B = p.bootstrap_samples;
    const Index n = pb.y_train.size();
    const Index block = std::min(p.bootstrap_block_size, n);
    const auto k = static_cast<Index>(best.factors.size());
    const MatrixXd X = pb.X_train(all, best.factors);
    const MatrixXd X_test = pb.X_test(all, best.factors);
    const ElasticNetParams settings = solver_settings(p, best.alpha, best.lambda);

    BootstrapSummary s;
    s.samples = B;
    s.coefficients.resize(B, k);
    VectorXd intercepts(B);
    VectorXd test_mse(B);
    std::vector<char> converged(static_cast<std::size_t>(B), 1);
    const int threads = nabia::detail::resolve_threads(p.num_threads, B);
    threads_used = std::max(threads_used, threads);
    nabia::detail::parallel_for(B, threads, [&](std::int64_t b) {
        // One generator per sample, seeded by (seed, b): independent of the thread count.
        std::seed_seq seq{static_cast<std::uint32_t>(p.bootstrap_seed), static_cast<std::uint32_t>(p.bootstrap_seed >> 32),
                          static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(static_cast<std::uint64_t>(b) >> 32)};
        std::mt19937_64 rng(seq);
        std::uniform_int_distribution<Index> start(0, n - block);
        std::vector<Index> rows;
        rows.reserve(static_cast<std::size_t>(n));
        while (static_cast<Index>(rows.size()) < n) {
            const Index s0 = start(rng);
            for (Index i = 0; i < block && static_cast<Index>(rows.size()) < n; ++i) {
                rows.push_back(s0 + i);
            }
        }
        const ElasticNetFit fit =
            elastic_net_path(X(rows, all), pb.y_train(rows), settings, {best.lambda}).front();
        s.coefficients.row(b) = fit.coefficients.transpose();
        intercepts(b) = fit.intercept;
        test_mse(b) = pb.y_test.size() > 0 ? mse(pb.y_test, fit.predict(X_test)) : kNaN;
        converged[static_cast<std::size_t>(b)] = fit.converged ? 1 : 0;
    });
    for (char c : converged) {
        all_converged = all_converged && c != 0;
    }

    const auto dB = static_cast<double>(B);
    const double q_lo = 0.5 * (1.0 - p.confidence_level);
    const double q_hi = 1.0 - q_lo;
    const auto std_error = [&](const VectorXd& x) {
        return B > 1 ? std::sqrt((x.array() - x.mean()).square().sum() / (dB - 1.0)) : 0.0;
    };
    s.mean.resize(k);
    s.std_error.resize(k);
    s.lower.resize(k);
    s.upper.resize(k);
    s.selection_frequency.resize(k);
    for (Index j = 0; j < k; ++j) {
        const VectorXd c = s.coefficients.col(j);
        s.mean(j) = c.mean();
        s.std_error(j) = std_error(c);
        s.lower(j) = percentile(to_vector(c), q_lo);
        s.upper(j) = percentile(to_vector(c), q_hi);
        s.selection_frequency(j) = static_cast<double>((c.array() != 0.0).count()) / dB;
    }
    s.intercept_mean = intercepts.mean();
    s.intercept_std_error = std_error(intercepts);
    if (pb.y_test.size() > 0) {
        s.test_mse_mean = test_mse.mean();
        s.test_mse_lower = percentile(to_vector(test_mse), q_lo);
        s.test_mse_upper = percentile(to_vector(test_mse), q_hi);
    } else {
        s.test_mse_mean = s.test_mse_lower = s.test_mse_upper = kNaN;
    }
    return s;
}

} // namespace

void validate(const RegressionEngineParams& p) {
    if (p.min_factors < 1) {
        throw std::invalid_argument("RegressionEngine: min_factors must be >= 1");
    }
    if (p.max_factors < 0 || (p.max_factors > 0 && p.max_factors < p.min_factors)) {
        throw std::invalid_argument("RegressionEngine: max_factors must be 0 (all) or >= min_factors");
    }
    if (p.max_exhaustive_subsets < 0) {
        throw std::invalid_argument("RegressionEngine: max_exhaustive_subsets must be >= 0");
    }
    if (!(p.train_fraction > 0.0 && p.train_fraction <= 1.0)) {
        throw std::invalid_argument("RegressionEngine: train_fraction must be in (0, 1]");
    }
    if (p.alphas.empty()) {
        throw std::invalid_argument("RegressionEngine: alphas must not be empty");
    }
    for (double a : p.alphas) {
        if (!(a >= 0.0 && a <= 1.0)) {
            throw std::invalid_argument("RegressionEngine: alphas must be in [0, 1]");
        }
    }
    for (double l : p.lambdas) {
        if (!(l >= 0.0) || !std::isfinite(l)) {
            throw std::invalid_argument("RegressionEngine: lambdas must be finite and >= 0");
        }
    }
    if (p.n_lambdas < 1) {
        throw std::invalid_argument("RegressionEngine: n_lambdas must be >= 1");
    }
    if (!(p.lambda_min_ratio > 0.0 && p.lambda_min_ratio <= 1.0)) {
        throw std::invalid_argument("RegressionEngine: lambda_min_ratio must be in (0, 1]");
    }
    const Index min_folds = p.validation == ValidationScheme::BlockedKFold ? 2 : 1;
    if (p.validation_folds < min_folds) {
        throw std::invalid_argument("RegressionEngine: validation_folds must be >= " + std::to_string(min_folds));
    }
    if (p.validation_gap < 0) {
        throw std::invalid_argument("RegressionEngine: validation_gap must be >= 0");
    }
    if (p.bootstrap_samples < 0 || p.bootstrap_block_size < 1) {
        throw std::invalid_argument("RegressionEngine: bootstrap_samples must be >= 0 and bootstrap_block_size >= 1");
    }
    if (!(p.confidence_level > 0.0 && p.confidence_level < 1.0)) {
        throw std::invalid_argument("RegressionEngine: confidence_level must be in (0, 1)");
    }
    if (p.num_threads < 0) {
        throw std::invalid_argument("RegressionEngine: num_threads must be >= 0");
    }
    nabia::regression::validate(solver_settings(p, 1.0));
}

RegressionComputation run_regression_engine(const RegressionEngineParams& p, const VectorXd& y, const MatrixXd& X,
                                            std::vector<std::string> names) {
    validate(p);
    const Index T = y.size();
    const Index N = X.cols();
    if (X.rows() != T) {
        throw std::invalid_argument("RegressionEngine: factors must have one row per target observation");
    }
    if (N == 0) {
        throw std::invalid_argument("RegressionEngine: no factors");
    }
    if (!y.allFinite() || !X.allFinite()) {
        throw std::invalid_argument("RegressionEngine: target and factors must be finite");
    }
    if (names.empty()) {
        for (Index j = 0; j < N; ++j) {
            names.push_back("x" + std::to_string(j));
        }
    } else if (static_cast<Index>(names.size()) != N) {
        throw std::invalid_argument("RegressionEngine: need one name per factor");
    }
    const Index kmin = p.min_factors;
    const Index kmax = p.max_factors == 0 ? N : p.max_factors;
    if (kmin > N || kmax > N) {
        throw std::invalid_argument("RegressionEngine: subset sizes exceed the " + std::to_string(N) + " factors");
    }
    // Small epsilon so that e.g. 0.7 * 10 counts as 7 rows despite rounding.
    const Index n_train = std::min(T, static_cast<Index>(std::floor(p.train_fraction * static_cast<double>(T) + 1e-9)));
    if (n_train < 2) {
        throw std::invalid_argument("RegressionEngine: fewer than 2 training rows");
    }

    Problem pb{p, X.topRows(n_train), y.head(n_train), X.bottomRows(T - n_train), y.tail(T - n_train), {}, false};
    const std::size_t grid_size = p.alphas.size() * (p.lambdas.empty() ? static_cast<std::size_t>(p.n_lambdas)
                                                                        : p.lambdas.size());
    pb.tune = grid_size > 1;
    if (pb.tune) {
        pb.folds = make_folds(p, n_train);
    }

    SubsetSearch search = p.search;
    if (search == SubsetSearch::Auto) {
        search = count_subsets(N, kmin, kmax, p.max_exhaustive_subsets) <= p.max_exhaustive_subsets
                     ? SubsetSearch::Exhaustive
                     : SubsetSearch::Forward;
    }
    SearchOutcome outcome =
        search == SubsetSearch::Exhaustive ? exhaustive_search(pb, N, kmin, kmax) : forward_search(pb, N, kmin, kmax);

    RegressionComputation r;
    r.search_used = search;
    r.subsets_evaluated = outcome.evaluated;
    r.threads_used = outcome.threads_used;
    r.all_converged = outcome.all_converged;
    r.train_rows = n_train;
    r.test_rows = T - n_train;
    r.best_by_size = std::move(outcome.best_by_size);
    const std::vector<Index> factors = r.best_by_size[argmin_score(r.best_by_size)].factors;
    // Evaluated again to record its validation curve (deterministic, so the result is the same).
    r.best = evaluate(pb, factors, pb.tune ? &r.validation_curve : nullptr);

    r.factor_positions = factors;
    for (Index j : factors) {
        r.factor_names.push_back(names[static_cast<std::size_t>(j)]);
    }
    r.alpha = r.best.alpha;
    r.lambda = r.best.lambda;
    const MatrixXd X_selected = X(all, factors);
    r.predictions = (X_selected * r.best.coefficients).array() + r.best.intercept;
    r.residuals = y - r.predictions;
    r.intercept = r.best.intercept;
    r.coefficients = r.best.coefficients;
    if (p.refit_on_full_sample && r.test_rows > 0) {
        const ElasticNetFit fit =
            elastic_net_path(X_selected, y, solver_settings(p, r.alpha), {r.lambda}).front();
        r.intercept = fit.intercept;
        r.coefficients = fit.coefficients;
        r.all_converged = r.all_converged && fit.converged;
    }
    if (p.bootstrap_samples > 0) {
        r.bootstrap = bootstrap(pb, r.best, r.all_converged, r.threads_used);
    }
    return r;
}

} // namespace nabia::regression::detail
