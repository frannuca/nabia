"""Extract source excerpts for doc/nabia.tex, the single nabia document.

Run by doc/build.sh before pdflatex, so the
code shown in the PDF is always the current code:

    python3 extract_snippets.py <repository root> <output directory>

For every entry of SNIPPETS it writes <name>.txt (the excerpt) and adds a line to
snippets.tex defining \\snippetsource{<name>} as "<file>:<first>-<last>"; REFS
entries only define the location (\\coderef{<name>} in the document). 
Selectors:
  ("py", "name")               Python function / class / constant, found with ast;
                               dotted names reach into classes and functions
                               ("AgentRuntime._loop", "build_server.news_summary").
  ("cpp", "signature text")    from the line containing the text to the brace
                               that closes the first "{" after it.
  ("between", start, end)      from the line containing `start` to the line
                               containing `end` (inclusive).
  ("text", "NAME")             the value of a module-level string constant (a
                               prompt), as plain text.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

MAX_LINES = 70
#: Snippets shown in full however long (prompts and key algorithms).
FULL = {"rb_simulate", "rb_run", "news_system", "stats_system", "execution_system", "backtest_note", "as_of_rule", "team_calls",
        "engine_phase2", "runtime_loop"}

SNIPPETS: list[tuple[str, str, tuple]] = [
    # --- C++ library and bindings ---
    ('df_drop_missing', 'cpp/include/nabia/dataframe/DataFrame.h', ('cpp', '[[nodiscard]] DataFrame drop_missing() const {')),
    ('ewma_impl', 'cpp/src/signal/EWMA.cpp', ('cpp', 'struct EWMA::Impl {')),
    ('lbfgs_line_search', 'cpp/src/optimization/convex/LBFGS.cpp', ('between', 'double t = 1.0;', 'if (!accepted) {')),
    ('parallel_evaluator', 'cpp/src/optimization/nongradopt/parallel_evaluator.hpp', ('cpp', '[[nodiscard]] std::vector<TResult> evaluate(const std::vector<T>& inputs) const {')),
    ('vol_gradient', 'cpp/src/risk/IRiskMeasure.cpp', ('cpp', 'Eigen::VectorXd IRiskMeasure::portfolio_volatility_gradient')),
    ('var_measure', 'cpp/src/risk/measures/ValueAtRisk.cpp', ('between', 'double ValueAtRisk::compute_risk', 'return -expected_returns_ + z_')),
    ('rb_objective', 'cpp/src/risk/RiskBudgeting.cpp', ('between', "// f(y) = R(y) - b' log(y)", 'return measure.compute_risk(y) - b.dot')),
    ('cov_estimate', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('cpp', 'WindowMoments estimate(')),
    ('solve_target', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('cpp', 'Target solve_target(')),
    ('engine_phase1', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('between', '// Phase 1 (parallel)', 'c.threads_used = threads;')),
    ('engine_phase2', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('between', '// Phase 2 (sequential)', 'equity = new_equity;')),
    ('parallel_for', 'cpp/src/common/parallel_for.hpp', ('cpp', 'void parallel_for(')),
    ('performance', 'cpp/src/strategy/PerformanceMetrics.cpp', ('cpp', 'PerformanceMetrics compute_performance(')),
    ('binding_prepare', 'cpp/bindings/src/rb_backtest_module.cpp', ('cpp', 'Prepared prepare(')),
    ('lbfgs_two_loop', 'cpp/src/optimization/convex/LBFGS.cpp', ('between', '// Two-loop recursion', 'd = -d;')),
    ('al_function', 'cpp/src/optimization/convex/AugmentedLagrangian.cpp', ('between', '// L_rho(x) = f', '        return value;')),
    ('al_updates', 'cpp/src/optimization/convex/AugmentedLagrangian.cpp', ('between', '// Violation measured before', 'inner_tol = std::max(p.tolerance, 0.1 * inner_tol);')),
    ('sqp_bfgs', 'cpp/src/optimization/convex/SQP.cpp', ('cpp', 'void damped_bfgs_update(')),
    ('sqp_merit', 'cpp/src/optimization/convex/SQP.cpp', ('between', '// l1 merit function phi', '            t *= options_.backtrack;')),
    ('nm_step', 'cpp/src/optimization/nongradopt/NelderMead.cpp', ('between', 'const VectorXd centroid', '            values(j) = f(simplex.col(j));')),
    ('de_generation', 'cpp/src/optimization/nongradopt/DifferentialEvolution.cpp', ('between', 'const double F = p.mutation_min', 'trials(i, j) = v;')),
    ('rb_solve', 'cpp/src/risk/RiskBudgeting.cpp', ('between', '// L-BFGS stops on ||grad||_inf', 'inner_tol *= 0.5 * params.tolerance / err;')),
    ('normal_quantile', 'cpp/src/stats/normal.cpp', ('between', '    // ... refined with one Halley step', 'return x - u / (1.0 + 0.5 * x * u);')),
    ('kalman_step', 'cpp/src/forecast/KalmanFilter.cpp', ('between', '        if (t > 0) {  // predict', '        filtered_covariances_[ts] = P;')),
    ('kalman_smoother', 'cpp/src/forecast/KalmanFilter.cpp', ('cpp', 'Eigen::MatrixXd KalmanFilter::smoothed_states() const {')),
    ('gp_nll', 'cpp/src/forecast/GaussianProcess.cpp', ('between', '// d(-log p)/d theta_j', '            return -f->log_ml;')),
    ('gp_predict', 'cpp/src/forecast/GaussianProcess.cpp', ('cpp', 'Forecast GaussianProcess::predict(')),
    ('ses_fit', 'cpp/src/forecast/exponential_smoothing.cpp', ('between', '    // Initialise the level', '        out.variance(h) = sigma2')),
    ('schedule_rows', 'cpp/include/nabia/strategy/RiskBudgetingPortfolio.h', ('between', '        for (Eigen::Index s = 0; s < budget_schedule.rows(); ++s) {', '            schedule.push_back(')),
    ('budgets_at', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('between', '    // Budgets in force at a rebalance row', '    };')),
    ('enet_sweep', 'cpp/src/regression/ElasticNet.cpp', ('cpp', 'double sweep(bool active_only) {')),
    ('enet_lambda_max', 'cpp/src/regression/ElasticNet.cpp', ('cpp', 'double lambda_max(const MatrixXd& X')),
    ('regression_folds', 'cpp/src/regression/RegressionEngine.cpp', ('cpp', 'std::vector<Fold> make_folds(')),
    ('regression_cv', 'cpp/src/regression/RegressionEngine.cpp', ('between', '    if (pb.tune) {', '        e.validation_se = se(best_a, best_l);')),
    ('regression_bootstrap_draw', 'cpp/src/regression/RegressionEngine.cpp', ('between', '// One generator per sample', 'elastic_net_path(X(rows, all)')),
    ('bridge_kwargs', 'cpp/bindings/src/bridge.hpp', ('cpp', 'std::unique_ptr<T> from_kwargs(const py::kwargs& kwargs) {')),
    ('bridge_pycallable', 'cpp/bindings/src/bridge.hpp', ('cpp', 'class PyCallable {')),
    ('frames_regress', 'cpp/bindings/nabia_python_bridge/frames.py', ('py', 'regress')),
    ('py_backtest_inputs', 'nabia_python/nabia/portfolio.py', ('py', '_inputs')),
    ('py_to_days', 'nabia_python/nabia/portfolio.py', ('py', '_to_days')),
    ('crypto_fetch', 'nabia_python/nabia/marketdata/sources.py', ('py', 'CryptoSource.fetch')),
    ('upsert_rows', 'nabia_python/nabia/marketdata/store.py', ('py', 'upsert_rows')),
    ('update_market_data', 'nabia_python/nabia/marketdata/download.py', ('py', 'update_market_data')),
    ('rolling_volatility', 'nabia_python/nabia/analytics.py', ('py', 'rolling_volatility')),
    ('trend_signals', 'nabia_python/nabia/analytics.py', ('py', 'trend_signals')),
    ('market_regime', 'nabia_python/nabia/analytics.py', ('py', 'market_regime')),
    # --- sentiment risk-budgeting pipeline ---
    ('rb_transform', 'nabia_python/nabia/sentiment_rb/features/transforms.py', ('py', 'transform')),
    ('rb_asset_matrix', 'nabia_python/nabia/sentiment_rb/features/transforms.py', ('py', 'asset_matrix')),
    ('rb_target', 'nabia_python/nabia/sentiment_rb/features/transforms.py', ('py', 'vol_scaled_target')),
    ('rb_news_truth', 'nabia_python/nabia/sentiment_rb/data/synthetic.py', ('py', 'news_truth')),
    ('rb_llm_call', 'nabia_python/nabia/sentiment_rb/llm/client.py', ('py', 'LLM.__call__')),
    ('rb_llm_anthropic', 'nabia_python/nabia/sentiment_rb/llm/client.py', ('py', 'LLM._anthropic')),
    ('rb_dedup', 'nabia_python/nabia/sentiment_rb/llm/news.py', ('between', 'recent = [j for j in recent', 'recent.append(i)')),
    ('rb_weights', 'nabia_python/nabia/sentiment_rb/llm/news.py', ('between', 'rel = {m.node_id: m.relevance', 'records.append((a.day')),
    ('rb_ema0', 'nabia_python/nabia/sentiment_rb/llm/news.py', ('py', 'ema0')),
    ('rb_node_features', 'nabia_python/nabia/sentiment_rb/llm/news.py', ('py', 'node_features')),
    ('rb_fit_asset', 'nabia_python/nabia/sentiment_rb/model/elastic_net.py', ('py', 'fit_asset')),
    ('rb_fit_at', 'nabia_python/nabia/sentiment_rb/model/elastic_net.py', ('py', 'walk_forward_scores.fit_at')),
    ('rb_normalize', 'nabia_python/nabia/sentiment_rb/model/elastic_net.py', ('between', 'for t in range(p + 1, stop + 1):', 'float(stats.normal_cdf')),
    ('rb_blend', 'nabia_python/nabia/sentiment_rb/model/elastic_net.py', ('between', 'live = S.dropna(how="all").index', 'assert ((smooth >= 0)')),
    ('rb_modulate', 'nabia_python/nabia/sentiment_rb/portfolio/budgets.py', ('py', 'modulate')),
    ('rb_directions', 'nabia_python/nabia/sentiment_rb/portfolio/budgets.py', ('py', 'directions')),
    ('rb_window_cov', 'nabia_python/nabia/sentiment_rb/portfolio/backtest.py', ('py', 'window_cov')),
    ('rb_solve', 'nabia_python/nabia/sentiment_rb/portfolio/backtest.py', ('py', 'solve')),
    ('rb_simulate', 'nabia_python/nabia/sentiment_rb/portfolio/backtest.py', ('py', 'simulate')),
    ('rb_backtests', 'nabia_python/nabia/sentiment_rb/portfolio/backtest.py', ('py', 'backtests')),
    ('rb_run', 'nabia_python/nabia/sentiment_rb/pipeline.py', ('py', 'run')),
]

#: Location-only references (\\coderef{name}): where a concept lives, without an excerpt.
REFS: list[tuple[str, str, tuple]] = [
    ('time_index', 'cpp/include/nabia/dataframe/TimeIndex.h', ('between', 'concept TimeIndex', 'concept TimeIndex')),
    ('dataframe', 'cpp/include/nabia/dataframe/DataFrame.h', ('cpp', 'class DataFrame {')),
    ('timeseries', 'cpp/include/nabia/dataframe/TimeSeries.h', ('cpp', 'class TimeSeries {')),
    ('pct_change', 'cpp/include/nabia/dataframe/DataFrame.h', ('cpp', '[[nodiscard]] DataFrame pct_change() const {')),
    ('log_returns', 'cpp/include/nabia/dataframe/DataFrame.h', ('cpp', '[[nodiscard]] DataFrame log_returns() const {')),
    ('ewma_class', 'cpp/include/nabia/signal/EWMA.h', ('cpp', 'class EWMA: public IFilter<EWMAParams> {')),
    ('ewma_alphas', 'cpp/src/signal/EWMA.cpp', ('between', 'double EWMA::alpha_from_span', 'double EWMA::alpha_from_halflife')),
    ('butterworth_design', 'cpp/src/signal/ButterworthBandPass.cpp', ('cpp', 'void design(Filter& filter')),
    ('butterworth_params', 'cpp/include/nabia/signal/ButterworthBandPass.h', ('cpp', 'struct ButterworthBandPassParams {')),
    ('moving_average', 'cpp/src/signal/smoothing.cpp', ('cpp', 'std::vector<double> moving_average(')),
    ('kalman_params', 'cpp/include/nabia/forecast/KalmanFilter.h', ('cpp', 'struct KalmanFilterParams {')),
    ('kalman_models', 'cpp/src/forecast/KalmanFilter.cpp', ('between', 'KalmanFilterParams KalmanFilterParams::local_level', '    p.initial_covariance = initial_variance * MatrixXd::Identity(2, 2);')),
    ('kalman_forecast', 'cpp/src/forecast/KalmanFilter.cpp', ('cpp', 'Forecast KalmanFilter::forecast(')),
    ('gp_fit', 'cpp/src/forecast/GaussianProcess.cpp', ('cpp', 'void GaussianProcess::fit(')),
    ('gp_factorize', 'cpp/src/forecast/GaussianProcess.cpp', ('cpp', 'std::optional<Factorization> factorize(')),
    ('kernels', 'cpp/src/forecast/Kernels.cpp', ('between', '// --- Squared exponential', '    return std::make_unique<ProductKernel>(k1_->clone(), k2_->clone());')),
    ('ioptimizer', 'cpp/include/nabia/optimization/IOptimizer.h', ('cpp', 'class IOptimizer')),
    ('lbfgs_params', 'cpp/include/nabia/optimization/convex/LBFGS.h', ('cpp', 'struct LBFGSParams')),
    ('lbfgs_optimize', 'cpp/src/optimization/convex/LBFGS.cpp', ('cpp', 'Eigen::VectorXd LBFGS::optimize(')),
    ('constraints', 'cpp/include/nabia/optimization/convex/Constraints.h', ('cpp', 'struct LinearConstraint')),
    ('al_options', 'cpp/include/nabia/optimization/convex/AugmentedLagrangian.h', ('cpp', 'struct AugmentedLagrangianOptions')),
    ('al_optimize', 'cpp/src/optimization/convex/AugmentedLagrangian.cpp', ('cpp', 'Eigen::VectorXd AugmentedLagrangian::optimize(')),
    ('sqp_options', 'cpp/include/nabia/optimization/convex/SQP.h', ('cpp', 'struct SQPOptions')),
    ('sqp_qp', 'cpp/src/optimization/convex/SQP.cpp', ('cpp', 'QPSolution solve_qp(')),
    ('sqp_optimize', 'cpp/src/optimization/convex/SQP.cpp', ('cpp', 'Eigen::VectorXd SQP::optimize(')),
    ('nm_coefficients', 'cpp/src/optimization/nongradopt/NelderMead.cpp', ('between', 'const double alpha = p.adaptive', 'const double sigma = p.adaptive')),
    ('nm_optimize', 'cpp/src/optimization/nongradopt/NelderMead.cpp', ('cpp', 'Eigen::VectorXd NelderMead::optimize(')),
    ('box', 'cpp/src/optimization/nongradopt/box.hpp', ('cpp', 'struct Box {')),
    ('de_params', 'cpp/include/nabia/optimization/nongradopt/DifferentialEvolution.h', ('cpp', 'struct DifferentialEvolutionParams')),
    ('de_optimize', 'cpp/src/optimization/nongradopt/DifferentialEvolution.cpp', ('cpp', 'Eigen::VectorXd DifferentialEvolution::optimize(')),
    ('irisk', 'cpp/include/nabia/risk/IRiskMeasure.h', ('cpp', 'class IRiskMeasure')),
    ('risk_contributions', 'cpp/src/risk/IRiskMeasure.cpp', ('cpp', 'Eigen::VectorXd IRiskMeasure::compute_risk_contributions')),
    ('es_measure', 'cpp/src/risk/measures/ExpectedShortfall.cpp', ('between', 'ExpectedShortfall::ExpectedShortfall(', 'return -expected_returns_ + k_')),
    ('volatility_measure', 'cpp/src/risk/measures/Volatility.cpp', ('between', 'double Volatility::compute_risk', 'return portfolio_volatility_gradient(x);')),
    ('rb_start', 'cpp/src/risk/RiskBudgeting.cpp', ('cpp', 'VectorXd default_start(')),
    ('rb_budgets', 'cpp/src/risk/RiskBudgeting.cpp', ('cpp', 'VectorXd normalized_budgets(')),
    ('rb_solver', 'cpp/src/risk/RiskBudgeting.cpp', ('cpp', 'RiskBudgetingResult solve_risk_budgeting(')),
    ('rb_template', 'cpp/include/nabia/risk/RiskBudgeting.h', ('cpp', 'class RiskBudgeting')),
    ('portfolio_params', 'cpp/include/nabia/strategy/RiskBudgetingPortfolio.h', ('cpp', 'struct RiskBudgetingPortfolioParams {')),
    ('portfolio_validate', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('cpp', 'void validate(const RiskBudgetingPortfolioParams& p) {')),
    ('portfolio_allocate', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('cpp', 'RiskBudgetingResult allocate(')),
    ('portfolio_engine', 'cpp/src/strategy/RiskBudgetingPortfolio.cpp', ('cpp', 'PortfolioComputation run_risk_budgeting_portfolio(')),
    ('drawdown_series', 'cpp/src/strategy/PerformanceMetrics.cpp', ('cpp', 'Eigen::VectorXd drawdown_series(')),
    ('resolve_threads', 'cpp/src/common/parallel_for.hpp', ('cpp', 'inline int resolve_threads(')),
    ('cli_main', 'cpp/apps/rb_backtest/main.cpp', ('cpp', 'int main(int argc, char** argv) {')),
    ('cli_config', 'cpp/apps/rb_backtest/config.cpp', ('cpp', 'BacktestConfig parse_config(')),
    ('cli_read_prices', 'cpp/apps/rb_backtest/csv_io.cpp', ('cpp', 'DataFrame<Date> read_prices_csv(')),
    ('cli_write', 'cpp/apps/rb_backtest/csv_io.cpp', ('cpp', 'void write_backtest_csv(')),
    ('binding_backtest', 'cpp/bindings/src/rb_backtest_module.cpp', ('cpp', 'py::dict backtest(')),
    ('binding_latest', 'cpp/bindings/src/rb_backtest_module.cpp', ('cpp', 'py::dict latest_weights(')),
    ('binding_module', 'cpp/bindings/src/rb_backtest_module.cpp', ('cpp', 'PYBIND11_MODULE(_rb_backtest, m) {')),
    ('enet_params', 'cpp/include/nabia/regression/ElasticNet.h', ('cpp', 'struct ElasticNetParams {')),
    ('enet_standardize', 'cpp/src/regression/ElasticNet.cpp', ('cpp', 'Standardized standardize_data(')),
    ('enet_cd', 'cpp/src/regression/ElasticNet.cpp', ('cpp', 'class CoordinateDescent {')),
    ('enet_df', 'cpp/src/regression/ElasticNet.cpp', ('cpp', 'double degrees_of_freedom(')),
    ('enet_path', 'cpp/src/regression/ElasticNet.cpp', ('cpp', 'std::vector<ElasticNetFit> elastic_net_path(const MatrixXd& X')),
    ('regression_params', 'cpp/include/nabia/regression/RegressionEngine.h', ('cpp', 'struct RegressionEngineParams {')),
    ('regression_class', 'cpp/include/nabia/regression/RegressionEngine.h', ('cpp', 'class RegressionEngine {')),
    ('regression_run', 'cpp/src/regression/RegressionEngine.cpp', ('cpp', 'RegressionComputation run_regression_engine(')),
    ('regression_evaluate', 'cpp/src/regression/RegressionEngine.cpp', ('cpp', 'SubsetEvaluation evaluate(')),
    ('regression_criterion', 'cpp/src/regression/RegressionEngine.cpp', ('cpp', 'double in_sample_criterion(')),
    ('regression_exhaustive', 'cpp/src/regression/RegressionEngine.cpp', ('cpp', 'SearchOutcome exhaustive_search(')),
    ('regression_forward', 'cpp/src/regression/RegressionEngine.cpp', ('cpp', 'SearchOutcome forward_search(')),
    ('regression_bootstrap', 'cpp/src/regression/RegressionEngine.cpp', ('cpp', 'BootstrapSummary bootstrap(')),
    ('bridge_module', 'cpp/bindings/src/module.cpp', ('cpp', 'PYBIND11_MODULE(_core, m) {')),
    ('bridge_any_rb', 'cpp/bindings/src/bind_risk.cpp', ('cpp', 'class AnyRiskBudgeting {')),
    ('bridge_objective', 'cpp/bindings/src/bind_optimization.cpp', ('cpp', 'Objective objective(py::function f) {')),
    ('bridge_schedule', 'cpp/bindings/src/bind_strategy.cpp', ('cpp', 'std::vector<sd::BudgetChange> schedule(')),
    ('bridge_regression', 'cpp/bindings/src/bind_regression.cpp', ('cpp', 'void bind_regression(py::module_& m) {')),
    ('bridge_reexport', 'cpp/bindings/nabia_python_bridge/_reexport.py', ('py', 'export')),
    ('frames_make_params', 'cpp/bindings/nabia_python_bridge/frames.py', ('py', '_make_params')),
    ('frames_backtest', 'cpp/bindings/nabia_python_bridge/frames.py', ('py', 'backtest')),
    ('py_backtest', 'nabia_python/nabia/portfolio.py', ('py', 'backtest')),
    ('py_latest_weights', 'nabia_python/nabia/portfolio.py', ('py', 'latest_weights')),
    ('universe', 'nabia_python/nabia/marketdata/universe.py', ('between', 'EQUITY_INDICES: tuple', 'DEFAULT_UNIVERSE: tuple')),
    ('yahoo_fetch', 'nabia_python/nabia/marketdata/sources.py', ('py', 'YahooFinanceSource.fetch')),
    ('load_prices', 'nabia_python/nabia/marketdata/store.py', ('py', 'MarketDataStore.load_prices')),
    ('correlation_matrix', 'nabia_python/nabia/analytics.py', ('py', 'correlation_matrix')),
    ('average_correlation', 'nabia_python/nabia/analytics.py', ('py', 'average_correlation')),
    ('rb_config', 'nabia_python/nabia/sentiment_rb/config.py', ('py', 'CONFIG')),
    ('rb_load_config', 'nabia_python/nabia/sentiment_rb/config.py', ('py', 'load_config')),
    ('rb_ancestors', 'nabia_python/nabia/sentiment_rb/features/tree.py', ('py', 'ancestors')),
    ('rb_node_sign', 'nabia_python/nabia/sentiment_rb/features/tree.py', ('py', 'node_sign')),
    ('rb_build_features', 'nabia_python/nabia/sentiment_rb/features/transforms.py', ('py', 'build_features')),
    ('rb_synthetic_market', 'nabia_python/nabia/sentiment_rb/data/synthetic.py', ('py', 'synthetic_market')),
    ('rb_synthetic_news', 'nabia_python/nabia/sentiment_rb/data/synthetic.py', ('py', 'synthetic_news')),
    ('rb_triage_schema', 'nabia_python/nabia/sentiment_rb/llm/schemas.py', ('py', 'Triage')),
    ('rb_score_schema', 'nabia_python/nabia/sentiment_rb/llm/schemas.py', ('py', 'Score')),
    ('rb_digest_schema', 'nabia_python/nabia/sentiment_rb/llm/schemas.py', ('py', 'Digest')),
    ('rb_mock', 'nabia_python/nabia/sentiment_rb/llm/client.py', ('py', 'mock_response')),
    ('rb_fill', 'nabia_python/nabia/sentiment_rb/llm/client.py', ('py', 'fill')),
    ('rb_embed', 'nabia_python/nabia/sentiment_rb/llm/news.py', ('py', 'embed')),
    ('rb_run_news', 'nabia_python/nabia/sentiment_rb/llm/news.py', ('py', 'run_news')),
    ('rb_walk_forward', 'nabia_python/nabia/sentiment_rb/model/elastic_net.py', ('py', 'walk_forward_scores')),
    ('rb_check_rc', 'nabia_python/nabia/sentiment_rb/portfolio/backtest.py', ('py', 'check_risk_contributions')),
    ('rb_tables', 'nabia_python/nabia/sentiment_rb/report/tables.py', ('py', 'tables')),
    ('rb_plots', 'nabia_python/nabia/sentiment_rb/report/figures.py', ('py', 'plots')),
    ('rb_plot_sweep', 'nabia_python/nabia/sentiment_rb/report/figures.py', ('py', 'plot_sweep')),
    ('rb_main', 'nabia_python/nabia/sentiment_rb/__main__.py', ('py', 'main')),
    ('rb_test_short', 'nabia_python/tests/test_sentiment_rb.py', ('py', 'test_short_positions_keep_their_risk_budget')),
    ('rb_test_simulator', 'nabia_python/tests/test_sentiment_rb.py', ('py', 'test_simulator_reproduces_the_nabia_engine_when_all_long')),
]

#: Pieces the packaging chapter \\filepiece's.
PIECES = [
    "workflow-head", "job-wheels", "job-collect", "job-release", "job-windows-script",
    "pyproject-build-system", "pyproject-project", "pyproject-project-optional-dependencies",
    "pyproject-tool-scikit-build",
    "pyproject-tool-scikit-build-cmake-define", "pyproject-tool-cibuildwheel",
    "pyproject-tool-cibuildwheel-linux", "pyproject-tool-cibuildwheel-macos",
    "bindings-cmake", "build-wheels-sh", "build-wheels-ps1", "setup-env-sh",
]


def split_workflow(text: str) -> dict[str, str]:
    """Head (everything before the first job) and one piece per job under `jobs:`."""
    lines = text.splitlines()
    out: dict[str, list[str]] = {"workflow-head": []}
    current = "workflow-head"
    in_jobs = False
    for line in lines:
        if line.rstrip() == "jobs:":
            in_jobs = True
            continue
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line) if in_jobs else None
        if m:
            current = f"job-{m.group(1)}"
            out[current] = []
        out[current].append(line)
    return {k: "\n".join(v).strip("\n") + "\n" for k, v in out.items()}


def split_toml(text: str) -> dict[str, str]:
    """One piece per [table], with the comment lines just above it."""
    out: dict[str, list[str]] = {}
    pending: list[str] = []
    current = None
    for line in text.splitlines():
        m = re.match(r"^\[([^\]]+)\]\s*$", line)
        if m:
            current = "pyproject-" + m.group(1).replace(".", "-")
            out[current] = pending + [line]
            pending = []
        elif current is None or (line.startswith("#") and not out[current][-1].strip()):
            pending.append(line)  # a comment block introducing the next table
        else:
            if pending:
                out[current].extend(pending)
                pending = []
            out[current].append(line)
    return {k: "\n".join(v).strip("\n") + "\n" for k, v in out.items()}


_ASCII_STRICT = {"\u2013": "-", "\u2014": "--", "\u2026": "...", "\u2192": "->", "\u2190": "<-",
          "\u2265": ">=", "\u2264": "<=", "\u00d7": "x", "\u2018": "'", "\u2019": "'",
          "\u201c": '"', "\u201d": '"'}


def _ascii_strict(text: str) -> str:
    """pdflatex listings need ASCII: replace the few typographic characters used in comments."""
    text = "".join(_ASCII_STRICT.get(c, c) for c in text)
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        sys.exit(f"extract_snippets: non-ASCII characters to map: {bad}")
    return text




def packaging_pieces(root: Path, out: Path) -> int:
    """The files the packaging chapter shows, split into pieces (\\filepiece{name}): wheels.yml per job,
    cpp/pyproject.toml per table, and whole scripts. A missing piece stops the build."""
    pieces: dict[str, str] = {}
    pieces.update(split_workflow((root / ".github/workflows/wheels.yml").read_text()))
    pieces.update(split_toml((root / "cpp/pyproject.toml").read_text()))
    pieces["bindings-cmake"] = (root / "cpp/bindings/CMakeLists.txt").read_text()
    pieces["build-wheels-sh"] = (root / "cpp/scripts/build_wheels.sh").read_text()
    pieces["build-wheels-ps1"] = (root / "cpp/scripts/build_wheels.ps1").read_text()
    pieces["setup-env-sh"] = (root / "nabia_python/scripts/setup_env.sh").read_text()
    missing = [name for name in PIECES if name not in pieces]
    if missing:
        sys.exit(f"extract_snippets: packaging pieces not found: {missing}")
    for name, text in pieces.items():
        (out / f"{name}.txt").write_text(_ascii_strict(text))
    return len(pieces)



def _python(source: str, dotted: str) -> tuple[int, int]:
    """Line range of a (possibly nested) function, class or module constant."""
    tree = ast.parse(source)
    parts = dotted.split(".")

    def search(node: ast.AST, path: list[str]) -> tuple[int, int] | None:
        for child in ast.iter_child_nodes(node):
            name = getattr(child, "name", None)
            if isinstance(child, ast.Assign) and len(path) == 1:
                if any(isinstance(t, ast.Name) and t.id == path[0] for t in child.targets):
                    return child.lineno, child.end_lineno
            if isinstance(child, ast.AnnAssign) and len(path) == 1 and getattr(child.target, "id", "") == path[0]:
                return child.lineno, child.end_lineno
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and name == path[0]:
                if len(path) == 1:
                    first = min([d.lineno for d in child.decorator_list] + [child.lineno])
                    return first, child.end_lineno
                found = search(child, path[1:])
                if found:
                    return found
            elif not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found = search(child, path)  # look inside if / with / try blocks
                if found:
                    return found
        return None

    found = search(tree, parts)
    if found is None:
        raise LookupError(f"{dotted} not found")
    return found


def _constant(source: str, name: str) -> str:
    """Value of a module-level string constant."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError(f"{name} not found")


def _cpp(lines: list[str], signature: str) -> tuple[int, int]:
    start = next(i for i, line in enumerate(lines) if signature in line)
    depth, opened = 0, False
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth == 0:
            return start + 1, i + 1
    raise LookupError(f"no closing brace for {signature}")


def _between(lines: list[str], start: str, end: str) -> tuple[int, int]:
    first = next(i for i, line in enumerate(lines) if start.strip("\n") in line)
    last = next(i for i in range(first, len(lines)) if end.strip("\n") in lines[i])
    return first + 1, last + 1


# pdflatex listings handle ASCII only: map the few non-ASCII characters used in comments and strings.
_ASCII = {"\u2026": "...", "\u2013": "-", "\u2014": "--", "\u2192": "->", "\u2190": "<-", "\u2265": ">=",
          "\u2264": "<=", "\u00d7": "x", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u25cf": "*"}


def _ascii(line: str) -> str:
    for char, replacement in _ASCII.items():
        line = line.replace(char, replacement)
    return line.encode("ascii", "replace").decode("ascii")












def _dedent(lines: list[str]) -> list[str]:
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    cut = min(indents) if indents else 0
    return [l[cut:] if l.strip() else "" for l in lines]


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = ["% Generated by extract_snippets.py: source location of each excerpt."]
    for name, path, selector in SNIPPETS + REFS:
        source = (root / path).read_text()
        lines = source.splitlines()
        kind = selector[0]
        try:
            if kind in ("py", "text"):
                first, last = _python(source, selector[1])
            elif kind == "cpp":
                first, last = _cpp(lines, selector[1])
            else:
                first, last = _between(lines, selector[1], selector[2])
        except (LookupError, StopIteration) as error:  # a renamed function: fail loudly, naming the entry
            sys.exit(f"extract_snippets: {name} ({path}): {selector[1]!r} not found ({error})")
        location = f"{path}:{first}" + (f"--{last}" if last != first else "")
        manifest.append(rf"\expandafter\def\csname snippet@{name}\endcsname{{\detokenize{{{location}}}}}")
        if (name, path, selector) in REFS:
            continue
        if kind == "text":
            excerpt = [_ascii(l) for l in _constant(source, selector[1]).strip("\n").splitlines()]
        else:
            excerpt = [_ascii(l) for l in _dedent(lines[first - 1:last])]
        if len(excerpt) > MAX_LINES and name not in FULL:
            omitted = len(excerpt) - MAX_LINES
            comment = "#" if path.endswith(".py") else "//"
            excerpt = excerpt[:MAX_LINES] + [f"{comment} ... ({omitted} more lines in the source)"]
        (out / f"{name}.txt").write_text("\n".join(excerpt) + "\n")
    (out / "snippets.tex").write_text("\n".join(manifest) + "\n")
    n = packaging_pieces(root, out)
    print(f"extracted {len(SNIPPETS)} snippets, {len(REFS)} references and {n} file pieces into {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
