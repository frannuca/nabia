"""Extract source excerpts for the project methodology document.

Run by the documentation build (doc/CMakeLists.txt) before pdflatex, so the
code shown in the PDF is always the current code:

    python3 extract_snippets.py <repository root> <output directory>

For every entry of SNIPPETS it writes <name>.txt (the excerpt) and adds a line to
snippets.tex defining \\snippetsource{<name>} as "<file>:<first>-<last>"; REFS
entries only define the location (\\coderef{<name>} in the document). It also
writes, parsed from the source so they match what runs:
  tools.tex     the MCP tool catalogue: every tool's name, description (its
                docstring, which is what the model is shown), parameters and
                whether it writes, plus the pydantic schemas of the tool inputs;
  config.tex    the AI configuration fields with their defaults and docs.

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
FULL = {"news_system", "stats_system", "execution_system", "backtest_note", "as_of_rule", "team_calls",
        "engine_phase2", "runtime_loop"}

SNIPPETS: list[tuple[str, str, tuple]] = [
    # --- C++ core ---------------------------------------------------------------------
    ("df_drop_missing", "include/nabia/dataframe/DataFrame.h", ("cpp", "[[nodiscard]] DataFrame drop_missing() const {")),
    ("ewma_impl", "src/signal/EWMA.cpp", ("cpp", "struct EWMA::Impl {")),
    ("lbfgs_line_search", "src/optimization/convex/LBFGS.cpp", ("between", "double t = 1.0;", "if (!accepted) {")),
    ("parallel_evaluator", "src/optimization/nongradopt/parallel_evaluator.hpp",
     ("cpp", "[[nodiscard]] std::vector<TResult> evaluate(const std::vector<T>& inputs) const {")),
    ("vol_gradient", "src/risk/IRiskMeasure.cpp", ("cpp", "Eigen::VectorXd IRiskMeasure::portfolio_volatility_gradient")),
    ("var_measure", "src/risk/measures/ValueAtRisk.cpp", ("between", "double ValueAtRisk::compute_risk", "return -expected_returns_ + z_")),
    ("rb_objective", "src/risk/RiskBudgeting.cpp", ("between", "// f(y) = R(y) - b' log(y)", "return measure.compute_risk(y) - b.dot")),
    ("cov_estimate", "src/strategy/RiskBudgetingPortfolio.cpp", ("cpp", "WindowMoments estimate(")),
    ("solve_target", "src/strategy/RiskBudgetingPortfolio.cpp", ("cpp", "Target solve_target(")),
    ("engine_phase1", "src/strategy/RiskBudgetingPortfolio.cpp", ("between", "// Phase 1 (parallel)", "c.threads_used = threads;")),
    ("engine_phase2", "src/strategy/RiskBudgetingPortfolio.cpp", ("between", "// Phase 2 (sequential)", "equity = new_equity;")),
    ("parallel_for", "src/common/parallel_for.hpp", ("cpp", "void parallel_for(")),
    ("performance", "src/strategy/PerformanceMetrics.cpp", ("cpp", "PerformanceMetrics compute_performance(")),
    ("binding_prepare", "python/src/nabia_module.cpp", ("cpp", "Prepared prepare(")),
    ("binding_backtest_core", "python/src/nabia_module.cpp", ("between", "    std::unique_ptr<PortfolioResult<Date>> r;\n", "        r = std::make_unique<PortfolioResult<Date>>(schedule")),
    # --- Python: portfolio and market data --------------------------------------------
    ("py_backtest_inputs", "python/nabia/portfolio.py", ("py", "_inputs")),
    ("py_to_days", "python/nabia/portfolio.py", ("py", "_to_days")),
    ("crypto_fetch", "python/nabia/marketdata/sources.py", ("py", "CryptoSource.fetch")),
    ("upsert_rows", "python/nabia/marketdata/store.py", ("py", "upsert_rows")),
    ("update_market_data", "python/nabia/marketdata/download.py", ("py", "update_market_data")),
    # --- Python: analytics -----------------------------------------------------------
    ("aligned_returns", "python/nabia/ai/analytics.py", ("py", "aligned_returns")),
    ("rolling_volatility", "python/nabia/ai/analytics.py", ("py", "rolling_volatility")),
    ("trend_signals", "python/nabia/ai/analytics.py", ("py", "trend_signals")),
    ("market_regime", "python/nabia/ai/analytics.py", ("py", "market_regime")),
    ("trend_tilted_budgets", "python/nabia/ai/analytics.py", ("py", "trend_tilted_budgets")),
    ("project_budgets", "python/nabia/ai/analytics.py", ("py", "project_budgets")),
    # --- Python: MCP servers -----------------------------------------------------------
    ("tool_decorator", "python/nabia/ai/servers/common.py", ("py", "tool")),
    ("parse_as_of", "python/nabia/ai/servers/common.py", ("py", "parse_as_of")),
    ("news_article", "python/nabia/ai/servers/news.py", ("py", "NewsArticle")),
    ("store_news_articles", "python/nabia/ai/servers/news.py", ("py", "build_server.store_news_articles")),
    ("fetch_news_feeds", "python/nabia/ai/servers/news.py", ("py", "build_server.fetch_news_feeds")),
    ("news_summary", "python/nabia/ai/servers/news.py", ("py", "build_server.news_summary")),
    ("set_risk_budgets", "python/nabia/ai/servers/portfolio.py", ("py", "build_server.set_risk_budgets")),
    ("compute_weights", "python/nabia/ai/servers/portfolio.py", ("py", "compute_weights")),
    # --- Python: agents ----------------------------------------------------------------
    ("news_system", "python/nabia/ai/agents/team.py", ("text", "NEWS_SYSTEM")),
    ("as_of_rule", "python/nabia/ai/agents/team.py", ("text", "AS_OF_RULE")),
    ("runtime_loop", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime._loop")),
    ("runtime_call_tool", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime._call_tool")),
    ("ollama_response", "python/nabia/ai/agents/ollama.py", ("py", "OllamaClient._response")),
    ("decision_dates", "python/nabia/ai/pipeline.py", ("py", "decision_dates")),
    ("evaluate_decisions", "python/nabia/ai/pipeline.py", ("py", "evaluate_decisions")),
    ("policy_config", "python/nabia/ai/pipeline.py", ("py", "policy_config")),
    # --- Python: web app -----------------------------------------------------------------
    ("job_log_handler", "python/nabia/web/jobs.py", ("py", "JobLogHandler")),
    ("job_run", "python/nabia/web/jobs.py", ("py", "JobRunner._run")),
    # --- prompts (plain text) and the code that sends them ------------------------------
    ("stats_system", "python/nabia/ai/agents/team.py", ("text", "STATS_SYSTEM")),
    ("execution_system", "python/nabia/ai/agents/team.py", ("text", "EXECUTION_SYSTEM")),
    ("backtest_note", "python/nabia/ai/agents/team.py", ("text", "BACKTEST_NOTE")),
    ("team_calls", "python/nabia/ai/agents/team.py", ("between", "    async def collect_news", "            effort=self.config.effort.get(\"execution\"")),
    ("team_fill", "python/nabia/ai/agents/team.py", ("py", "AgentTeam._fill")),
    ("live_output", "python/nabia/ai/agents/runtime.py", ("py", "LiveOutput")),
    ("runtime_create", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime._create")),
    ("ollama_stream", "python/nabia/ai/agents/ollama.py", ("py", "_Stream")),
    # --- more C++ ------------------------------------------------------------------------
    ("lbfgs_two_loop", "src/optimization/convex/LBFGS.cpp", ("between", "// Two-loop recursion", "d = -d;")),
    ("al_function", "src/optimization/convex/AugmentedLagrangian.cpp", ("between", "// L_rho(x) = f", "        return value;")),
    ("al_updates", "src/optimization/convex/AugmentedLagrangian.cpp", ("between", "// Violation measured before", "inner_tol = std::max(p.tolerance, 0.1 * inner_tol);")),
    ("sqp_bfgs", "src/optimization/convex/SQP.cpp", ("cpp", "void damped_bfgs_update(")),
    ("sqp_merit", "src/optimization/convex/SQP.cpp", ("between", "// l1 merit function phi", "            t *= options_.backtrack;")),
    ("nm_step", "src/optimization/nongradopt/NelderMead.cpp", ("between", "const VectorXd centroid", "            values(j) = f(simplex.col(j));")),
    ("de_generation", "src/optimization/nongradopt/DifferentialEvolution.cpp", ("between", "const double F = p.mutation_min", "trials(i, j) = v;")),
    ("rb_solve", "src/risk/RiskBudgeting.cpp", ("between", "// L-BFGS stops on ||grad||_inf", "inner_tol *= 0.5 * params.tolerance / err;")),
    ("normal_quantile", "src/stats/normal.cpp", ("between", "    // ... refined with one Halley step", "return x - u / (1.0 + 0.5 * x * u);")),
    ("kalman_step", "src/forecast/KalmanFilter.cpp", ("between", "        if (t > 0) {  // predict", "        filtered_covariances_[ts] = P;")),
    ("kalman_smoother", "src/forecast/KalmanFilter.cpp", ("cpp", "Eigen::MatrixXd KalmanFilter::smoothed_states() const {")),
    ("gp_nll", "src/forecast/GaussianProcess.cpp", ("between", "// d(-log p)/d theta_j", "            return -f->log_ml;")),
    ("gp_predict", "src/forecast/GaussianProcess.cpp", ("cpp", "Forecast GaussianProcess::predict(")),
    ("ses_fit", "src/forecast/exponential_smoothing.cpp", ("between", "    // Initialise the level", "        out.variance(h) = sigma2")),
    ("schedule_rows", "include/nabia/strategy/RiskBudgetingPortfolio.h", ("between", "        for (Eigen::Index s = 0; s < budget_schedule.rows(); ++s) {", "            schedule.push_back(")),
    ("budgets_at", "src/strategy/RiskBudgetingPortfolio.cpp", ("between", "    // Budgets in force at a rebalance row", "    };")),
]

#: Location-only references (\\coderef{name}): where a concept lives, without an excerpt.
REFS: list[tuple[str, str, tuple]] = [
    ("time_index", "include/nabia/dataframe/TimeIndex.h", ("between", "concept TimeIndex", "concept TimeIndex")),
    ("dataframe", "include/nabia/dataframe/DataFrame.h", ("cpp", "class DataFrame {")),
    ("timeseries", "include/nabia/dataframe/TimeSeries.h", ("cpp", "class TimeSeries {")),
    ("pct_change", "include/nabia/dataframe/DataFrame.h", ("cpp", "[[nodiscard]] DataFrame pct_change() const {")),
    ("log_returns", "include/nabia/dataframe/DataFrame.h", ("cpp", "[[nodiscard]] DataFrame log_returns() const {")),
    ("ifilter", "include/nabia/signal/IFilter.h", ("cpp", "class IFilter")),
    ("ewma_class", "include/nabia/signal/EWMA.h", ("cpp", "class EWMA: public IFilter<EWMAParams> {")),
    ("ewma_alphas", "src/signal/EWMA.cpp", ("between", "double EWMA::alpha_from_span", "double EWMA::alpha_from_halflife")),
    ("butterworth_design", "src/signal/ButterworthBandPass.cpp", ("cpp", "void design(Filter& filter")),
    ("butterworth_params", "include/nabia/signal/ButterworthBandPass.h", ("cpp", "struct ButterworthBandPassParams {")),
    ("moving_average", "src/signal/smoothing.cpp", ("cpp", "std::vector<double> moving_average(")),
    ("iforecaster", "include/nabia/forecast/IForecaster.h", ("cpp", "class IForecaster")),
    ("ses_forecast", "src/forecast/exponential_smoothing.cpp", ("cpp", "Forecast SimpleExponentialSmoothing::forecast(")),
    ("kalman_params", "include/nabia/forecast/KalmanFilter.h", ("cpp", "struct KalmanFilterParams {")),
    ("kalman_models", "src/forecast/KalmanFilter.cpp", ("between", "KalmanFilterParams KalmanFilterParams::local_level", "    p.initial_covariance = initial_variance * MatrixXd::Identity(2, 2);")),
    ("kalman_forecast", "src/forecast/KalmanFilter.cpp", ("cpp", "Forecast KalmanFilter::forecast(")),
    ("gp_fit", "src/forecast/GaussianProcess.cpp", ("cpp", "void GaussianProcess::fit(")),
    ("gp_factorize", "src/forecast/GaussianProcess.cpp", ("cpp", "std::optional<Factorization> factorize(")),
    ("kernels", "src/forecast/Kernels.cpp", ("between", "// --- Squared exponential", "    return std::make_unique<ProductKernel>(k1_->clone(), k2_->clone());")),
    ("ioptimizer", "include/nabia/optimization/IOptimizer.h", ("cpp", "class IOptimizer")),
    ("lbfgs_params", "include/nabia/optimization/convex/LBFGS.h", ("cpp", "struct LBFGSParams")),
    ("lbfgs_optimize", "src/optimization/convex/LBFGS.cpp", ("cpp", "Eigen::VectorXd LBFGS::optimize(")),
    ("constraints", "include/nabia/optimization/convex/Constraints.h", ("cpp", "struct LinearConstraint")),
    ("al_options", "include/nabia/optimization/convex/AugmentedLagrangian.h", ("cpp", "struct AugmentedLagrangianOptions")),
    ("al_optimize", "src/optimization/convex/AugmentedLagrangian.cpp", ("cpp", "Eigen::VectorXd AugmentedLagrangian::optimize(")),
    ("sqp_options", "include/nabia/optimization/convex/SQP.h", ("cpp", "struct SQPOptions")),
    ("sqp_qp", "src/optimization/convex/SQP.cpp", ("cpp", "QPSolution solve_qp(")),
    ("sqp_optimize", "src/optimization/convex/SQP.cpp", ("cpp", "Eigen::VectorXd SQP::optimize(")),
    ("nm_coefficients", "src/optimization/nongradopt/NelderMead.cpp", ("between", "const double alpha = p.adaptive", "const double sigma = p.adaptive")),
    ("nm_optimize", "src/optimization/nongradopt/NelderMead.cpp", ("cpp", "Eigen::VectorXd NelderMead::optimize(")),
    ("box", "src/optimization/nongradopt/box.hpp", ("cpp", "struct Box {")),
    ("de_params", "include/nabia/optimization/nongradopt/DifferentialEvolution.h", ("cpp", "struct DifferentialEvolutionParams")),
    ("de_optimize", "src/optimization/nongradopt/DifferentialEvolution.cpp", ("cpp", "Eigen::VectorXd DifferentialEvolution::optimize(")),
    ("irisk", "include/nabia/risk/IRiskMeasure.h", ("cpp", "class IRiskMeasure")),
    ("risk_contributions", "src/risk/IRiskMeasure.cpp", ("cpp", "Eigen::VectorXd IRiskMeasure::compute_risk_contributions")),
    ("es_measure", "src/risk/measures/ExpectedShortfall.cpp", ("between", "ExpectedShortfall::ExpectedShortfall(", "return -expected_returns_ + k_")),
    ("volatility_measure", "src/risk/measures/Volatility.cpp", ("between", "double Volatility::compute_risk", "return portfolio_volatility_gradient(x);")),
    ("rb_start", "src/risk/RiskBudgeting.cpp", ("cpp", "VectorXd default_start(")),
    ("rb_budgets", "src/risk/RiskBudgeting.cpp", ("cpp", "VectorXd normalized_budgets(")),
    ("rb_solver", "src/risk/RiskBudgeting.cpp", ("cpp", "RiskBudgetingResult solve_risk_budgeting(")),
    ("rb_template", "include/nabia/risk/RiskBudgeting.h", ("cpp", "class RiskBudgeting")),
    ("portfolio_params", "include/nabia/strategy/RiskBudgetingPortfolio.h", ("cpp", "struct RiskBudgetingPortfolioParams {")),
    ("portfolio_validate", "src/strategy/RiskBudgetingPortfolio.cpp", ("cpp", "void validate(const RiskBudgetingPortfolioParams& p) {")),
    ("portfolio_allocate", "src/strategy/RiskBudgetingPortfolio.cpp", ("cpp", "RiskBudgetingResult allocate(")),
    ("portfolio_engine", "src/strategy/RiskBudgetingPortfolio.cpp", ("cpp", "PortfolioComputation run_risk_budgeting_portfolio(")),
    ("drawdown_series", "src/strategy/PerformanceMetrics.cpp", ("cpp", "Eigen::VectorXd drawdown_series(")),
    ("resolve_threads", "src/common/parallel_for.hpp", ("cpp", "inline int resolve_threads(")),
    ("cli_main", "apps/rb_backtest/main.cpp", ("cpp", "int main(int argc, char** argv) {")),
    ("cli_config", "apps/rb_backtest/config.cpp", ("cpp", "BacktestConfig parse_config(")),
    ("cli_read_prices", "apps/rb_backtest/csv_io.cpp", ("cpp", "DataFrame<Date> read_prices_csv(")),
    ("cli_write", "apps/rb_backtest/csv_io.cpp", ("cpp", "void write_backtest_csv(")),
    ("binding_backtest", "python/src/nabia_module.cpp", ("cpp", "py::dict backtest(")),
    ("binding_latest", "python/src/nabia_module.cpp", ("cpp", "py::dict latest_weights(")),
    ("binding_module", "python/src/nabia_module.cpp", ("cpp", "PYBIND11_MODULE(_nabia, m) {")),
    ("py_backtest", "python/nabia/portfolio.py", ("py", "backtest")),
    ("py_latest_weights", "python/nabia/portfolio.py", ("py", "latest_weights")),
    ("universe", "python/nabia/marketdata/universe.py", ("between", "EQUITY_INDICES: tuple", "DEFAULT_UNIVERSE: tuple")),
    ("yahoo_fetch", "python/nabia/marketdata/sources.py", ("py", "YahooFinanceSource.fetch")),
    ("load_prices", "python/nabia/marketdata/store.py", ("py", "MarketDataStore.load_prices")),
    ("correlation_matrix", "python/nabia/ai/analytics.py", ("py", "correlation_matrix")),
    ("average_correlation", "python/nabia/ai/analytics.py", ("py", "average_correlation")),
    ("ai_config", "python/nabia/ai/config.py", ("py", "AIConfig")),
    ("server_context", "python/nabia/ai/servers/common.py", ("py", "ServerContext")),
    ("server_prices", "python/nabia/ai/servers/common.py", ("py", "ServerContext.prices")),
    ("clean", "python/nabia/ai/servers/common.py", ("py", "clean")),
    ("news_server", "python/nabia/ai/servers/news.py", ("py", "build_server")),
    ("stats_server", "python/nabia/ai/servers/stats.py", ("py", "build_server")),
    ("portfolio_server", "python/nabia/ai/servers/portfolio.py", ("py", "build_server")),
    ("read_feed", "python/nabia/ai/servers/news.py", ("py", "_read_feed")),
    ("query_news", "python/nabia/ai/servers/news.py", ("py", "build_server.query_news")),
    ("stats_report_model", "python/nabia/ai/servers/stats.py", ("py", "StatsReport")),
    ("runtime_run", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime.run")),
    ("runtime_client", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime.client")),
    ("server_parameters", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime._server_parameters")),
    ("ollama_messages", "python/nabia/ai/agents/ollama.py", ("py", "OllamaClient._messages")),
    ("ollama_chat", "python/nabia/ai/agents/ollama.py", ("py", "OllamaClient._chat")),
    ("team_run", "python/nabia/ai/agents/team.py", ("py", "AgentTeam._run")),
    ("team_universe", "python/nabia/ai/agents/team.py", ("py", "AgentTeam._universe")),
    ("run_daily", "python/nabia/ai/pipeline.py", ("py", "run_daily")),
    ("run_backtest", "python/nabia/ai/pipeline.py", ("py", "run_backtest")),
    ("trend_rule_decider", "python/nabia/ai/pipeline.py", ("py", "trend_rule_decider")),
    ("agent_decider", "python/nabia/ai/pipeline.py", ("py", "agent_decider")),
    ("make_decider", "python/nabia/ai/pipeline.py", ("py", "make_decider")),
    ("db_tables", "python/nabia/ai/db.py", ("between", "news_table = Table(", "backtest_runs_table = Table(")),
    ("store_news", "python/nabia/ai/db.py", ("py", "AgentStore.store_news")),
    ("save_decision", "python/nabia/ai/db.py", ("py", "AgentStore.save_decision")),
    ("web_app", "python/nabia/web/app.py", ("py", "create_app")),
    ("web_news", "python/nabia/web/app.py", ("py", "create_app.news")),
    ("job_runner", "python/nabia/web/jobs.py", ("py", "JobRunner")),
    ("job_backtest", "python/nabia/web/jobs.py", ("py", "JobRunner._backtest")),
    ("log_viewer", "python/nabia/web/static/app.js", ("between", "async function openJobLog", "async function openJobLog")),
]


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


_TEX = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
        "^": r"\textasciicircum{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}"}


def tex(text: str) -> str:
    """Escapes plain text for LaTeX; `code` spans become \\texttt."""
    out = "".join(_TEX.get(c, c) for c in _ascii(text))
    return re.sub(r"`([^`]+)`", r"\\texttt{\1}", out)


def _paragraphs(doc: str) -> str:
    """A docstring as LaTeX paragraphs (blank lines separate them)."""
    return "\n\n".join(tex(" ".join(p.split())) for p in doc.split("\n\n") if p.strip())


def _tool_catalogue(root: Path) -> str:
    """The MCP servers' tools, parsed from their source: what the model is shown."""
    out = ["% Generated by extract_snippets.py from python/nabia/ai/servers/*.py"]
    for module in ("news", "stats", "portfolio"):
        path = f"python/nabia/ai/servers/{module}.py"
        source = (root / path).read_text()
        tree = ast.parse(source)
        server_name, instructions = module, ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "MCPServer":
                server_name = ast.literal_eval(node.args[0])
                instructions = next((ast.literal_eval(k.value) for k in node.keywords if k.arg == "instructions"), "")
        out.append(rf"\subsection{{Server \texttt{{{tex(server_name)}}}}}\label{{server:{module}}}")
        out.append(rf"\noindent\textbf{{Module:}} \texttt{{nabia.ai.servers.{module}}} "
                   rf"(\texttt{{\detokenize{{{path}}}}}). \textbf{{Instructions sent to the client:}} "
                   rf"\emph{{{tex(instructions)}}}")

        def visit(node: ast.AST, writes: bool) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.If) and ast.unparse(child.test) == "not ctx.read_only":
                    visit(child, True)
                elif isinstance(child, ast.FunctionDef) and any(
                        isinstance(d, ast.Call) and getattr(d.func, "id", "") == "tool" for d in child.decorator_list):
                    tools.append((child, writes))
                elif not isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                    visit(child, writes)
                elif isinstance(child, ast.FunctionDef) and child.name == "build_server":
                    visit(child, writes)

        tools: list[tuple[ast.FunctionDef, bool]] = []
        visit(tree, False)
        for fn, writes in tools:
            first = min(d.lineno for d in fn.decorator_list)
            args = fn.args.args
            defaults = [None] * (len(args) - len(fn.args.defaults)) + list(fn.args.defaults)
            mode = "writes; removed in read-only mode" if writes else "read-only"
            out.append(rf"\toolentry{{{tex(fn.name)}}}{{{mode}}}{{\detokenize{{{path}:{first}--{fn.end_lineno}}}}}")
            out.append(_paragraphs(ast.get_docstring(fn) or ""))
            if args:
                out.append(r"\begin{toolparams}")
                for a, default in zip(args, defaults):
                    kind = tex(ast.unparse(a.annotation)) if a.annotation else ""
                    extra = "required" if default is None else "default " + tex(ast.unparse(default))
                    out.append(rf"\item[\texttt{{{tex(a.arg)}}}] \texttt{{{kind}}}, {extra}")
                out.append(r"\end{toolparams}")
            else:
                out.append(r"\noindent\emph{No parameters.}")
        models = [n for n in tree.body if isinstance(n, ast.ClassDef) and any(getattr(b, "id", "") == "BaseModel" for b in n.bases)]
        for model in models:
            out.append(rf"\modelentry{{{tex(model.name)}}}{{\detokenize{{{path}:{model.lineno}--{model.end_lineno}}}}}")
            if ast.get_docstring(model):
                out.append(_paragraphs(ast.get_docstring(model)))
            out.append(r"\begin{toolparams}")
            for field in model.body:
                if not isinstance(field, ast.AnnAssign):
                    continue
                notes = []
                if isinstance(field.value, ast.Call):
                    for k in field.value.keywords:
                        if k.arg == "description":
                            notes.append(tex(ast.literal_eval(k.value)))
                        elif k.arg in ("ge", "le"):
                            notes.append(("$\\ge$ " if k.arg == "ge" else "$\\le$ ") + tex(ast.unparse(k.value)))
                out.append(rf"\item[\texttt{{{tex(field.target.id)}}}] \texttt{{{tex(ast.unparse(field.annotation))}}}"
                           + (". " + "; ".join(notes) if notes else ""))
            out.append(r"\end{toolparams}")
    return "\n".join(out) + "\n"


def _config_table(root: Path) -> str:
    """AIConfig fields: name, default and the docstring that follows each field."""
    path = "python/nabia/ai/config.py"
    tree = ast.parse((root / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AIConfig")
    rows, body = [], cls.body
    for i, node in enumerate(body):
        if isinstance(node, ast.AnnAssign):
            default = ast.unparse(node.value) if node.value is not None else ""
            default = re.sub(r"field\(default_factory=lambda: (.*)\)", r"\1", default)
            named = {"list(DEFAULT_SYMBOLS)": "the 9 symbols of Table~\\ref{tab:universe-default}",
                     "dict(DEFAULT_PORTFOLIO)": "Table~\\ref{tab:portfolio}",
                     "dict(DEFAULT_NEWS_FEEDS)": "10 feeds, \\S\\ref{sec:news-collection}"}
            doc = ""
            if i + 1 < len(body) and isinstance(body[i + 1], ast.Expr) and isinstance(getattr(body[i + 1], "value", None), ast.Constant):
                doc = " ".join(str(body[i + 1].value.value).split())
            if default in named:
                shown = named[default]
            else:
                shown = r"\texttt{" + tex(default if len(default) <= 40 else default[:37] + "...") + "}"
            rows.append(rf"\texttt{{{tex(node.target.id)}}} & {shown} & {tex(doc)} \\")
    head = [f"% Generated by extract_snippets.py from {path}",
            r"\begin{tabularx}{\linewidth}{@{}l >{\raggedright\arraybackslash}p{4.2cm} X@{}}",
            r"\toprule", r"Field & Default & Meaning \\", r"\midrule"]
    return "\n".join(head + rows + [r"\bottomrule", r"\end{tabularx}"]) + "\n"


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
    (out / "tools.tex").write_text(_tool_catalogue(root))
    (out / "config.tex").write_text(_config_table(root))
    print(f"extracted {len(SNIPPETS)} snippets and {len(REFS)} references into {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
