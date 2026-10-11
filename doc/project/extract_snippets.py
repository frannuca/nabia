"""Extract source excerpts for the project methodology document.

Run by the documentation build (doc/CMakeLists.txt) before pdflatex, so the
code shown in the PDF is always the current code:

    python3 extract_snippets.py <repository root> <output directory>

For every entry of SNIPPETS it writes <name>.txt (the excerpt) and adds a line to
snippets.tex defining \\snippetsource{<name>} as "<file>:<first>-<last>".

Selectors:
  ("py", "name")               Python function / class / constant, found with ast;
                               dotted names reach into classes and functions
                               ("AgentRuntime._loop", "build_server.news_summary").
  ("cpp", "signature text")    from the line containing the text to the brace
                               that closes the first "{" after it.
  ("between", start, end)      from the line containing `start` to the line
                               containing `end` (inclusive).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_LINES = 70

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
    ("news_system", "python/nabia/ai/agents/team.py", ("py", "NEWS_SYSTEM")),
    ("as_of_rule", "python/nabia/ai/agents/team.py", ("py", "AS_OF_RULE")),
    ("runtime_loop", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime._loop")),
    ("runtime_call_tool", "python/nabia/ai/agents/runtime.py", ("py", "AgentRuntime._call_tool")),
    ("ollama_response", "python/nabia/ai/agents/ollama.py", ("py", "OllamaClient._response")),
    ("decision_dates", "python/nabia/ai/pipeline.py", ("py", "decision_dates")),
    ("evaluate_decisions", "python/nabia/ai/pipeline.py", ("py", "evaluate_decisions")),
    ("policy_config", "python/nabia/ai/pipeline.py", ("py", "policy_config")),
    # --- Python: web app -----------------------------------------------------------------
    ("job_log_handler", "python/nabia/web/jobs.py", ("py", "JobLogHandler")),
    ("job_run", "python/nabia/web/jobs.py", ("py", "JobRunner._run")),
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
    for name, path, selector in SNIPPETS:
        source = (root / path).read_text()
        lines = source.splitlines()
        kind = selector[0]
        if kind == "py":
            first, last = _python(source, selector[1])
        elif kind == "cpp":
            first, last = _cpp(lines, selector[1])
        else:
            first, last = _between(lines, selector[1], selector[2])
        excerpt = [_ascii(l) for l in _dedent(lines[first - 1:last])]
        if len(excerpt) > MAX_LINES:
            omitted = len(excerpt) - MAX_LINES
            comment = "#" if path.endswith(".py") else "//"
            excerpt = excerpt[:MAX_LINES] + [f"{comment} ... ({omitted} more lines in the source)"]
        (out / f"{name}.txt").write_text("\n".join(excerpt) + "\n")
        location = f"{path}:{first}" + (f"--{last}" if last != first else "")
        manifest.append(rf"\expandafter\def\csname snippet@{name}\endcsname{{\detokenize{{{location}}}}}")
    (out / "snippets.tex").write_text("\n".join(manifest) + "\n")
    print(f"extracted {len(SNIPPETS)} snippets into {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
