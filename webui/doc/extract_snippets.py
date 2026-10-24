"""Extract source excerpts for nabia_webui_reference.tex.

Run by doc/build.sh before pdflatex, so the
code shown in the PDF is always the current code:

    python3 extract_snippets.py <webui folder> <output directory>

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
    ('trend_tilted_budgets', '../python/nabia/analytics.py', ('py', 'trend_tilted_budgets')),
    ('project_budgets', '../python/nabia/analytics.py', ('py', 'project_budgets')),
    ('tool_decorator', 'nabia_webui/ai/servers/common.py', ('py', 'tool')),
    ('parse_as_of', 'nabia_webui/ai/servers/common.py', ('py', 'parse_as_of')),
    ('news_article', 'nabia_webui/ai/servers/news.py', ('py', 'NewsArticle')),
    ('store_news_articles', 'nabia_webui/ai/servers/news.py', ('py', 'build_server.store_news_articles')),
    ('fetch_news_feeds', 'nabia_webui/ai/servers/news.py', ('py', 'build_server.fetch_news_feeds')),
    ('news_summary', 'nabia_webui/ai/servers/news.py', ('py', 'build_server.news_summary')),
    ('set_risk_budgets', 'nabia_webui/ai/servers/portfolio.py', ('py', 'build_server.set_risk_budgets')),
    ('compute_weights', 'nabia_webui/ai/servers/portfolio.py', ('py', 'compute_weights')),
    ('news_system', 'nabia_webui/ai/agents/team.py', ('text', 'NEWS_SYSTEM')),
    ('as_of_rule', 'nabia_webui/ai/agents/team.py', ('text', 'AS_OF_RULE')),
    ('runtime_loop', 'nabia_webui/ai/agents/runtime.py', ('py', 'AgentRuntime._loop')),
    ('runtime_call_tool', 'nabia_webui/ai/agents/runtime.py', ('py', 'AgentRuntime._call_tool')),
    ('ollama_response', 'nabia_webui/ai/agents/ollama.py', ('py', 'OllamaClient._response')),
    ('decision_dates', 'nabia_webui/ai/pipeline.py', ('py', 'decision_dates')),
    ('evaluate_decisions', 'nabia_webui/ai/pipeline.py', ('py', 'evaluate_decisions')),
    ('job_log_handler', 'nabia_webui/web/jobs.py', ('py', 'JobLogHandler')),
    ('job_run', 'nabia_webui/web/jobs.py', ('py', 'JobRunner._run')),
    ('stats_system', 'nabia_webui/ai/agents/team.py', ('text', 'STATS_SYSTEM')),
    ('execution_system', 'nabia_webui/ai/agents/team.py', ('text', 'EXECUTION_SYSTEM')),
    ('backtest_note', 'nabia_webui/ai/agents/team.py', ('text', 'BACKTEST_NOTE')),
    ('team_calls', 'nabia_webui/ai/agents/team.py', ('between', '    async def collect_news', '            effort=self.config.effort.get("execution"')),
    ('team_fill', 'nabia_webui/ai/agents/team.py', ('py', 'AgentTeam._fill')),
    ('live_output', 'nabia_webui/ai/agents/runtime.py', ('py', 'LiveOutput')),
    ('runtime_create', 'nabia_webui/ai/agents/runtime.py', ('py', 'AgentRuntime._create')),
    ('ollama_stream', 'nabia_webui/ai/agents/ollama.py', ('py', '_Stream')),
]

#: Location-only references (\\coderef{name}): where a concept lives, without an excerpt.
REFS: list[tuple[str, str, tuple]] = [
    ('ai_config', 'nabia_webui/ai/config.py', ('py', 'AIConfig')),
    ('server_context', 'nabia_webui/ai/servers/common.py', ('py', 'ServerContext')),
    ('server_prices', 'nabia_webui/ai/servers/common.py', ('py', 'ServerContext.prices')),
    ('clean', 'nabia_webui/ai/servers/common.py', ('py', 'clean')),
    ('news_server', 'nabia_webui/ai/servers/news.py', ('py', 'build_server')),
    ('stats_server', 'nabia_webui/ai/servers/stats.py', ('py', 'build_server')),
    ('portfolio_server', 'nabia_webui/ai/servers/portfolio.py', ('py', 'build_server')),
    ('read_feed', 'nabia_webui/ai/servers/news.py', ('py', '_read_feed')),
    ('query_news', 'nabia_webui/ai/servers/news.py', ('py', 'build_server.query_news')),
    ('runtime_run', 'nabia_webui/ai/agents/runtime.py', ('py', 'AgentRuntime.run')),
    ('runtime_client', 'nabia_webui/ai/agents/runtime.py', ('py', 'AgentRuntime.client')),
    ('server_parameters', 'nabia_webui/ai/agents/runtime.py', ('py', 'AgentRuntime._server_parameters')),
    ('ollama_messages', 'nabia_webui/ai/agents/ollama.py', ('py', 'OllamaClient._messages')),
    ('ollama_chat', 'nabia_webui/ai/agents/ollama.py', ('py', 'OllamaClient._chat')),
    ('team_run', 'nabia_webui/ai/agents/team.py', ('py', 'AgentTeam._run')),
    ('team_universe', 'nabia_webui/ai/agents/team.py', ('py', 'AgentTeam._universe')),
    ('run_daily', 'nabia_webui/ai/pipeline.py', ('py', 'run_daily')),
    ('run_backtest', 'nabia_webui/ai/pipeline.py', ('py', 'run_backtest')),
    ('trend_rule_decider', 'nabia_webui/ai/pipeline.py', ('py', 'trend_rule_decider')),
    ('agent_decider', 'nabia_webui/ai/pipeline.py', ('py', 'agent_decider')),
    ('make_decider', 'nabia_webui/ai/pipeline.py', ('py', 'make_decider')),
    ('db_tables', 'nabia_webui/ai/db.py', ('between', 'news_table = Table(', 'backtest_runs_table = Table(')),
    ('store_news', 'nabia_webui/ai/db.py', ('py', 'AgentStore.store_news')),
    ('web_app', 'nabia_webui/web/app.py', ('py', 'create_app')),
    ('web_news', 'nabia_webui/web/app.py', ('py', 'create_app.news')),
    ('job_runner', 'nabia_webui/web/jobs.py', ('py', 'JobRunner')),
    ('job_backtest', 'nabia_webui/web/jobs.py', ('py', 'JobRunner._backtest')),
    ('log_viewer', 'nabia_webui/web/static/app.js', ('between', 'async function openJobLog', 'async function openJobLog')),
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
    out = ["% Generated by extract_snippets.py from nabia_webui/ai/servers/*.py"]
    for module in ("news", "stats", "portfolio"):
        path = f"nabia_webui/ai/servers/{module}.py"
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
    path = "nabia_webui/ai/config.py"
    tree = ast.parse((root / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AIConfig")
    rows, body = [], cls.body
    for i, node in enumerate(body):
        if isinstance(node, ast.AnnAssign):
            default = ast.unparse(node.value) if node.value is not None else ""
            default = re.sub(r"field\(default_factory=lambda: (.*)\)", r"\1", default)
            named = {"list(DEFAULT_SYMBOLS)": "the 9 default symbols (\\xref{Python}{Default AI portfolio universe})",
                     "dict(DEFAULT_PORTFOLIO)": "\\xref{C++}{Portfolio parameters}",
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
