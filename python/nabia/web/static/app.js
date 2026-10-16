"use strict";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const main = $("#main");
const charts = [];

function esc(value) {
  // Everything shown comes from the database (news from the web included): always escape.
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function safeUrl(url) {
  return /^https?:\/\//i.test(url || "") ? url : null;
}
const pct = (x, d = 1) => (x == null ? "–" : `${(100 * x).toFixed(d)}%`);
const num = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 4000);
}

function openModal(html) {
  $("#modal-body").innerHTML = html;
  $("#modal").hidden = false;
}
$("#modal").addEventListener("click", (e) => {
  if (e.target.id === "modal" || e.target.classList.contains("modal-close")) $("#modal").hidden = true;
});

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
const PALETTE = ["#2f6fd6", "#e07a2e", "#18865a", "#b8457a", "#7a5cc7", "#c99a12", "#2a9bb5", "#c2412d", "#5c6f7f", "#8a9a2b"];

function chart(canvas, config) {
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--border");
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  const c = new Chart(canvas, {
    ...config,
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { boxWidth: 12 } } },
      ...config.options,
    },
  });
  charts.push(c);
  return c;
}

function sentimentBadge(s) {
  const cls = s > 0.15 ? "good" : s < -0.15 ? "bad" : "plain";
  return `<span class="badge ${cls}">${s > 0 ? "+" : ""}${num(s, 2)}</span>`;
}
function statusBadge(status) {
  const cls = { done: "good", failed: "bad", running: "warn", queued: "plain" }[status] || "plain";
  return `<span class="badge ${cls}">${esc(status)}</span>`;
}

async function submitJob(kind, params) {
  const { id } = await api("/api/jobs", { method: "POST", body: JSON.stringify({ kind, params }) });
  toast(`Job #${id} (${kind}) started`);
  pollJobs();
  return id;
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
const pages = { overview, news, executions, backtests, jobs };

async function route() {
  const [page, arg] = (location.hash.slice(1) || "overview").split("/");
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${page}`));
  while (charts.length) charts.pop().destroy();
  main.innerHTML = `<div class="empty">Loading…</div>`;
  try {
    await (pages[page] || overview)(arg ? decodeURIComponent(arg) : undefined);
  } catch (error) {
    main.innerHTML = `<div class="panel error">${esc(error.message)}</div>`;
  }
}
window.addEventListener("hashchange", route);

// Running-job indicator in the top bar.
async function pollJobs() {
  try {
    const { jobs } = await api("/api/jobs?limit=20");
    const active = jobs.filter((j) => j.status === "running" || j.status === "queued");
    const el = $("#job-indicator");
    el.hidden = active.length === 0;
    el.textContent = active.length ? `● ${active.length} job${active.length > 1 ? "s" : ""} running: ${active[0].kind} ${active[0].message || ""}` : "";
    el.dataset.jobId = active.length ? active[0].id : "";
    el.title = "Show the log";
    clearTimeout(pollJobs.timer);
    if (active.length) pollJobs.timer = setTimeout(pollJobs, 3000);
    else if (pollJobs.wasActive && location.hash.startsWith("#jobs")) route();
    pollJobs.wasActive = active.length > 0;
  } catch { /* the indicator is best effort */ }
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------
async function overview() {
  const o = await api("/api/overview");
  const d = o.live_decision;
  const stale = (last) => !last || (Date.now() - new Date(last)) / 864e5 > 5;
  main.innerHTML = `
    <div class="page-head">
      <div><h1>Overview</h1><div class="muted">${o.config.symbols.length} assets · model ${esc(o.config.model)}</div></div>
      <div class="actions">
        <button class="secondary" data-job="download">Update market data</button>
        <button class="secondary" data-job="news">Collect news</button>
        <button data-job="daily">Run daily decision</button>
      </div>
    </div>
    <div class="grid three">
      <div class="panel"><div class="muted">Live decision</div><div class="stat">${d ? esc(d.as_of) : "none"}</div>
        <div class="muted">${d ? `leverage ${num(d.leverage)} · by ${esc(d.decided_by)}` : "run the daily decision"}</div></div>
      <div class="panel"><div class="muted">News, last 7 days</div><div class="stat">${o.news_last_7_days}</div>
        <div class="muted"><a href="#news">browse news</a></div></div>
      <div class="panel"><div class="muted">Runs with decisions</div><div class="stat">${o.runs.length}</div>
        <div class="muted"><a href="#backtests">backtests</a> · <a href="#executions">executions</a></div></div>
    </div>
    <div class="grid two" style="margin-top:16px">
      <div class="panel">
        <h2>Current risk budgets and weights</h2>
        ${d ? `<div class="chart small"><canvas id="live-chart"></canvas></div>
          <h3>Rationale</h3><div class="rationale">${esc(d.rationale)}</div>` : `<div class="empty">No live decision yet.</div>`}
      </div>
      <div class="panel">
        <h2>Market data</h2>
        <div class="table-wrap"><table>
          <tr><th>Symbol</th><th>Name</th><th>Class</th><th>Last price date</th><th class="num">Prices</th></tr>
          ${o.coverage.map((c) => `<tr><td class="mono">${esc(c.symbol)}</td><td>${esc(c.name || "")}</td><td>${esc(c.asset_class || "")}</td>
            <td>${esc(c.last_date || "–")} ${stale(c.last_date) ? '<span class="badge warn">stale</span>' : ""}</td><td class="num">${c.prices}</td></tr>`).join("")}
        </table></div>
        <h3>Recent jobs</h3>
        ${jobsTable(o.jobs)}
      </div>
    </div>`;
  main.querySelectorAll("[data-job]").forEach((b) =>
    b.addEventListener("click", async () => {
      b.disabled = true;
      try {
        await submitJob(b.dataset.job, b.dataset.job === "download" ? { start: "2010-01-01" } : {});
      } catch (e) { toast(e.message); }
      b.disabled = false;
    }));
  if (d) {
    const symbols = o.config.symbols;
    chart($("#live-chart"), {
      type: "bar",
      data: {
        labels: symbols,
        datasets: [
          { label: "Risk budget", data: symbols.map((s) => d.budgets[s]), backgroundColor: PALETTE[0] },
          { label: "Weight", data: symbols.map((s) => (d.weights ? d.weights[s] : null)), backgroundColor: PALETTE[1] },
        ],
      },
      options: { scales: { y: { ticks: { callback: (v) => pct(v, 0) } } } },
    });
  }
}

function jobsTable(list) {
  if (!list.length) return `<div class="empty">No jobs yet.</div>`;
  return `<div class="table-wrap"><table class="jobs-table">
    <tr><th>#</th><th>Kind</th><th>Status</th><th>Started</th><th>Message</th><th></th></tr>
    ${list.map((j) => `<tr class="clickable" data-job-id="${j.id}" title="Show the log"><td>${j.id}</td><td>${esc(j.kind)}</td>
      <td>${statusBadge(j.status)}</td><td class="muted">${esc(j.created_at)}</td>
      <td>${j.error ? `<span class="error">${esc(j.error)}</span>` : esc(j.message || "")}</td><td><a href="#" class="muted">log</a></td></tr>`).join("")}
  </table></div>`;
}

// Clicking a job row opens its live log.
document.addEventListener("click", (e) => {
  const row = e.target.closest(".jobs-table tr[data-job-id]");
  if (!row) return;
  e.preventDefault();
  openJobLog(Number(row.dataset.jobId));
});

// ---------------------------------------------------------------------------
// Live job log
// ---------------------------------------------------------------------------
let logSession = 0;

async function openJobLog(jobId) {
  const session = ++logSession;
  openModal(`<div class="log-head"><h2>Job #${jobId} <span id="log-status"></span></h2>
      <label class="log-follow"><input type="checkbox" id="log-follow" checked> follow</label></div>
    <div class="muted" id="log-summary"></div>
    <div class="log-view" id="log-view"><div class="muted">Loading…</div></div>`);
  const view = $("#log-view");
  let after = 0;
  let first = true;
  const open = () => !$("#modal").hidden && session === logSession;
  view.addEventListener("scroll", () => {
    // Stop following when the user scrolls up; resume at the bottom.
    $("#log-follow").checked = view.scrollTop + view.clientHeight >= view.scrollHeight - 30;
  });
  while (open()) {
    let data;
    try {
      data = await api(`/api/jobs/${jobId}/logs?after=${after}`);
    } catch (error) {
      view.innerHTML = `<div class="error">${esc(error.message)}</div>`;
      return;
    }
    if (!open()) return;
    const j = data.job;
    $("#log-status").innerHTML = statusBadge(j.status);
    $("#log-summary").innerHTML = `${esc(j.kind)} ${esc(JSON.stringify(j.params))} · started ${esc(j.created_at)}`
      + (j.finished_at ? ` · finished ${esc(j.finished_at)}` : "")
      + (j.error ? `<div class="error">${esc(j.error)}</div>` : j.message ? `<div>${esc(j.message)}</div>` : "");
    if (first) {
      view.innerHTML = data.logs.length ? "" : `<div class="muted log-empty">No log lines yet.</div>`;
      first = false;
    }
    if (data.logs.length) {
      const empty = view.querySelector(".log-empty");
      if (empty) empty.remove();
      // Streamed model output (logger nabia.ai.agents.model) gets its own style: thinking vs written text.
      const kind = (l) => l.logger.endsWith(".model") ? (/^\S+ thinking:/.test(l.message) ? " model thinking" : " model") : "";
      view.insertAdjacentHTML("beforeend", data.logs.map((l) => `<div class="log-line ${esc(l.level.toLowerCase())}${kind(l)}">`
        + `<span class="log-ts">${esc(l.ts.slice(11))}</span><span class="log-src">${esc(l.logger.split(".").pop())}</span>`
        + `<span class="log-msg">${esc(l.message)}</span></div>`).join(""));
      after = data.logs[data.logs.length - 1].id;
      if ($("#log-follow").checked) view.scrollTop = view.scrollHeight;
    }
    const active = j.status === "running" || j.status === "queued";
    if (!active && data.logs.length < 1000) return;  // finished and fully loaded
    await new Promise((r) => setTimeout(r, active ? 1000 : 0));
  }
}

// ---------------------------------------------------------------------------
// News
// ---------------------------------------------------------------------------
const TOPICS = ["monetary_policy", "inflation", "growth", "labor", "earnings", "fiscal", "geopolitics", "energy", "metals",
  "agriculture", "crypto", "rates", "credit", "fx", "regulation", "other"];
const CLASSES = ["equity_index", "bond_future", "commodity", "crypto"];

async function news() {
  const today = new Date().toISOString().slice(0, 10);
  const monthAgo = new Date(Date.now() - 30 * 864e5).toISOString().slice(0, 10);
  const { config } = await api("/api/overview");
  main.innerHTML = `
    <div class="page-head"><div><h1>News</h1><div class="muted">Collected by the news analyzer</div></div>
      <button class="secondary" id="collect">Collect today's news</button></div>
    <div class="panel"><form class="filters" id="news-filters">
      <label>From <input type="date" name="start" value="${monthAgo}"></label>
      <label>To <input type="date" name="end" value="${today}"></label>
      <label>Topic <select name="topic"><option value="">all</option>${TOPICS.map((t) => `<option>${t}</option>`).join("")}</select></label>
      <label>Asset class <select name="asset_class"><option value="">all</option>${CLASSES.map((t) => `<option>${t}</option>`).join("")}</select></label>
      <label>Symbol <select name="symbol"><option value="">all</option>${config.symbols.map((s) => `<option>${esc(s)}</option>`).join("")}</select></label>
      <label>Min relevance <input type="number" name="min_relevance" min="0" max="1" step="0.1" value="0" style="width:90px"></label>
      <label>Search <input type="search" name="q" placeholder="title or summary"></label>
      <button>Apply</button>
    </form></div>
    <div id="news-body"></div>`;
  $("#collect").addEventListener("click", () => submitJob("news", {}).catch((e) => toast(e.message)));
  const form = $("#news-filters");
  const load = async () => {
    const params = new URLSearchParams([...new FormData(form)].filter(([, v]) => v !== ""));
    const data = await api(`/api/news?${params}`);
    while (charts.length) charts.pop().destroy();
    $("#news-body").innerHTML = data.total === 0 ? `<div class="panel empty" style="margin-top:16px">No articles for these filters.</div>` : `
      <div class="grid two" style="margin-top:16px">
        <div class="panel"><h2>Articles per day</h2><div class="chart small"><canvas id="per-day"></canvas></div></div>
        <div class="panel"><h2>Average sentiment by topic</h2><div class="chart small"><canvas id="by-topic"></canvas></div></div>
      </div>
      <div class="panel" style="margin-top:16px"><h2>${data.total} articles</h2><div class="table-wrap"><table id="news-table">
        <tr><th>Date</th><th>Title</th><th>Topic</th><th>Assets</th><th class="num">Sentiment</th><th class="num">Relevance</th></tr>
        ${data.articles.map((a, i) => `<tr class="clickable" data-i="${i}"><td class="muted">${esc(a.published_date)}</td>
          <td>${esc(a.title)}<div class="muted">${esc(a.source)}</div></td><td><span class="badge plain">${esc(a.topic)}</span></td>
          <td><div class="chips">${a.symbols.map((s) => `<span class="badge">${esc(s)}</span>`).join("") || a.asset_classes.map((c) => `<span class="badge plain">${esc(c)}</span>`).join("")}</div></td>
          <td class="num">${sentimentBadge(a.sentiment)}</td><td class="num">${num(a.relevance, 2)}</td></tr>`).join("")}
      </table></div></div>`;
    if (data.total === 0) return;
    chart($("#per-day"), { type: "bar", data: { labels: Object.keys(data.per_day), datasets: [{ label: "Articles", data: Object.values(data.per_day), backgroundColor: PALETTE[0] }] },
      options: { plugins: { legend: { display: false } } } });
    const topics = Object.keys(data.by_topic);
    chart($("#by-topic"), { type: "bar", data: { labels: topics, datasets: [{ label: "Sentiment",
      data: topics.map((t) => data.by_topic[t].avg_sentiment),
      backgroundColor: topics.map((t) => (data.by_topic[t].avg_sentiment >= 0 ? css("--good") : css("--bad"))) }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: -1, max: 1 } } } });
    $("#news-table").addEventListener("click", (e) => {
      const row = e.target.closest("tr[data-i]");
      if (!row) return;
      const a = data.articles[Number(row.dataset.i)];
      const url = safeUrl(a.url);
      openModal(`<h2>${esc(a.title)}</h2>
        <div class="muted">${esc(a.source)} · ${esc(a.published_date)} · ${esc(a.topic)} · sentiment ${num(a.sentiment)} · relevance ${num(a.relevance)}</div>
        <p>${esc(a.summary)}</p>
        <div class="chips">${a.asset_classes.map((c) => `<span class="badge plain">${esc(c)}</span>`).join("")} ${a.symbols.map((s) => `<span class="badge">${esc(s)}</span>`).join("")}</div>
        ${url ? `<p><a href="${esc(url)}" target="_blank" rel="noopener noreferrer">Open article ↗</a></p>` : ""}`);
    });
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); load().catch((err) => toast(err.message)); });
  await load();
}

// ---------------------------------------------------------------------------
// Executions: decisions and agent runs per run
// ---------------------------------------------------------------------------
async function executions(selected) {
  const { runs } = await api("/api/runs");
  const run = selected || (runs.find((r) => r.run === "live") || runs[0] || {}).run;
  if (!run) {
    main.innerHTML = `<h1>Executions</h1><div class="panel empty">No decisions yet: run the daily decision or a backtest.</div>`;
    return;
  }
  const [{ decisions, symbols }, { agent_runs }] = await Promise.all([
    api(`/api/runs/${encodeURIComponent(run)}/decisions`),
    api(`/api/agent_runs?run=${encodeURIComponent(run)}&limit=300`),
  ]);
  main.innerHTML = `
    <div class="page-head"><div><h1>Executions</h1><div class="muted">Risk budget decisions and the agent runs behind them</div></div>
      <label>Run <select id="run-select">${runs.map((r) => `<option ${r.run === run ? "selected" : ""}>${esc(r.run)}</option>`).join("")}</select></label></div>
    <div class="panel"><h2>Risk budgets over time</h2><div class="chart"><canvas id="budgets-chart"></canvas></div></div>
    <div class="grid two" style="margin-top:16px">
      <div class="panel"><h2>${decisions.length} decisions</h2><div class="table-wrap"><table id="decisions-table">
        <tr><th>Date</th><th>By</th><th class="num">Leverage</th><th>Largest budgets</th></tr>
        ${decisions.slice().reverse().map((d) => {
          const top = Object.entries(d.budgets).sort((a, b) => b[1] - a[1]).slice(0, 3);
          return `<tr class="clickable" data-as-of="${esc(d.as_of)}"><td>${esc(d.as_of)}</td><td><span class="badge plain">${esc(d.decided_by)}</span></td>
            <td class="num">${num(d.leverage)}</td><td>${top.map(([s, b]) => `${esc(s)} ${pct(b, 0)}`).join(", ")}</td></tr>`;
        }).join("")}
      </table></div></div>
      <div class="panel" id="decision-detail"><div class="empty">Select a decision to see its budgets, weights, rationale and agent runs.</div></div>
    </div>`;
  $("#run-select").addEventListener("change", (e) => (location.hash = `#executions/${encodeURIComponent(e.target.value)}`));
  chart($("#budgets-chart"), {
    type: "line",
    data: { labels: decisions.map((d) => d.as_of),
      datasets: symbols.map((s, i) => ({ label: s, data: decisions.map((d) => d.budgets[s]), borderColor: PALETTE[i % 10],
        backgroundColor: PALETTE[i % 10], fill: i === 0 ? "origin" : "-1", stepped: true, pointRadius: 0, borderWidth: 1 })) },
    options: { scales: { y: { stacked: true, max: 1, ticks: { callback: (v) => pct(v, 0) } } } },
  });
  $("#decisions-table").addEventListener("click", async (e) => {
    const row = e.target.closest("tr[data-as-of]");
    if (!row) return;
    document.querySelectorAll("#decisions-table tr").forEach((r) => r.classList.toggle("selected", r === row));
    const d = decisions.find((x) => x.as_of === row.dataset.asOf);
    const { report } = await api(`/api/runs/${encodeURIComponent(run)}/stats_report?as_of=${d.as_of}`);
    const runsForDate = agent_runs.filter((a) => a.as_of === d.as_of);
    $("#decision-detail").innerHTML = `
      <h2>Decision ${esc(d.as_of)}</h2>
      <table><tr><th>Asset</th><th class="num">Budget</th><th class="num">Weight</th></tr>
        ${symbols.map((s) => `<tr><td class="mono">${esc(s)}</td><td class="num">${pct(d.budgets[s])}</td><td class="num">${d.weights ? pct(d.weights[s]) : "–"}</td></tr>`).join("")}
      </table>
      <h3>Rationale</h3><div class="rationale">${esc(d.rationale)}</div>
      ${report && report.as_of === d.as_of ? `<h3>Stats report</h3><p>${esc(report.summary)}</p><p class="muted">Regime: ${esc(report.regime)}</p>
        <table><tr><th>Asset</th><th>Trend</th><th>Volatility</th><th class="num">Tilt</th></tr>
        ${report.assets.map((a) => `<tr title="${esc(a.reason)}"><td class="mono">${esc(a.symbol)}</td><td>${esc(a.trend)}</td><td>${esc(a.volatility_regime)}</td><td class="num">${num(a.budget_tilt)}</td></tr>`).join("")}</table>` : ""}
      <h3>Agent runs</h3>
      ${runsForDate.length ? agentRunsTable(runsForDate) : `<div class="muted">No agent runs recorded for this date (rule-based decision or earlier run).</div>`}`;
    bindAgentRuns($("#decision-detail"));
  });
}

function agentRunsTable(list) {
  return `<div class="table-wrap"><table>
    <tr><th>Agent</th><th>Date</th><th class="num">Turns</th><th class="num">Tools</th><th class="num">Web</th><th class="num">Tokens out</th><th></th></tr>
    ${list.map((a) => `<tr class="clickable" data-agent-run="${a.id}"><td>${esc(a.agent)}</td><td>${esc(a.as_of)}</td>
      <td class="num">${a.turns}</td><td class="num">${a.tool_call_count}${a.tool_errors ? ` <span class="badge bad">${a.tool_errors} err</span>` : ""}</td>
      <td class="num">${a.server_tool_calls}</td><td class="num">${a.usage.output_tokens ?? "–"}</td>
      <td>${a.error ? '<span class="badge bad">failed</span>' : ""}</td></tr>`).join("")}
  </table></div>`;
}

function bindAgentRuns(root) {
  root.querySelectorAll("[data-agent-run]").forEach((row) => row.addEventListener("click", async () => {
    const a = await api(`/api/agent_runs/${row.dataset.agentRun}`);
    openModal(`<h2>${esc(a.agent)} · ${esc(a.as_of)} <span class="muted">(${esc(a.run)})</span></h2>
      <div class="muted">${esc(a.started_at)} → ${esc(a.finished_at)} · ${a.turns} turns · ${a.server_tool_calls} web calls ·
        tokens in ${a.usage.input_tokens ?? 0} (cached ${a.usage.cache_read_input_tokens ?? 0}) / out ${a.usage.output_tokens ?? 0}</div>
      ${a.error ? `<pre class="error">${esc(a.error)}</pre>` : ""}
      <h3>Tool calls</h3>
      ${a.tool_calls.map((c, i) => `<details><summary>${i + 1}. <span class="mono">${esc(c.name)}</span> ${c.is_error ? '<span class="badge bad">error</span>' : ""}</summary>
        <div class="muted">input</div><pre>${esc(JSON.stringify(c.input, null, 2))}</pre><div class="muted">output</div><pre>${esc(c.output)}</pre></details>`).join("") || '<div class="muted">none</div>'}
      <h3>Final answer</h3><div class="rationale">${esc(a.final_text || "")}</div>`);
  }));
}

// ---------------------------------------------------------------------------
// Backtests
// ---------------------------------------------------------------------------
async function backtests(selected) {
  const [{ runs }, { config }] = await Promise.all([api("/api/runs"), api("/api/overview")]);
  const backtestRuns = runs.filter((r) => r.run !== "live");
  main.innerHTML = `
    <div class="page-head"><div><h1>Backtests</h1><div class="muted">Risk budgets decided over time, against equal risk budgets</div></div></div>
    <div class="panel"><h2>New backtest</h2>
      <form class="inline" id="bt-form">
        <label>Run name <input name="run" required pattern="[A-Za-z0-9_\\-]+" placeholder="agents-2024"></label>
        <label>Policy <select name="policy">
          <option value="trend_rule">trend rule (no model)</option>
          <option value="ollama">agents, local Ollama (${esc(config.ollama_model)})</option>
          <option value="claude">agents, Claude (${esc(config.claude_model)}, billed)</option>
        </select></label>
        <label>Start <input type="date" name="start" required value="2023-01-01"></label>
        <label>End <input type="date" name="end" required value="${new Date().toISOString().slice(0, 10)}"></label>
        <label>Decide every <select name="frequency"><option value="ME">month end</option><option value="QE">quarter end</option><option value="W-FRI">week (Friday)</option><option value="D">trading day</option></select></label>
        <button>Start</button>
      </form>
      <div class="muted" style="margin-top:8px">Agent policies run the stats and execution agents at each decision date (two runs per date): free but slow on the local model, billed on Claude. Re-using a run name resumes it.</div>
    </div>
    <div class="panel" style="margin-top:16px"><h2>Runs</h2>${backtestRuns.length ? `<div class="table-wrap"><table id="bt-table">
      <tr><th>Run</th><th>Policy</th><th>Period</th><th class="num">Decisions</th><th class="num">Sharpe</th><th class="num">vs equal risk</th><th class="num">Max drawdown</th></tr>
      ${backtestRuns.map((r) => { const m = r.backtest && r.backtest.metrics; return `<tr class="clickable ${r.run === selected ? "selected" : ""}" data-run="${esc(r.run)}">
        <td>${esc(r.run)}</td><td>${esc(r.backtest ? r.backtest.policy : "?")}</td>
        <td class="muted">${esc(r.backtest ? `${r.backtest.start} → ${r.backtest.end}` : `${r.first} → ${r.last}`)}</td><td class="num">${r.decisions}</td>
        <td class="num">${m ? num(m.decided.sharpe_ratio) : "–"}</td><td class="num">${m ? num(m.equal_risk.sharpe_ratio) : "–"}</td>
        <td class="num">${m ? pct(m.decided.max_drawdown) : "–"}</td></tr>`; }).join("")}
    </table></div>` : `<div class="empty">No backtests yet.</div>`}</div>
    <div id="bt-detail"></div>`;
  $("#bt-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const params = Object.fromEntries(new FormData(e.target));
    if (params.policy !== "trend_rule") {
      const model = params.policy === "ollama" ? `the local model ${config.ollama_model}` : `Claude (${config.claude_model}), billed per call`;
      const daily = params.frequency === "D" ? "\n\nDaily decisions mean two agent runs per trading day (about 500 a year)." : "";
      if (!confirm(`Run the agents on ${model} at every decision date?${daily}`)) return;
    }
    try { await submitJob("backtest", params); location.hash = "#jobs"; } catch (err) { toast(err.message); }
  });
  const table = $("#bt-table");
  if (table) table.addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-run]");
    if (row) location.hash = `#backtests/${encodeURIComponent(row.dataset.run)}`;
  });
  if (selected) await backtestDetail(selected);
}

async function backtestDetail(run) {
  const bt = await api(`/api/backtests/${encodeURIComponent(run)}`);
  const m = bt.metrics;
  const rows = [["Annualized return", "annualized_return", pct], ["Annualized volatility", "annualized_volatility", pct],
    ["Sharpe ratio", "sharpe_ratio", num], ["Sortino ratio", "sortino_ratio", num], ["Max drawdown", "max_drawdown", pct],
    ["Calmar ratio", "calmar_ratio", num], ["Annualized turnover", "annualized_turnover", num], ["Total costs", "total_costs", (x) => pct(x, 2)]];
  $("#bt-detail").innerHTML = `
    <div class="grid two" style="margin-top:16px">
      <div class="panel"><h2>${esc(run)}: equity</h2><div class="chart"><canvas id="equity"></canvas></div></div>
      <div class="panel"><h2>Metrics</h2><table><tr><th></th><th class="num">Decided budgets</th><th class="num">Equal risk</th></tr>
        ${rows.map(([label, key, f]) => `<tr><td>${label}</td><td class="num">${f(m.decided[key])}</td><td class="num">${f(m.equal_risk[key])}</td></tr>`).join("")}
      </table><div class="muted" style="margin-top:8px">${esc(bt.start)} → ${esc(bt.end)} · ${bt.schedule.dates.length} decisions</div></div>
      <div class="panel"><h2>Drawdown</h2><div class="chart small"><canvas id="drawdown"></canvas></div></div>
      <div class="panel"><h2>Risk budgets decided</h2><div class="chart small"><canvas id="schedule"></canvas></div></div>
    </div>
    <div class="panel" style="margin-top:16px"><h2>Weights (weekly)</h2><div class="chart"><canvas id="weights"></canvas></div></div>`;
  const line = (label, data, color, dashed) => ({ label, data, borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 1.5, borderDash: dashed ? [4, 3] : [] });
  chart($("#equity"), { type: "line", data: { labels: bt.dates, datasets: [line("Decided budgets", bt.equity, PALETTE[0]), line("Equal risk", bt.baseline_equity, css("--muted"), true)] },
    options: { scales: { x: { ticks: { maxTicksLimit: 8 } } } } });
  chart($("#drawdown"), { type: "line", data: { labels: bt.dates, datasets: [line("Decided budgets", bt.drawdown.map((x) => -x), css("--bad")), line("Equal risk", bt.baseline_drawdown.map((x) => -x), css("--muted"), true)] },
    options: { scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { ticks: { callback: (v) => pct(v, 0) } } } } });
  chart($("#schedule"), { type: "bar", data: { labels: bt.schedule.dates,
    datasets: bt.symbols.map((s, i) => ({ label: s, data: bt.schedule.budgets[s], backgroundColor: PALETTE[i % 10] })) },
    options: { scales: { x: { stacked: true, ticks: { maxTicksLimit: 8 } }, y: { stacked: true, max: 1, ticks: { callback: (v) => pct(v, 0) } } } } });
  chart($("#weights"), { type: "line", data: { labels: bt.weights.dates,
    datasets: bt.symbols.map((s, i) => ({ ...line(s, bt.weights.values[s], PALETTE[i % 10]), fill: i === 0 ? "origin" : "-1" })) },
    options: { scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { stacked: true, ticks: { callback: (v) => pct(v, 0) } } } } });
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------
async function jobs() {
  const { jobs: list } = await api("/api/jobs?limit=100");
  main.innerHTML = `
    <div class="page-head"><div><h1>Jobs</h1><div class="muted">Downloads, agent runs and backtests run one at a time in the background</div></div>
      <div class="actions"><button class="secondary" id="refresh">Refresh</button></div></div>
    <div class="grid three">
      <div class="panel"><h2>Market data</h2><form class="inline" id="dl-form">
        <label>From <input type="date" name="start" value="2010-01-01"></label>
        <label>Classes <select name="asset_classes"><option value="">all</option>${CLASSES.map((c) => `<option>${c}</option>`).join("")}</select></label>
        <button>Download</button></form></div>
      <div class="panel"><h2>News</h2><p class="muted">News analyzer: searches the web and stores today's articles.</p><button id="news-job">Collect news</button></div>
      <div class="panel"><h2>Daily decision</h2><form class="inline" id="daily-form">
        <label>As of <input type="date" name="as_of" value="${new Date().toISOString().slice(0, 10)}"></label>
        <label><span>Collect news first</span><input type="checkbox" name="collect_news" checked></label>
        <button>Run</button></form></div>
    </div>
    <div class="panel" style="margin-top:16px"><h2>History</h2>${jobsTable(list)}</div>`;
  $("#refresh").addEventListener("click", route);
  $("#dl-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    submitJob("download", { start: f.start, asset_classes: f.asset_classes ? [f.asset_classes] : null }).then(route).catch((err) => toast(err.message));
  });
  $("#news-job").addEventListener("click", () => submitJob("news", {}).then(route).catch((err) => toast(err.message)));
  $("#daily-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    submitJob("daily", { as_of: f.get("as_of"), collect_news: f.get("collect_news") === "on" }).then(route).catch((err) => toast(err.message));
  });
  pollJobs();
}

$("#job-indicator").addEventListener("click", (e) => {
  if (e.currentTarget.dataset.jobId) openJobLog(Number(e.currentTarget.dataset.jobId));
});

route();
pollJobs();
