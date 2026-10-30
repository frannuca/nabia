"""Figures and tables of doc/news_risk_budgeting.tex from the database: python doc/figures.py [config.json]
(run by doc/build.sh). Writes doc/build/*.pdf and doc/build/*.tex."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datalayer import MarketDB, NewsDB, load_config  # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
out = Path(__file__).with_name("build")
out.mkdir(exist_ok=True)
tex = lambda s: s.replace("_", r"\_").replace("^", r"\textasciicircum{}").replace("&", r"\&")
factors, assets = list(cfg["factors"]), cfg["exposures"]["assets"]

with MarketDB(cfg["db"]) as db, NewsDB(cfg["db"]) as news:
    E = pd.DataFrame(db.exposures())
    H = pd.DataFrame([dict(r) for r in db.con.execute("SELECT as_of, asset, factor, exposure FROM asset_exposures")])
    W = pd.DataFrame([dict(r) for r in db.con.execute("SELECT * FROM portfolio_weights ORDER BY date")])
    stats = news.stats()

# exposure map at the latest estimation: heat map and table
m = E.pivot(index="factor", columns="asset", values="exposure").reindex(index=factors, columns=assets)
fig, ax = plt.subplots(figsize=(9, 8.5))
im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(assets)), assets, rotation=60, ha="right"), ax.set_yticks(range(len(factors)), factors)
for (i, j), v in np.ndenumerate(m.values):
    if abs(v) >= 0.05:
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.6, label="exposure $e_{a,f}$")
ax.set_title(f"Asset exposures to the factors, {E.as_of.iloc[0]}")
fig.tight_layout(), fig.savefig(out / "exposures.pdf"), plt.close(fig)

methods = E.groupby("factor").method.first().reindex(factors)
(out / "methods.tex").write_text("\n".join(f"{tex(f)} & {tex(cfg['factors'][f]['group'])} & {tex(cfg['factors'][f]['region'])} & "
                                           f"{methods[f]} & {tex(cfg['factors'][f]['positive'])} \\\\" for f in factors) + "\n")

# exposure history of a few pairs
pairs = [("ZN=F", "POLICY_FED"), ("^GSPC", "GROWTH_US"), ("BZ=F", "ENERGY_SUPPLY"), ("GC=F", "USD"), ("ZB=F", "FISCAL_US")]
fig, ax = plt.subplots(figsize=(9, 3.6))
for a, f in pairs:
    s = H[(H.asset == a) & (H.factor == f)].set_index("as_of").exposure
    ax.plot(pd.to_datetime(s.index), s.values, label=f"{a} / {f}")
ax.axhline(0, color="grey", lw=0.5), ax.set_ylabel("exposure"), ax.legend(fontsize=7, ncol=3), ax.grid(alpha=0.3)
ax.set_title("Exposure history (quarter ends)")
fig.tight_layout(), fig.savefig(out / "exposure_history.pdf"), plt.close(fig)

# weights over time and at the last rebalance
w = W.pivot(index="date", columns="asset", values="weight").reindex(columns=assets)
w.index = pd.to_datetime(w.index)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.stackplot(w.index, w.T.values, labels=assets)
ax.set_ylim(0, 1), ax.set_ylabel("weight"), ax.legend(fontsize=6, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.set_title("Risk budgeting weights at each rebalance")
fig.tight_layout(), fig.savefig(out / "weights.pdf"), plt.close(fig)

last = W[W.date == W.date.max()].set_index("asset").reindex(assets)
(out / "weights.tex").write_text("\n".join(f"\\code{{{a}}} & {r.budget:.4f} & {r.weight:.4f} & {r.risk_contribution:.4f} \\\\"
                                           for a, r in last.iterrows()) + "\n")
(out / "facts.tex").write_text(
    f"\\newcommand{{\\NewsCount}}{{{stats['news']:,}}}\n\\newcommand{{\\NewsFirst}}{{{stats['first'][:10]}}}\n"
    f"\\newcommand{{\\NewsLast}}{{{stats['last'][:10]}}}\n\\newcommand{{\\ScoredCount}}{{{sum(stats['scored'].values())}}}\n"
    f"\\newcommand{{\\ExposureDate}}{{{E.as_of.iloc[0]}}}\n\\newcommand{{\\ExposureDates}}{{{H.as_of.nunique()}}}\n"
    f"\\newcommand{{\\WeightDate}}{{{W.date.max()}}}\n\\newcommand{{\\Rebalances}}{{{W.date.nunique()}}}\n"
    f"\\newcommand{{\\PortfolioRisk}}{{{100 * last.portfolio_risk.iloc[0]:.2f}\\%}}\n")
print("figures and tables in", out)
