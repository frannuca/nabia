"""Report tables (benchmark comparison, IC, LLM-feature IC, snapshot, selection) and CSV outputs."""

from __future__ import annotations

import pandas as pd


def tables(cfg, X, y, S, s, betas, bench, overlay, sched, b0):
    """The report tables: comparison, IC (S and s, per asset and pooled), LLM-feature IC, snapshot, selection."""
    assets = list(cfg["assets"])
    keys = ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "annualized_turnover", "total_costs"]
    comparison = pd.DataFrame({"ERC (b0)": bench.metrics, "sentiment overlay": overlay.metrics}).loc[keys]
    yy = pd.DataFrame(y).loc[S.index]
    ic = pd.DataFrame({"IC(S, y)": [S[a].corr(yy[a]) for a in assets], "IC(s, y)": [s[a].corr(yy[a]) for a in assets]}, assets)
    ic.loc["pooled"] = [S.stack().corr(yy.stack()), s.stack().corr(yy.stack())]
    llm_ic = pd.Series({k: pd.concat([X[a].get(f"llm_{k}@{cfg['assets'][a][0]}", pd.Series(dtype=float)).loc[S.index]
                                      for a in assets]).corr(pd.concat([yy[a] for a in assets]))
                        for k in cfg["llm_feature_signs"]}, name="pooled IC (own L3 node)")
    last = sched.index[-1]
    beta_last = {a: betas[max(k for k in betas if k[1] == a and k[0] <= last)] for a in assets}
    top = {a: ", ".join(f"{k} {v:+.2f}" for k, v in (X[a].loc[last].fillna(0) * beta_last[a]).nlargest(3).items() if v)
           for a in assets}
    snapshot = pd.DataFrame({"s": s.loc[last], "b0": b0, "b": sched.loc[last], "w": overlay.target_weights.loc[last],
                             "RC": overlay.risk_contributions.loc[last], "top features (beta*z)": pd.Series(top)})
    selected = pd.DataFrame({a: (pd.DataFrame({k[0]: v for k, v in betas.items() if k[1] == a}).T != 0).mean()
                             for a in assets}).fillna(0)
    return {"comparison": comparison, "ic": ic, "llm_ic": llm_ic, "snapshot": snapshot, "selection_frequency": selected}


def show_tables(t, last):
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print("\n== benchmark comparison ==\n", t["comparison"].round(4))
    print("\n== information coefficients ==\n", t["ic"].round(3), "\n\n", t["llm_ic"].round(3))
    print(f"\n== last rebalance {last.date()} ==\n", t["snapshot"].round(3))


def save(out, t, sched, s, news, bench, overlay, ics):
    for name, table in {**t, "budgets": sched, "signal": s}.items():
        table.to_csv(out / f"{name}.csv")
    news.drop(columns=[c for c in news if c.startswith("_")]).to_csv(out / "news.csv", index=False)
    pd.DataFrame({"ERC (b0)": bench.performance.equity, "sentiment overlay": overlay.performance.equity}).to_csv(out / "equity.csv")
    ics.to_csv(out / "ic_in_out_of_sample.csv", index=False)
