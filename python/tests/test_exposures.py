"""python -m unittest discover -s tests (from python/, with .venv: numpy and pandas): the exposure estimator."""

import unittest

import numpy as np
import pandas as pd

from datalayer import load_config
from exposures import fit_window
from exposures.estimate import decide


class ExposuresTest(unittest.TestCase):
    def test_orthogonalized_fit(self):
        """y loads on the second proxy only through its part not explained by the first."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=2000)
        b = 0.8 * a + 0.6 * rng.normal(size=2000)  # correlated with a
        y = 1.0 * a + 0.5 * (b - 0.8 * a) + 0.3 * rng.normal(size=2000)
        exposure, t = fit_window(np.c_[a, b], y)
        self.assertGreater(exposure[0], 0.8)
        self.assertAlmostEqual(exposure[1] / exposure[0], 0.5 * 0.6 / 1.0, delta=0.05)  # residual of b has sd 0.6
        self.assertTrue((np.abs(t) > 10).all())
        noise, tn = fit_window(np.c_[a, b], rng.normal(size=2000))
        self.assertTrue((np.abs(tn) < 4).all())

    def test_decide(self):
        cfg = load_config()
        fits = {"POLICY_FED": (-0.9, -20.0, "proxy"), "USD": (0.05, 0.5, "proxy"), "GEOPOLITICS": (0.4, 3.0, "news")}
        rows = {r["factor"]: r for r in decide(cfg, pd.Timestamp("2026-09-30"), "ZN=F", fits, {})}
        self.assertAlmostEqual(rows["POLICY_FED"]["exposure"], -0.9, places=6)  # the only fit of its group: shrinking keeps it
        self.assertEqual(rows["USD"]["exposure"], 0.0)  # not significant
        self.assertEqual(rows["POLICY_ECB"]["method"], "group")
        self.assertAlmostEqual(rows["POLICY_ECB"]["exposure"], 0.5 * -0.9)
        self.assertEqual((rows["GEOPOLITICS"]["method"], rows["GEOPOLITICS"]["exposure"]), ("news", 0.4))
        self.assertEqual(rows["TRADE_POLICY"]["method"], "group")  # borrows from GEOPOLITICS (event group)
        self.assertEqual(len(rows), 27)


if __name__ == "__main__":
    unittest.main()


class OverrideTest(unittest.TestCase):
    def test_news_overrides_proxy(self):
        """An asset driven by Fed news that its rate proxy does not show: the news fit replaces the proxy fit."""
        import tempfile
        from pathlib import Path

        from datalayer import MarketDB, NewsDB
        from exposures import estimate

        rng = np.random.default_rng(1)
        days = pd.bdate_range("2018-01-01", "2023-12-31")
        weeks = pd.Series(1, days).resample("W-FRI").last().index
        shock = pd.Series(rng.normal(size=len(weeks)), weeks)  # weekly Fed news shock, hawkish > 0
        growth = rng.normal(0, 0.01, len(days))
        asset = growth + rng.normal(0, 0.003, len(days))
        asset[days.weekday == 4] -= 0.01 * shock.reindex(days[days.weekday == 4]).values  # hawkish news: asset falls
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config()
            cfg["db"] = str(Path(tmp) / "t.db")
            cfg["factors"] = {"GROWTH_US": {**cfg["factors"]["GROWTH_US"], "proxy": [["price:G", "logret", 1]]},
                              "POLICY_FED": {**cfg["factors"]["POLICY_FED"], "proxy": [["price:R", "logret", 1]]}}
            cfg["exposures"].update(assets=["A"], start="2018-01-01", news_min_weeks=52)
            cfg["score"].update(model="stub", prompt="v")
            with MarketDB(cfg["db"]) as db:
                db.con.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, adj_close REAL)")
                for sym, r in (("G", growth), ("R", rng.normal(0, 0.01, len(days))), ("A", asset)):
                    db.con.executemany("INSERT INTO prices VALUES (?, ?, ?, ?)",
                                       [(sym, str(d.date()), p, p) for d, p in zip(days, 100 * np.exp(np.cumsum(r)))])
                db.con.commit()
            with NewsDB(cfg["db"]) as db:
                for k, (week, s) in enumerate(shock.items()):
                    ts = f"{(week - pd.Timedelta(days=2)).date()}T12:00:00Z"
                    db.add_news([{"headline": f"Fed item {k}", "source": "test", "published_at": ts, "received_at": ts}])
                    db.save_scoring(k + 1, "stub", "v", True, [("POLICY_FED", float(np.tanh(s)), 1.0)])
            rows = estimate(cfg, download=False)
        last = rows[rows.as_of == rows.as_of.max()].set_index("factor")
        self.assertEqual(last.loc["POLICY_FED", "method"], "news")
        self.assertLess(last.loc["POLICY_FED", "exposure"], -0.2)
        self.assertEqual(last.loc["GROWTH_US", "method"], "proxy")
        self.assertGreater(last.loc["GROWTH_US", "exposure"], 0.5)
