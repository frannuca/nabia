"""python -m unittest discover -s tests (from python/, with .venv): risk budgets from factor signals and exposures."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from budgets import asset_budgets, compute, factor_signals
from datalayer import MarketDB, NewsDB, load_config


def scores(*rows):
    return pd.DataFrame(rows, columns=["received_at", "factor", "sentiment", "confidence"]).assign(
        received_at=lambda d: pd.to_datetime(d.received_at))


class BudgetsTest(unittest.TestCase):
    def test_signal_decay_lag_and_evidence(self):
        s = scores(("2026-01-05 10:00", "USD", 1.0, 1.0))
        sig = factor_signals(s, pd.date_range("2026-01-05", "2026-01-11"), ["USD", "CNY"], half_life=1, evidence=1.0)
        self.assertEqual(sig.loc["2026-01-05", "USD"], 0.0)  # received that day: used from the next day only
        self.assertAlmostEqual(sig.loc["2026-01-06", "USD"], 1 * 1 / (1 + 1))  # level 1, evidence n/(n+1) with n = 1
        self.assertAlmostEqual(sig.loc["2026-01-07", "USD"], 0.5 / 1.5)  # n halves after one half-life
        self.assertTrue((sig["CNY"] == 0).all())

    def test_budgets_bounds_and_sign(self):
        cfg = load_config()
        e = pd.DataFrame({"POLICY_FED": [-0.9, 0.0, 0.5], "USD": [0.0, 0.0, -0.5]}, index=["BOND", "NONE", "GOLD"])
        out = asset_budgets(e, pd.Series({"POLICY_FED": 0.8, "USD": 0.4}), pd.Series(1 / 3, e.index), cfg["budgets"])
        self.assertAlmostEqual(out.budget.sum(), 1.0)
        self.assertAlmostEqual(out.score["BOND"], -0.8)  # hawkish Fed: bonds down
        self.assertAlmostEqual(out.score["GOLD"], (0.5 * 0.8 - 0.5 * 0.4) / 1.0)
        self.assertEqual(out.score["NONE"], 0.0)
        self.assertAlmostEqual(out.multiplier["BOND"], 1 - 0.5 * 0.8)
        self.assertLess(out.budget["BOND"], out.budget["NONE"])
        big = asset_budgets(e, pd.Series({"POLICY_FED": 10.0, "USD": 0.0}), pd.Series(1 / 3, e.index), cfg["budgets"])
        self.assertEqual(big.multiplier["BOND"], cfg["budgets"]["min_multiplier"])

    def test_compute_is_point_in_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config()
            cfg["db"] = str(Path(tmp) / "t.db")
            cfg["exposures"]["assets"] = ["BOND"]
            cfg["score"].update(model="stub", prompt="v")
            with MarketDB(cfg["db"]) as db:  # exposures estimated on 2026-01-07
                db.save_exposures([{"as_of": "2026-01-07", "asset": "BOND", "factor": "POLICY_FED", "exposure": -0.9,
                                    "raw": -0.9, "t_stat": -10.0, "stability": 1.0, "method": "proxy"}])
            with NewsDB(cfg["db"]) as db:
                db.add_news([{"headline": "Fed hikes", "source": "t", "received_at": "2026-01-06T12:00:00Z"}])
                db.save_scoring(1, "stub", "v", True, [("POLICY_FED", 1.0, 1.0)])
            rows = compute(cfg, "2026-01-02", "2026-01-16").set_index("date")
        self.assertNotIn("2026-01-02", rows.index)  # no exposures estimated yet
        self.assertEqual(rows.loc["2026-01-09", "exposures"], "2026-01-07")
        self.assertLess(rows.loc["2026-01-09", "score"], 0)
        self.assertAlmostEqual(rows.loc["2026-01-09", "budget"], 1.0)  # a single asset always has the whole budget


if __name__ == "__main__":
    unittest.main()


class WeightsTest(unittest.TestCase):
    def test_optimizer_matches_budgets(self):
        import numpy as np

        from budgets import covariance, optimize
        cfg = load_config()
        rng = np.random.default_rng(0)
        R = pd.DataFrame(rng.normal(0, [0.02, 0.01, 0.005], size=(104, 3)))
        cov = covariance(R, cfg["weights"]["shrinkage"])
        self.assertTrue(np.allclose(cov, cov.T) and (np.diag(cov) > 0).all())
        b = np.array([0.5, 0.3, 0.2])
        w, res = optimize(cov, b, cfg["weights"])
        self.assertTrue(res.converged)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue(np.allclose(res.relative_contributions, b, atol=1e-6))
        self.assertGreater(w[2], w[0])  # the least volatile asset gets the most capital per unit of budget


class VersionTest(unittest.TestCase):
    def test_only_one_version_is_read(self):
        """Scores of another prompt version are ignored, not averaged with the current ones."""
        from exposures.proxies import article_scores
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config()
            cfg["db"] = str(Path(tmp) / "t.db")
            with NewsDB(cfg["db"]) as db:
                db.add_news([{"headline": "War", "source": "t", "received_at": "2026-01-06T12:00:00Z"}])
                db.save_scoring(1, "m", "old", True, [("GEOPOLITICS", -0.6, 0.8)])
                db.save_scoring(1, "m", "new", True, [("GEOPOLITICS", 0.6, 0.8)])
                cfg["score"].update(model="m", prompt="new")
                rows = article_scores(db, cfg)
        self.assertEqual(rows.sentiment.tolist(), [0.6])
