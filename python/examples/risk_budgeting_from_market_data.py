"""End to end: download market data into SQLite, backtest a risk budgeting
portfolio on it and print the weights to hold now.

    PYTHONPATH=python python python/examples/risk_budgeting_from_market_data.py --db sqlite:///market.db
"""

from __future__ import annotations

import argparse
import datetime as dt

import nabia
from nabia.marketdata import MarketDataStore, YahooFinanceSource, find_assets, update_market_data

# One or two liquid instruments per asset class. Brent rather than WTI: the WTI
# front-month future (CL=F) settled at a negative price on 2020-04-20, where
# simple returns are undefined.
SYMBOLS = ["^GSPC", "^STOXX50E", "^N225", "ZN=F", "ZB=F", "BZ=F", "GC=F", "HG=F"]

PARAMS = {
    "covariance_window": 120,          # ~6 months of daily returns
    "covariance_estimator": "exponential",
    "ewma_halflife": 40,
    "rebalance_frequency": 21,         # monthly
    "risk_measure": "volatility",      # equal risk contribution (no budgets given)
    "target_volatility": 0.10,
    "max_leverage": 3.0,
    "transaction_cost": 0.0005,        # 5 bp of traded notional
    "initial_capital": 1_000_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="sqlite:///market.db")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2015, 1, 1))
    parser.add_argument("--output", default="risk_budgeting_backtest.csv")
    parser.add_argument("--no-download", action="store_true", help="use what is already in the database")
    args = parser.parse_args()

    store = MarketDataStore(args.db)
    if not args.no_download:
        written = update_market_data(store, YahooFinanceSource(), find_assets(SYMBOLS), start=args.start)
        print(f"downloaded {sum(written.values())} rows")

    prices = store.load_prices(SYMBOLS, start=args.start)
    result = nabia.backtest(prices, PARAMS)
    m = result.metrics
    print(f"\n{len(prices)} price dates, {result.dropped_dates} dropped (missing for some asset)")
    print(f"{result.weights.index[0].date()} .. {result.weights.index[-1].date()}, {m['rebalances']} rebalances")
    print(f"annualized return {m['annualized_return']:.2%}  volatility {m['annualized_volatility']:.2%}  "
          f"Sharpe {m['sharpe_ratio']:.2f}  max drawdown {m['max_drawdown']:.2%}  costs {m['total_costs']:.3%}")
    print(f"final equity {result.equity.iloc[-1]:,.0f}")

    result.weights.join(result.performance).to_csv(args.output)
    print(f"daily weights and PnL written to {args.output}")

    target = nabia.latest_weights(prices, PARAMS)
    print(f"\ntarget weights from {target.date.date()} (leverage {target.leverage:.2f}):")
    print(target.weights.to_frame().join(target.risk_contributions).to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
