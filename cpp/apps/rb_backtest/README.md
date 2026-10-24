# nabia_rb_backtest

Walk-forward backtest of a long-only **risk budgeting portfolio** from a file of
asset prices and a JSON file of portfolio parameters. It writes a CSV with the
weights, risk contributions, trading costs, returns, PnL, equity and drawdown at
every date, and prints a performance summary.

```sh
cmake --preset release && cmake --build --preset release --target nabia_rb_backtest
./build/release/apps/rb_backtest/nabia_rb_backtest \
    --prices  apps/rb_backtest/example/prices.csv \
    --config  apps/rb_backtest/example/portfolio.json \
    --output  result.csv \
    --metrics metrics.csv          # optional
```

On error (missing file, bad CSV line, invalid parameter) the program prints the
reason, naming the file and line or the config key, and exits with status 1.

## Inputs

### Prices (`--prices`)
A CSV file with a header row: `date`, then one column per asset. Dates are
`YYYY-MM-DD` in strictly increasing order, and prices must be positive numbers.

```
date,EQUITY,BONDS,COMMODITIES,GOLD
2020-01-02,100.0000,100.0000,100.0000,100.0000
2020-01-03,99.7020,100.1551,98.6550,100.0014
```

Quoted fields, Windows line endings and blank lines are accepted. A price can
be missing: an empty field, `NA`, `N/A`, `#N/A`, `NaN` or `null`, which is
typical when markets have different holidays. Only the dates where **every**
asset has a price are kept. The others are dropped, and the program reports how
many. Simple returns are then computed between consecutive kept dates (across
any gap), so the results start at the second kept date. Anything else that is
not a positive number is an error.

### Portfolio parameters (`--config`)
A JSON object. Every key is optional, and unknown keys are rejected, so typos are
caught instead of silently using a default.

| Key | Default | Meaning |
|---|---|---|
| `start`, `end` | whole file | first / last price date to use (`"YYYY-MM-DD"`, inclusive) |
| `covariance_window` | 60 | number of past return points used for each covariance estimate |
| `covariance_estimator` | `"sample"` | `"sample"` or `"exponential"` (recent points weigh more) |
| `ewma_halflife` | 20 | half-life in points for `"exponential"` |
| `rebalance_frequency` | 1 | trade every k points: 1 = every point, 5 = every fifth point, ... |
| `risk_measure` | `"volatility"` | `"volatility"`, `"value_at_risk"` or `"expected_shortfall"` (Gaussian) |
| `confidence_level` | 0.99 | for VaR / ES, in (0.5, 1) |
| `use_expected_returns` | false | VaR / ES: use the window's mean return instead of zero |
| `budgets` | equal | risk budgets, `{"ASSET": value, ...}` for every asset, or a list in column order; normalized to sum to 1 |
| `target_volatility` | none | annualized volatility to scale the portfolio to (e.g. 0.10); `null` for none |
| `max_leverage` | 2.0 | cap on the sum of weights when targeting volatility |
| `transaction_cost` | 0 | cost per unit of traded notional (0.0005 = 5 basis points) |
| `drift_weights` | true | weights drift with prices between trades (false: held constant) |
| `initial_capital` | 1 | starting equity, the unit of `pnl` and `equity` |
| `periods_per_year` | 252 | for annualized figures and volatility targeting |
| `solver_tolerance` | 1e-8 | risk budgeting accuracy, max \|contribution − budget\| |
| `solver_max_iter` | 5000 | solver iteration limit |
| `num_threads` | 0 | threads for the risk budgeting solves (0 = all cores) |

See `example/portfolio.json`.

## Output (`--output`)
One row per return date:

| Column | Meaning |
|---|---|
| `date` | end of the period |
| `weight_<asset>` | weight held during the period (fraction of equity; 1 − sum is cash) |
| `rebalance` | 1 if the portfolio traded at the close of this date |
| `leverage` | sum of the new target weights (rebalance rows only) |
| `risk_contribution_<asset>` | share of portfolio risk of each asset in the new weights (rebalance rows only; matches the budgets) |
| `turnover` | sum of absolute weight changes traded at the close |
| `transaction_cost` | cost paid at the close, as a fraction of equity |
| `gross_return` | portfolio return before costs, sum of weight × asset return |
| `net_return` | (1 + gross_return)(1 − transaction_cost) − 1 |
| `pnl` | change in equity, in currency |
| `equity` | capital at the end of the period |
| `drawdown` | 1 − equity / running peak |

Weights are decided with data up to the close of a date and earn returns from
the next date on, so there is no look-ahead. Rows before the first rebalance
(the first `covariance_window − 1` dates) hold no positions.

`--metrics` writes `metric,value` rows. They cover total and annualized return,
volatility, Sharpe and Sortino ratios, maximum drawdown and its duration,
Calmar ratio, hit rate, best and worst period, number of rebalances, turnover
and total costs, all computed from the first rebalance on.
