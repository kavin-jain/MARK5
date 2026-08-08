# MARK6 — Overfitting & Statistical-Significance Analysis

Bailey & López de Prado tests on the DEPLOYED v7.5 config (momentum-heavy / n_hold=20 / tilt=1.5 / 126-bar refresh, rank-transformed scores, sector cap enforced, FY-netting tax), using every strategy variant explored across the project as the trial set (factor-weight grid, rebalance frequencies, asymmetric exits, TLH, FIP, sleeve frequencies). All on daily returns, 2016-2026, universe `data/pit_cache`.

## Deflated Sharpe Ratio (is the Sharpe real, given how many we tried?)

- Strategy variants tried (N): **124**
- Observed Sharpe: **1.05** annualised (0.066 daily)
- Probabilistic Sharpe Ratio vs 0 (P true SR>0): **99.9%**
- Expected max Sharpe from pure luck across 124 trials: 0.32 annualised
- **Deflated Sharpe Ratio (P skill survives multiple-testing): 98.9%**

## Probability of Backtest Overfitting (PBO via CSCV)

- Strategies in matrix: 70 | train/test combos: 924
- **PBO: 59.6%** (fraction of splits where the in-sample-best strategy lands below the out-of-sample median)
- Median performance-degradation logit: -0.25 (negative = overfit)

## Verdict

- DSR PASS: deflated-Sharpe 99% — the Sharpe survives multiple-testing; >95% confidence it is skill, not the luckiest draw.
- PBO WEAK: 60% — elevated overfitting risk.

These are the statistics professional quant funds use to vet a strategy before risking capital — most retail/student backtests never compute them.