# MARK6 — Overfitting & Statistical-Significance Analysis

Bailey & López de Prado tests on the deployed config (blend / n_hold=12 / tilt=1.5), using the grid of strategy variants we explored as the trial set. All on daily returns, 2016-2026.

## Deflated Sharpe Ratio (is the Sharpe real, given how many we tried?)

- Strategy variants tried (N): **60**
- Observed Sharpe: **0.90** annualised (0.056 daily)
- Probabilistic Sharpe Ratio vs 0 (P true SR>0): **99.7%**
- Expected max Sharpe from pure luck across 60 trials: 0.15 annualised
- **Deflated Sharpe Ratio (P skill survives multiple-testing): 99.0%**

## Probability of Backtest Overfitting (PBO via CSCV)

- Strategies in matrix: 60 | train/test combos: 924
- **PBO: 75.6%** (fraction of splits where the in-sample-best strategy lands below the out-of-sample median)
- Median performance-degradation logit: -0.95 (negative = overfit)

## Verdict

- DSR PASS: deflated-Sharpe 99% — the Sharpe survives multiple-testing; >95% confidence it is skill, not the luckiest draw.
- PBO WEAK: 76% — elevated overfitting risk.

These are the statistics professional quant funds use to vet a strategy before risking capital — most retail/student backtests never compute them.