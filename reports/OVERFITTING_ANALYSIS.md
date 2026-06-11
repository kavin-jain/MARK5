# MARK6 — Overfitting & Statistical-Significance Analysis

Bailey & López de Prado tests on the deployed v7.0 config (momentum-heavy / n_hold=12 / tilt=1.5 / 126-bar refresh, FY-netting tax), using every strategy variant explored across the project as the trial set (factor-weight grid, rebalance frequencies, asymmetric exits, TLH, FIP, sleeve frequencies). All on daily returns, 2016-2026.

## Deflated Sharpe Ratio (is the Sharpe real, given how many we tried?)

- Strategy variants tried (N): **77**
- Observed Sharpe: **0.96** annualised (0.060 daily)
- Probabilistic Sharpe Ratio vs 0 (P true SR>0): **99.8%**
- Expected max Sharpe from pure luck across 77 trials: 0.16 annualised
- **Deflated Sharpe Ratio (P skill survives multiple-testing): 99.3%**

## Probability of Backtest Overfitting (PBO via CSCV)

- Strategies in matrix: 65 | train/test combos: 924
- **PBO: 74.5%** (fraction of splits where the in-sample-best strategy lands below the out-of-sample median)
- Median performance-degradation logit: -1.06 (negative = overfit)

## Verdict

- DSR PASS: deflated-Sharpe 99% — the Sharpe survives multiple-testing; >95% confidence it is skill, not the luckiest draw.
- PBO WEAK: 74% — elevated overfitting risk.

These are the statistics professional quant funds use to vet a strategy before risking capital — most retail/student backtests never compute them.