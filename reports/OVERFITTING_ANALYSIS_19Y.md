# MARK6 — Overfitting & Statistical-Significance Analysis

Bailey & López de Prado tests on the DEPLOYED v7.5 config (momentum-heavy / n_hold=20 / tilt=1.5 / 126-bar refresh, rank-transformed scores, sector cap enforced, FY-netting tax), using every strategy variant explored across the project as the trial set (factor-weight grid, rebalance frequencies, asymmetric exits, TLH, FIP, sleeve frequencies). All on daily returns, **2007-2026 (19.5y, includes the 2008 crash)**, universe `data/pit_cache_2007`.

## Deflated Sharpe Ratio (is the Sharpe real, given how many we tried?)

- Strategy variants tried (N): **124**
- Observed Sharpe: **0.63** annualised (0.040 daily)
- Probabilistic Sharpe Ratio vs 0 (P true SR>0): **99.7%**
- Expected max Sharpe from pure luck across 124 trials: 0.46 annualised
- **Deflated Sharpe Ratio (P skill survives multiple-testing): 76.8%**

## Probability of Backtest Overfitting (PBO via CSCV)

- Strategies in matrix: 70 | train/test combos: 924
- **PBO: 53.1%** (fraction of splits where the in-sample-best strategy lands below the out-of-sample median)
- Median performance-degradation logit: -0.20 (negative = overfit)

## Verdict

- DSR WEAK: deflated-Sharpe 77% — caution — significance is borderline after deflation.
- PBO WEAK: 53% — elevated overfitting risk.

These are the statistics professional quant funds use to vet a strategy before risking capital — most retail/student backtests never compute them.