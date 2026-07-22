# MARK6 — Honest Smart-Beta Portfolio System

> ⚠️ **HISTORICAL DESIGN DOCUMENT (v6.0, 2026-06-08).** The numbers below
> (+13.4% CAGR, walk-forward 3/8) describe the ORIGINAL 20-name annual-rebalance
> book *before* the warmup-bug fix, concentration upgrade (P5), momentum-heavy
> weights (P9), FY tax netting (P11), semi-annual refresh (P12), and the v7.1
> audit fixes (TRI benchmark, FIFO lots, next-close execution). Current
> validated numbers live in [`../README.md`](../README.md) and
> [`../reports/MARK6_REPORT.md`](../reports/MARK6_REPORT.md) — equity sleeve
> +20.0% net, walk-forward 7/8 vs Nifty TRI. The design *rationale* below is
> still the system's foundation; the figures are superseded.

> **The one-line truth:** You cannot beat *same-universe* buy-and-hold by trading
> (we proved it across 25+ configs, 2 universes, 2 windows, net of tax). You **can**
> beat the **cap-weighted Nifty 50** — the thing a normal investor means by "buy and
> hold" — with disciplined multi-factor construction held through the cycle. MARK6
> does that: **+13.4% net CAGR vs Nifty's +10.4% over 2016-2026 (+3.0pp/yr), with a
> higher Sharpe (0.86 vs 0.69) and a 10pp smaller drawdown than naive equal-weight.**

This system is the product of a full investigation that dismantled the prior
MARK5 ML/momentum stack. The headline finding of that investigation
([honest-oos-verdict] in project memory): the old system returned **+7.8%** on data
it was never tuned on while equal-weight buy-and-hold returned **+23.2%** — the
*trading machinery was destroying value.* MARK6 keeps only what survived honest,
survivorship-corrected, net-of-tax, walk-forward validation.

---

## 1. What MARK6 is (and is not)

| | |
|---|---|
| **Is** | A long-only, fully-invested, multi-factor equity portfolio. Quarterly→annual rebalanced. Survivorship-aware. Tax-aware. |
| **Is not** | A timing system. There are **no** stop-losses, regime-to-cash gates, or circuit breakers — every one of those was proven to *destroy* return net of tax. |
| **Beats** | The cap-weighted Nifty 50 over a full cycle, with better risk-adjusted return. |
| **Does not beat** | Same-universe buy-and-hold (no long-only daily-OHLCV strategy does, net of Indian tax). |

## 2. Why these design choices (each is a lesson paid for in the investigation)

- **Multi-factor, not momentum-only.** Single-factor momentum looked like +5pp gross
  but *died* net of tax (400%+ turnover → STCG). A diversified composite
  (momentum + low-vol + trend + stability) is more robust and lower-turnover.
- **Inverse-volatility weighting.** The −44% drawdown of naive equal-weight is the
  real enemy of compounding. Inverse-vol + the low-vol factor cut it to −34% while
  matching return — the single biggest *risk-adjusted* improvement.
- **Annual rebalance + ranking buffer.** Turnover is the tax killer. Holding past
  365 days converts STCG (20%) into LTCG (12.5%). This lever alone moved the strategy
  from net-negative to net-positive vs the benchmark.
- **Point-in-time universe.** Eligibility (seasoned + liquid *as of* each date) is
  computed with no look-ahead — the structural defence against the survivorship bias
  that inflated earlier numbers by ~9pp.

## 3. Architecture

```
core/portfolio/
  factors.py        Causal factor library (momentum, low_vol, trend, stability)
                    + cross-sectional z-scoring + composite blending. NO look-ahead.
  universe.py       DataPanel (OHLCV loader) + point-in-time eligibility
                    (existence + liquidity as-of date). Structural exclusions only.
  construction.py   ConstructionConfig + PortfolioConstructor: composite -> selection
                    (buffered) -> weighting (equal / inverse-vol + score tilt) ->
                    name & sector caps (water-filling, convergent).
  backtest.py       Backtester: daily NAV, per-name lot tax (LTCG/STCG + terminal),
                    costs, full metrics (CAGR/Sharpe/Sortino/MaxDD/Calmar/turnover).

scripts/
  run_mark6.py             -> reports/mark6_results.json + reports/MARK6_REPORT.md
  survivorship_validation.py  Bounds residual survivorship bias (failure injection).
  factor_tilt_optimize.py     The net-of-tax config sweep that informed the design.

tests/test_portfolio.py   22 tests: causality, scoring, constraints, tax, integration.
```

## 4. Results (net of tax, point-in-time universe)

### Headline windows
| Window | MARK6 | EqualWeight | Nifty50 B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| **FULL 2016-2026** | **+13.4%** (Sh 0.86, DD −34%) | +13.6% (DD −44%) | +10.4% (Sh 0.69) | **+3.0pp** | −0.2pp |
| OOS-era 2016-2021 | +16.3% (Sh 1.00) | +13.0% | +12.9% | +3.4pp | +3.3pp |
| recent 2022-2026 | +12.8% (Sh 0.90, DD −25%) | +10.9% | +6.2% (Sh 0.51) | **+6.7pp** | +1.9pp |

### Rolling 3-year walk-forward (the honesty test)
**Beats Nifty50 in 3/8 windows; beats EqualWeight in 5/8.** The index outperformance
is *real over the cycle* but *regime-dependent*: MARK6 wins big in broad/midcap
regimes (2022-24: +14pp) and lags in narrow large-cap rallies (2017-19: −13.5pp).
This is beta-plus-construction, **not** reliable alpha — and we show it plainly
rather than reporting only the favourable full-period number.

## 5. Integrity guarantees

- **No look-ahead:** every factor value at bar *t* uses only data ≤ *t* (unit-tested).
- **Net-to-net:** tax + costs applied to the strategy *and* every benchmark.
- **Survivorship-aware:** point-in-time eligibility; residual bias from fully-delisted
  names bounded at ~2-3pp by `survivorship_validation.py` (failure injection).
- **Walk-forward:** evaluated on rolling windows, not a single tuned period.
- **Tested:** 22 unit/integration tests; 1543 repo tests pass.

## 6. How to run

```bash
# Full performance report (uses cached data in data/cache/)
python3 scripts/run_mark6.py
# -> reports/MARK6_REPORT.md, reports/mark6_results.json

# Bound the residual survivorship bias
python3 scripts/survivorship_validation.py

# Tests
pytest tests/test_portfolio.py -v
```

To refresh / extend the universe (network required):
```bash
python3 scripts/fetch_midcaps.py     # caches NSE midcaps to data/cache/
```

## 7. Honest limitations (read before risking capital)

1. **Drawdowns are equity-level (−25% to −44%).** Inverse-vol weighting reduces but
   cannot remove them. The legacy 5%-hard-stop design is *incompatible* with equity
   returns and was proven to destroy the edge — do not re-add it.
2. **The edge over the index is regime-dependent**, concentrated in midcap-favourable
   periods. Over a *full* cycle it is positive (+3pp); over any given 3 years it may
   not be.
3. **Survivorship:** the universe is today's listed names; the true forward number is
   modestly below the backtest. Bounded, not eliminated, without point-in-time
   constituent data (not freely available).
4. **No fundamentals.** Factors are price/volume-only (no earnings/value/quality from
   financials). Adding fundamental factors is the most promising honest extension.

## 8. What would genuinely improve this (and what would not)

**Would (logically-backed):** point-in-time index-constituent data (kills residual
survivorship); fundamental quality/value factors; a low-beta sleeve for the large-cap
regimes where it lags; tax-loss harvesting.

**Would not (the treadmill):** more tuning on 2022-2026; any timing/stop overlay;
chasing a single hot factor; promising returns above the equity-beta ceiling. On NSE
daily OHLCV the honest ceiling is *index + a few points with real drawdowns* — a
genuinely good result, and the one this system delivers.
