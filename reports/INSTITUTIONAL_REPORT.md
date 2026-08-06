# MARK6 — Institutional Evaluation Report

**System:** 50% 20-name momentum-heavy factor book (refreshed every 6 months, FY tax netting, FIFO lots, next-close execution) + 25% gold (GOLDBEES) + 25% US Nasdaq-100 (MON100) — three uncorrelated sleeves, sleeves rebalanced annually. **Mode:** PAPER. **Period:** 2016-01-01 → 2026-07-21. All figures **net of Indian tax (LTCG 12.5% / STCG 20%) + 0.29% costs + 0.10% slippage**. Benchmark is **Nifty 50 total-return** (dividends reinvested), taxed at terminal LTCG like the strategy. Universe eligibility is point-in-time and the candidate list is too: 1337 symbols from `data/pit_cache`, **180** of which delisted inside the window and are held until they stop trading. No survivorship haircut applies to the headline.

## 1. Headline performance

| Metric | MARK6 (deployed) | Nifty50 TRI B&H |
|---|---|---|
| Net CAGR | **+21.8%** | +10.9% |
| Volatility (ann.) | 14.4% | 14.7% |
| Sharpe (rf=0, raw) | 1.57 | 0.86 |
| **Sharpe (excess of 6.5% risk-free)** | **1.13** | 0.43 |
| Sortino | 1.55 | 0.58 |
| Max drawdown | -23.8% | -36.3% |
| Calmar | 0.92 | 0.30 |
| Excess return vs Nifty 50 TRI | **+10.9pp** | — |
| Jensen's α vs Nifty 50 (CAPM, single-factor) | +15.0%/yr | — |
| Factor+refresh alpha (vs equal-weight same universe, computed) | **+10.1pp/yr** | — |
| Beta vs Nifty | 0.61 | 1.00 |
| Max-DD recovery | 79 days | — |

₹500,000 → **₹4,016,885** over 10.6 years (net).

## 2. Trade ledger (evidence)

- Total trades: **879** (323 buys, 556 sells) over 10.6 years — full detail in `reports/trade_ledger.csv`.
- **Win rate: 69%** (386 wins / 170 losses on closed sells).
- **Profit factor: 3.47** (₹4,808,161 gross profit / ₹1,385,337 gross loss).
- Tax efficiency: 139/556 sells qualified for LTCG (long holds).
- Avg holding period: 223 days.

  Largest winners (₹, scaled to capital):

  | date | ticker | held(d) | P&L ₹ |
  |---|---|---|---|
  | 2024-09-09 | TATAMTRDVR | 378 | 127,106 |
  | 2023-08-28 | CGPOWER | 742 | 105,284 |
  | 2024-09-09 | PFC | 378 | 102,484 |
  | 2024-09-09 | ANANTRAJ | 378 | 99,586 |
  | 2024-03-01 | RVNL | 374 | 96,074 |
  | 2024-09-09 | HAL | 749 | 94,503 |
  | 2024-03-01 | PFC | 186 | 90,205 |
  | 2026-03-19 | NAVINFLUOR | 373 | 80,916 |

## 3. Year-by-year net return

| Year | MARK6 | Nifty50 |
|---|---|---|
| 2016 | +7.4% | +3.7% |
| 2017 | +44.0% | +29.9% |
| 2018 | -14.7% | +5.4% |
| 2019 | +20.1% | +13.3% |
| 2020 | +41.3% | +15.2% |
| 2021 | +35.2% | +25.6% |
| 2022 | -3.7% | +3.8% |
| 2023 | +48.0% | +20.2% |
| 2024 | +45.3% | +10.0% |
| 2025 | +15.3% | +11.2% |
| 2026 | +22.7% | -10.6% |

## 4. Stress tests — real crises (drawdown survival)

| Scenario | MARK6 | Nifty50 | MARK6 max DD in window |
|---|---|---|---|
| 2018 NBFC/IL&FS | -9.9% | -4.0% | -16.8% |
| COVID crash 2020 | +2.2% | -16.3% | -23.8% |
| 2022 bear/rate-shock | -13.3% | -10.0% | -19.5% |
| 2024-25 correction | -1.6% | -6.5% | -14.0% |

## 5. Monte Carlo — unpredicted-event robustness (2000 block-bootstrap 5-yr paths)

- Median 5-yr CAGR: **+24.4%** | 5th-percentile (bad luck): +12.0% | 95th: +37.9%
- Worst simulated drawdown: **-51.6%** | 5th-pctile DD: -30.0%
- Probability of a NEGATIVE 5-year outcome: **0.1%**

## 6. Industry-standard scorecard

| Dimension | This system | Industry reference | Verdict |
|---|---|---|---|
| Sharpe (excess of rf) | 1.13 | MF ~0.5-0.8, HF ~1.0, Medallion ~2+ | institutional/hedge-fund-tier |
| Calmar | 0.92 | >0.5 good, >1.0 excellent | good |
| Jensen's α vs Nifty 50 | +15.0%/yr | >0 = adds value (note: partly multi-asset) | positive |
| Max drawdown | -23.8% | equity norm -30 to -55% | within norm |
| Beta | 0.61 | <1 = defensive | defensive |

## 7. Honest verdict

- **Excess Sharpe 1.13, excess return +10.9pp vs Nifty 50 TRI, Calmar 0.92** — a genuine, index-beating smart-beta portfolio in the strong-MF tier. (The full excess return reflects multi-asset allocation + universe + factor; factor ranking + 6-mo refresh contributes +10.1pp/yr above equal-weight of the same universe — the rest is asset allocation any multi-asset fund also captures.)
- Survivorship: none — 180 of 1337 candidates delisted in-window and their failure is priced in. Forward expectation is still regime-dependent: single years have ranged -15% to +45%.
- It is not a Sharpe-2 machine (that needs leverage/infrastructure unavailable at retail).
- Drawdowns of -25 to -35% are real and unavoidable; the Monte Carlo bad-luck tail is the honest risk you must be able to hold through.
- All claims are evidenced by the trade ledger and reproducible via this script (local data cache; a fresh clone rebuilds it with scripts/refetch_all.py from the pinned config/universe_tickers.json).
