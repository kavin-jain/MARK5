# MARK6 — Institutional Evaluation Report

**System:** 50% 20-name momentum-heavy factor book (refreshed every 6 months, FY tax netting, FIFO lots, next-close execution) + 25% gold (GOLDBEES) + 25% US Nasdaq-100 (MON100) — three uncorrelated sleeves, sleeves rebalanced annually. **Mode:** PAPER. **Period:** 2016-01-01 → 2026-07-21. All figures **net of Indian tax (LTCG 12.5% / STCG 20%) + 0.29% costs + 0.10% slippage**. Benchmark is **Nifty 50 total-return** (dividends reinvested), taxed at terminal LTCG like the strategy. Universe eligibility is point-in-time and the candidate list is too: 1341 symbols from `data/pit_cache`, **185** of which delisted inside the window and are held until they stop trading. No survivorship haircut applies to the headline.

## 1. Headline performance

| Metric | MARK6 (deployed) | Nifty50 TRI B&H |
|---|---|---|
| Net CAGR | **+21.1%** | +10.9% |
| Volatility (ann.) | 14.2% | 14.7% |
| Sharpe (rf=0, raw) | 1.55 | 0.86 |
| **Sharpe (excess of 6.5% risk-free)** | **1.10** | 0.43 |
| Sortino | 1.50 | 0.58 |
| Max drawdown | -23.5% | -36.3% |
| Calmar | 0.89 | 0.30 |
| Excess return vs Nifty 50 TRI | **+10.1pp** | — |
| Jensen's α vs Nifty 50 (CAPM, single-factor) | +14.5%/yr | — |
| Factor+refresh alpha (vs equal-weight same universe, computed) | **+8.6pp/yr** | — |
| Beta vs Nifty | 0.59 | 1.00 |
| Max-DD recovery | 79 days | — |

₹500,000 → **₹3,756,281** over 10.6 years (net).

## 2. Trade ledger (evidence)

- Total trades: **882** (343 buys, 539 sells) over 10.6 years — full detail in `reports/trade_ledger.csv`.
- **Win rate: 66%** (356 wins / 183 losses on closed sells).
- **Profit factor: 2.71** (₹5,056,687 gross profit / ₹1,866,436 gross loss).
- Tax efficiency: 108/539 sells qualified for LTCG (long holds).
- Avg holding period: 206 days.

  Largest winners (₹, scaled to capital):

  | date | ticker | held(d) | P&L ₹ |
  |---|---|---|---|
  | 2024-09-09 | ANANTRAJ | 378 | 192,608 |
  | 2024-03-01 | RVNL | 374 | 110,667 |
  | 2023-08-28 | CGPOWER | 742 | 108,855 |
  | 2024-03-01 | ACE | 186 | 108,670 |
  | 2022-08-22 | ATGL | 187 | 98,986 |
  | 2024-09-09 | PFC | 378 | 97,215 |
  | 2024-03-01 | PFC | 186 | 88,998 |
  | 2025-03-11 | NEULANDLAB | 561 | 82,062 |

## 3. Year-by-year net return

| Year | MARK6 | Nifty50 |
|---|---|---|
| 2016 | +4.7% | +3.7% |
| 2017 | +41.3% | +29.9% |
| 2018 | -15.0% | +5.4% |
| 2019 | +22.7% | +13.3% |
| 2020 | +45.8% | +15.2% |
| 2021 | +35.6% | +25.6% |
| 2022 | -0.8% | +3.8% |
| 2023 | +41.3% | +20.2% |
| 2024 | +37.6% | +10.0% |
| 2025 | +16.3% | +11.2% |
| 2026 | +20.2% | -10.6% |

## 4. Stress tests — real crises (drawdown survival)

| Scenario | MARK6 | Nifty50 | MARK6 max DD in window |
|---|---|---|---|
| 2018 NBFC/IL&FS | -8.6% | -4.0% | -16.5% |
| COVID crash 2020 | +2.8% | -16.3% | -23.5% |
| 2022 bear/rate-shock | -10.1% | -10.0% | -19.0% |
| 2024-25 correction | -1.8% | -6.5% | -12.7% |

## 5. Monte Carlo — unpredicted-event robustness (2000 block-bootstrap 5-yr paths)

- Median 5-yr CAGR: **+23.8%** | 5th-percentile (bad luck): +11.5% | 95th: +36.9%
- Worst simulated drawdown: **-50.2%** | 5th-pctile DD: -30.1%
- Probability of a NEGATIVE 5-year outcome: **0.2%**

## 6. Industry-standard scorecard

| Dimension | This system | Industry reference | Verdict |
|---|---|---|---|
| Sharpe (excess of rf) | 1.10 | MF ~0.5-0.8, HF ~1.0, Medallion ~2+ | institutional/hedge-fund-tier |
| Calmar | 0.89 | >0.5 good, >1.0 excellent | good |
| Jensen's α vs Nifty 50 | +14.5%/yr | >0 = adds value (note: partly multi-asset) | positive |
| Max drawdown | -23.5% | equity norm -30 to -55% | within norm |
| Beta | 0.59 | <1 = defensive | defensive |

## 7. Honest verdict

- **Excess Sharpe 1.10, excess return +10.1pp vs Nifty 50 TRI, Calmar 0.89** — a genuine, index-beating smart-beta portfolio in the strong-MF tier. (The full excess return reflects multi-asset allocation + universe + factor; factor ranking + 6-mo refresh contributes +8.6pp/yr above equal-weight of the same universe — the rest is asset allocation any multi-asset fund also captures.)
- Survivorship: none — 185 of 1341 candidates delisted in-window and their failure is priced in. Forward expectation is still regime-dependent: single years have ranged -15% to +45%.
- It is not a Sharpe-2 machine (that needs leverage/infrastructure unavailable at retail).
- Drawdowns of -25 to -35% are real and unavoidable; the Monte Carlo bad-luck tail is the honest risk you must be able to hold through.
- All claims are evidenced by the trade ledger and reproducible via this script (local data cache; a fresh clone rebuilds it with scripts/refetch_all.py from the pinned config/universe_tickers.json).
