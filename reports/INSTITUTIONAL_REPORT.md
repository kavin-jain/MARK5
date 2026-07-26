# MARK6 — Institutional Evaluation Report

**System:** 50% 20-name momentum-heavy factor book (refreshed every 6 months, FY tax netting, FIFO lots, next-close execution) + 25% gold (GOLDBEES) + 25% US Nasdaq-100 (MON100) — three uncorrelated sleeves, sleeves rebalanced annually. **Mode:** PAPER. **Period:** 2016-01-01 → 2026-07-21. All figures **net of Indian tax (LTCG 12.5% / STCG 20%) + 0.29% costs + 0.10% slippage**. Benchmark is **Nifty 50 total-return** (dividends reinvested), taxed at terminal LTCG like the strategy. Universe eligibility is point-in-time, but the candidate list is today's survivors — headline is inflated an estimated ~1-2pp/yr by residual survivorship.

## 1. Headline performance

| Metric | MARK6 (deployed) | Nifty50 TRI B&H |
|---|---|---|
| Net CAGR | **+21.2%** | +10.9% |
| Volatility (ann.) | 14.9% | 14.9% |
| Sharpe (rf=0, raw) | 1.40 | 0.79 |
| **Sharpe (excess of 6.5% risk-free)** | **0.97** | 0.36 |
| Sortino | 1.28 | 0.48 |
| Max drawdown | -23.8% | -36.3% |
| Calmar | 0.89 | 0.30 |
| Excess return vs Nifty 50 TRI | **+10.2pp** | — |
| Jensen's α vs Nifty 50 (CAPM, single-factor) | +13.1%/yr | — |
| Factor+refresh alpha (vs equal-weight same universe, computed) | **+9.6pp/yr** | — |
| Beta vs Nifty | 0.61 | 1.00 |
| Max-DD recovery | 79 days | — |

₹500,000 → **₹3,789,964** over 10.6 years (net).

## 2. Trade ledger (evidence)

- Total trades: **857** (322 buys, 535 sells) over 10.6 years — full detail in `reports/trade_ledger.csv`.
- **Win rate: 69%** (370 wins / 165 losses on closed sells).
- **Profit factor: 3.28** (₹4,591,767 gross profit / ₹1,399,647 gross loss).
- Tax efficiency: 140/535 sells qualified for LTCG (long holds).
- Avg holding period: 227 days.

  Largest winners (₹, scaled to capital):

  | date | ticker | held(d) | P&L ₹ |
  |---|---|---|---|
  | 2024-09-09 | TATAMTRDVR | 378 | 122,330 |
  | 2023-08-28 | CGPOWER | 742 | 100,370 |
  | 2024-09-09 | ANANTRAJ | 378 | 97,780 |
  | 2024-09-09 | PFC | 378 | 95,128 |
  | 2024-09-09 | HAL | 749 | 92,970 |
  | 2024-03-01 | PFC | 186 | 89,362 |
  | 2024-03-01 | UJJIVAN | 374 | 72,087 |
  | 2022-08-22 | ATGL | 187 | 70,392 |

## 3. Year-by-year net return

| Year | MARK6 | Nifty50 |
|---|---|---|
| 2016 | +5.2% | +3.7% |
| 2017 | +44.6% | +29.9% |
| 2018 | -14.0% | +5.4% |
| 2019 | +20.0% | +13.3% |
| 2020 | +42.6% | +15.2% |
| 2021 | +36.3% | +25.6% |
| 2022 | -5.2% | +3.8% |
| 2023 | +46.1% | +20.2% |
| 2024 | +42.3% | +10.0% |
| 2025 | +17.6% | +11.2% |
| 2026 | +2.4% | -10.6% |

## 4. Stress tests — real crises (drawdown survival)

| Scenario | MARK6 | Nifty50 | MARK6 max DD in window |
|---|---|---|---|
| 2018 NBFC/IL&FS | -9.9% | -4.0% | -17.5% |
| COVID crash 2020 | +2.3% | -16.3% | -23.8% |
| 2022 bear/rate-shock | -13.6% | -10.0% | -19.6% |
| 2024-25 correction | -1.2% | -6.5% | -13.4% |

## 5. Monte Carlo — unpredicted-event robustness (2000 block-bootstrap 5-yr paths)

- Median 5-yr CAGR: **+23.8%** | 5th-percentile (bad luck): +11.6% | 95th: +37.3%
- Worst simulated drawdown: **-53.1%** | 5th-pctile DD: -30.2%
- Probability of a NEGATIVE 5-year outcome: **0.1%**

## 6. Industry-standard scorecard

| Dimension | This system | Industry reference | Verdict |
|---|---|---|---|
| Sharpe (excess of rf) | 0.97 | MF ~0.5-0.8, HF ~1.0, Medallion ~2+ | strong (top-quartile MF) |
| Calmar | 0.89 | >0.5 good, >1.0 excellent | good |
| Jensen's α vs Nifty 50 | +13.1%/yr | >0 = adds value (note: partly multi-asset) | positive |
| Max drawdown | -23.8% | equity norm -30 to -55% | within norm |
| Beta | 0.61 | <1 = defensive | defensive |

## 7. Honest verdict

- **Excess Sharpe 0.97, excess return +10.2pp vs Nifty 50 TRI, Calmar 0.89** — a genuine, index-beating smart-beta portfolio in the strong-MF tier. (The full excess return reflects multi-asset allocation + universe + factor; factor ranking + 6-mo refresh contributes +9.6pp/yr above equal-weight of the same universe — the rest is asset allocation any multi-asset fund also captures.)
- Survivorship caveat: subtract ~1-2pp/yr from the headline for the missing delisted names; the realistic forward expectation is ~19-21% CAGR over a full cycle, with single years anywhere from -15% to +40%.
- It is not a Sharpe-2 machine (that needs leverage/infrastructure unavailable at retail).
- Drawdowns of -25 to -35% are real and unavoidable; the Monte Carlo bad-luck tail is the honest risk you must be able to hold through.
- All claims are evidenced by the trade ledger and reproducible via this script (local data cache; a fresh clone rebuilds it with scripts/refetch_all.py from the pinned config/universe_tickers.json).
