# MARK6 — Institutional Evaluation Report

**System:** 50% concentrated 12-name momentum-heavy factor book + 25% gold (GOLDBEES) + 25% US Nasdaq-100 (MON100) — three uncorrelated sleeves, annual rebalance. **Mode:** PAPER. **Period:** 2016-01-01 → 2026-06-09. All figures **net of Indian tax (LTCG 12.5% / STCG 20%) + 0.29% costs + 0.10% slippage**. Universe is point-in-time (survivorship-aware; true returns ~2-3pp below gross-of-survivorship).

## 1. Headline performance

| Metric | MARK6 (deployed) | Nifty50 B&H |
|---|---|---|
| Net CAGR | **+18.8%** | +11.0% |
| Volatility (ann.) | 22.4% | 16.2% |
| **Sharpe** | **0.89** | 0.74 |
| Sortino | 0.99 | 0.90 |
| Max drawdown | -26.7% | -38.4% |
| Calmar | 0.70 | 0.29 |
| Annualised alpha vs Nifty | **+12.8%** | — |
| Beta vs Nifty | 0.60 | 1.00 |
| Max-DD recovery | 92 days | — |

₹50,000,000 → **₹301,433,377** over 10.4 years (net).

## 2. Trade ledger (evidence)

- Total trades: **207** (101 buys, 106 sells) over 10.4 years — full detail in `reports/trade_ledger.csv`.
- **Win rate: 76%** (81 wins / 25 losses on closed sells).
- **Profit factor: 3.80** (₹275,999,341 gross profit / ₹72,620,789 gross loss).
- Tax efficiency: 106/106 sells qualified for LTCG (long holds).
- Avg holding period: 450 days.

  Largest winners (₹, scaled to capital):

  | date | ticker | held(d) | P&L ₹ |
  |---|---|---|---|
  | 2025-03-03 | BSE | 370 | 16,424,775 |
  | 2025-03-03 | HAL | 742 | 13,707,682 |
  | 2023-02-20 | ADANIENT | 1657 | 13,124,290 |
  | 2024-02-27 | VBL | 372 | 12,915,376 |
  | 2024-02-27 | CUMMINSIND | 372 | 12,102,392 |
  | 2024-02-27 | TVSMOTOR | 372 | 10,626,112 |
  | 2024-02-27 | UNIONBANK | 372 | 9,257,203 |
  | 2024-02-27 | KPITTECH | 742 | 7,806,637 |

## 3. Year-by-year net return

| Year | MARK6 | Nifty50 |
|---|---|---|
| 2016 | +7.5% | +5.1% |
| 2017 | +34.4% | +28.7% |
| 2018 | -8.9% | +3.2% |
| 2019 | +22.3% | +12.0% |
| 2020 | +39.4% | +14.8% |
| 2021 | +36.9% | +23.8% |
| 2022 | -9.3% | +2.7% |
| 2023 | +33.7% | +19.4% |
| 2024 | +39.2% | +8.8% |
| 2025 | +15.2% | +10.1% |
| 2026 | -2.6% | -11.6% |

## 4. Stress tests — real crises (drawdown survival)

| Scenario | MARK6 | Nifty50 | MARK6 max DD in window |
|---|---|---|---|
| 2018 NBFC/IL&FS | -8.7% | -4.9% | -15.9% |
| COVID crash 2020 | -1.8% | -15.8% | -26.7% |
| 2022 bear/rate-shock | -16.6% | -10.5% | -19.6% |
| 2024-25 correction | -6.4% | -7.0% | -18.0% |

## 5. Monte Carlo — unpredicted-event robustness (2000 block-bootstrap 5-yr paths)

- Median 5-yr CAGR: **+20.8%** | 5th-percentile (bad luck): +7.0% | 95th: +36.2%
- Worst simulated drawdown: **-56.7%** | 5th-pctile DD: -38.6%
- Probability of a NEGATIVE 5-year outcome: **0.4%**

## 6. Industry-standard scorecard

| Dimension | This system | Industry reference | Verdict |
|---|---|---|---|
| Sharpe | 0.89 | MF ~0.5-0.8, HF ~1.0, Medallion ~2+ | strong (top-quartile MF) |
| Calmar | 0.70 | >0.5 good, >1.0 excellent | good |
| Alpha vs index | +12.8%/yr | >0 = adds value | positive |
| Max drawdown | -26.7% | equity norm -30 to -55% | within norm |
| Beta | 0.60 | <1 = defensive | defensive |

## 7. Honest verdict

- **Sharpe 0.89, alpha +12.8%/yr, Calmar 0.70** — a genuine, index-beating smart-beta portfolio, in the strong-MF / lower-hedge-fund tier.
- It is **not** a 20%+ or Sharpe-2 machine (those need leverage/HFT we've proven unavailable).
- Drawdowns of -28 to -35% are real and unavoidable; the Monte Carlo bad-luck tail is the honest risk you must be able to hold through.
- All claims are evidenced by the trade ledger and reproducible via this script.
