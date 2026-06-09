# MARK6 — Institutional Evaluation Report

**System:** 70% concentrated 12-name factor book + 15% gold (GOLDBEES) + 15% US Nasdaq-100 (MON100) — three uncorrelated sleeves, annual rebalance. **Mode:** PAPER. **Period:** 2016-01-01 → 2026-06-05. All figures **net of Indian tax (LTCG 12.5% / STCG 20%) + 0.29% costs + 0.10% slippage**. Universe is point-in-time (survivorship-aware; true returns ~2-3pp below gross-of-survivorship).

## 1. Headline performance

| Metric | MARK6 (deployed) | Nifty50 B&H |
|---|---|---|
| Net CAGR | **+17.3%** | +11.1% |
| Volatility (ann.) | 18.1% | 16.2% |
| **Sharpe** | **0.99** | 0.74 |
| Sortino | 1.10 | 0.91 |
| Max drawdown | -28.2% | -38.4% |
| Calmar | 0.61 | 0.29 |
| Annualised alpha vs Nifty | **+9.7%** | — |
| Beta vs Nifty | 0.68 | 1.00 |
| Max-DD recovery | 107 days | — |

₹500,000 → **₹2,635,967** over 10.4 years (net).

## 2. Trade ledger (evidence)

- Total trades: **203** (100 buys, 103 sells) over 10.4 years — full detail in `reports/trade_ledger.csv`.
- **Win rate: 84%** (87 wins / 16 losses on closed sells).
- **Profit factor: 8.65** (₹2,562,054 gross profit / ₹296,155 gross loss).
- Tax efficiency: 103/103 sells qualified for LTCG (long holds).
- Avg holding period: 461 days.

  Largest winners (₹, scaled to capital):

  | date | ticker | held(d) | P&L ₹ |
  |---|---|---|---|
  | 2024-02-27 | COALINDIA | 372 | 143,171 |
  | 2024-02-27 | PFC | 372 | 117,515 |
  | 2025-03-03 | HAL | 742 | 116,193 |
  | 2025-03-03 | BSE | 370 | 106,219 |
  | 2025-03-03 | TVSMOTOR | 688 | 103,784 |
  | 2024-02-27 | CUMMINSIND | 372 | 98,307 |
  | 2025-03-03 | PFC | 742 | 94,208 |
  | 2026-03-09 | BSE | 741 | 89,770 |

## 3. Year-by-year net return

| Year | MARK6 | Nifty50 |
|---|---|---|
| 2016 | +4.5% | +5.1% |
| 2017 | +35.5% | +28.7% |
| 2018 | -6.5% | +3.2% |
| 2019 | +16.1% | +12.0% |
| 2020 | +30.2% | +14.8% |
| 2021 | +39.3% | +23.8% |
| 2022 | -5.9% | +2.7% |
| 2023 | +44.5% | +19.4% |
| 2024 | +43.0% | +8.8% |
| 2025 | +6.0% | +10.1% |
| 2026 | -11.0% | -10.6% |

## 4. Stress tests — real crises (drawdown survival)

| Scenario | MARK6 | Nifty50 | MARK6 max DD in window |
|---|---|---|---|
| 2018 NBFC/IL&FS | -9.6% | -4.9% | -14.1% |
| COVID crash 2020 | -2.5% | -15.8% | -28.2% |
| 2022 bear/rate-shock | -16.5% | -10.5% | -20.2% |
| 2024-25 correction | -8.2% | -7.0% | -17.5% |

## 5. Monte Carlo — unpredicted-event robustness (2000 block-bootstrap 5-yr paths)

- Median 5-yr CAGR: **+19.2%** | 5th-percentile (bad luck): +5.7% | 95th: +33.6%
- Worst simulated drawdown: **-55.6%** | 5th-pctile DD: -37.4%
- Probability of a NEGATIVE 5-year outcome: **0.9%**

## 6. Industry-standard scorecard

| Dimension | This system | Industry reference | Verdict |
|---|---|---|---|
| Sharpe | 0.99 | MF ~0.5-0.8, HF ~1.0, Medallion ~2+ | strong (top-quartile MF) |
| Calmar | 0.61 | >0.5 good, >1.0 excellent | good |
| Alpha vs index | +9.7%/yr | >0 = adds value | positive |
| Max drawdown | -28.2% | equity norm -30 to -55% | within norm |
| Beta | 0.68 | <1 = defensive | defensive |

## 7. Honest verdict

- **Sharpe 0.99, alpha +9.7%/yr, Calmar 0.61** — a genuine, index-beating smart-beta portfolio, in the strong-MF / lower-hedge-fund tier.
- It is **not** a 20%+ or Sharpe-2 machine (those need leverage/HFT we've proven unavailable).
- Drawdowns of -28 to -35% are real and unavoidable; the Monte Carlo bad-luck tail is the honest risk you must be able to hold through.
- All claims are evidenced by the trade ledger and reproducible via this script.
