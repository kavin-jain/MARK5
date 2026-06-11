# MARK6 — Institutional Evaluation Report

**System:** 50% concentrated 12-name momentum-heavy factor book (refreshed every 6 months, FY tax netting) + 25% gold (GOLDBEES) + 25% US Nasdaq-100 (MON100) — three uncorrelated sleeves, sleeves rebalanced annually. **Mode:** PAPER. **Period:** 2016-01-01 → 2026-06-09. All figures **net of Indian tax (LTCG 12.5% / STCG 20%) + 0.29% costs + 0.10% slippage**. Universe is point-in-time (survivorship-aware; true returns ~2-3pp below gross-of-survivorship).

## 1. Headline performance

| Metric | MARK6 (deployed) | Nifty50 B&H |
|---|---|---|
| Net CAGR | **+20.8%** | +11.0% |
| Volatility (ann.) | 22.7% | 16.2% |
| **Sharpe** | **0.96** | 0.74 |
| Sortino | 1.07 | 0.90 |
| Max drawdown | -26.6% | -38.4% |
| Calmar | 0.78 | 0.29 |
| Excess return vs Nifty 50 | **+9.8pp** | — |
| Jensen's α vs Nifty 50 (CAPM, single-factor) | +14.5%/yr | — |
| Factor+refresh alpha (vs equal-weight same universe) | **+5.3pp/yr** | — |
| Beta vs Nifty | 0.61 | 1.00 |
| Max-DD recovery | 88 days | — |

₹50,000,000 → **₹359,062,694** over 10.4 years (net).

## 2. Trade ledger (evidence)

- Total trades: **471** (156 buys, 315 sells) over 10.4 years — full detail in `reports/trade_ledger.csv`.
- **Win rate: 73%** (229 wins / 86 losses on closed sells).
- **Profit factor: 3.27** (₹417,465,173 gross profit / ₹127,617,515 gross loss).
- Tax efficiency: 97/315 sells qualified for LTCG (long holds).
- Avg holding period: 262 days.

  Largest winners (₹, scaled to capital):

  | date | ticker | held(d) | P&L ₹ |
  |---|---|---|---|
  | 2024-09-02 | HAL | 644 | 28,182,593 |
  | 2024-09-02 | RECLTD | 374 | 28,156,472 |
  | 2024-09-02 | PFC | 374 | 22,209,767 |
  | 2024-02-27 | UNIONBANK | 317 | 13,194,451 |
  | 2024-09-02 | IRCON | 374 | 13,136,139 |
  | 2024-02-27 | VBL | 557 | 12,424,471 |
  | 2024-09-02 | RVNL | 374 | 11,427,407 |
  | 2025-03-03 | RVNL | 556 | 10,257,332 |

## 3. Year-by-year net return

| Year | MARK6 | Nifty50 |
|---|---|---|
| 2016 | +8.4% | +5.1% |
| 2017 | +35.7% | +28.7% |
| 2018 | -6.3% | +3.2% |
| 2019 | +22.5% | +12.0% |
| 2020 | +31.1% | +14.8% |
| 2021 | +28.8% | +23.8% |
| 2022 | -1.6% | +2.7% |
| 2023 | +52.1% | +19.4% |
| 2024 | +37.9% | +8.8% |
| 2025 | +20.2% | +10.1% |
| 2026 | -2.4% | -11.1% |

## 4. Stress tests — real crises (drawdown survival)

| Scenario | MARK6 | Nifty50 | MARK6 max DD in window |
|---|---|---|---|
| 2018 NBFC/IL&FS | -3.9% | -4.9% | -15.0% |
| COVID crash 2020 | +0.2% | -15.8% | -26.6% |
| 2022 bear/rate-shock | -14.3% | -10.5% | -19.0% |
| 2024-25 correction | -5.1% | -7.0% | -15.5% |

## 5. Monte Carlo — unpredicted-event robustness (2000 block-bootstrap 5-yr paths)

- Median 5-yr CAGR: **+23.0%** | 5th-percentile (bad luck): +8.9% | 95th: +37.4%
- Worst simulated drawdown: **-51.2%** | 5th-pctile DD: -37.7%
- Probability of a NEGATIVE 5-year outcome: **0.1%**

## 6. Industry-standard scorecard

| Dimension | This system | Industry reference | Verdict |
|---|---|---|---|
| Sharpe | 0.96 | MF ~0.5-0.8, HF ~1.0, Medallion ~2+ | strong (top-quartile MF) |
| Calmar | 0.78 | >0.5 good, >1.0 excellent | good |
| Jensen's α vs Nifty 50 | +14.5%/yr | >0 = adds value (note: partly multi-asset) | positive |
| Max drawdown | -26.6% | equity norm -30 to -55% | within norm |
| Beta | 0.61 | <1 = defensive | defensive |

## 7. Honest verdict

- **Sharpe 0.96, excess return +9.8pp vs Nifty 50, Calmar 0.78** — a genuine, index-beating smart-beta portfolio, in the strong-MF / lower-hedge-fund tier. (The full excess return reflects multi-asset allocation + universe + factor; factor ranking + 6-mo refresh contributes +5.3pp/yr above equal-weight same universe.)
- It is **not** a 20%+ or Sharpe-2 machine (those need leverage/HFT we've proven unavailable).
- Drawdowns of -28 to -35% are real and unavoidable; the Monte Carlo bad-luck tail is the honest risk you must be able to hold through.
- All claims are evidenced by the trade ledger and reproducible via this script.
