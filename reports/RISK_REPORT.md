# MARK6 — Institutional Risk Report

Deployed configuration on the survivorship-free point-in-time universe (1341 symbols), 2016-01-01 → 2026-07-21. All figures computed from the daily net series; nothing here is illustrative.

## 1. Tail risk

Historical VaR makes no distributional assumption; parametric assumes normality. The **gap between them is the fat tail** — where parametric is smaller, a normal model is understating how bad the bad days get.

| Horizon | Confidence | Historical VaR | Historical CVaR | Parametric VaR |
|---|---|---|---|---|
| 1 day | 95% | -1.37% | -2.25% | -1.39% |
| 1 day | 99% | -2.83% | -3.80% | -2.00% |
| 21 days | 95% | -5.57% | -8.56% | -4.94% |
| 21 days | 99% | -9.72% | -14.46% | -7.75% |

Daily skew **-1.00**, kurtosis **9.9** (normal = 3.0) — fat-tailed and negatively skewed, the usual equity shape. Worst day -8.14%, best +4.77%.

### Full system 50/25/25

Annualised alpha **+10.96%** (t = 3.68), R² = 0.56.

| Factor | Beta | t-stat |
|---|---|---|
| market | +0.514 | 31.89 |
| size | +0.055 | 2.15 |
| momentum | +0.358 | 22.08 |
| lowvol | -0.230 | -16.14 |

### Equity sleeve only

Annualised alpha **+5.79%** (t = 1.61), R² = 0.71.

| Factor | Beta | t-stat |
|---|---|---|
| market | +0.768 | 39.34 |
| size | +0.060 | 1.94 |
| momentum | +0.567 | 28.86 |
| lowvol | -0.498 | -28.79 |

Factors are long/short terciles rebuilt every 21 bars from the same point-in-time universe (size = small minus big by turnover; momentum and low-volatility from the engine's own causal definitions). A high R² with a large momentum beta would mean the book is simply a momentum index; a surviving positive alpha means the ranking adds something the raw factors do not.

## 3. Drawdown attribution

Each drawdown worse than −5%, decomposed into the rupee P&L each sleeve contributed while it was happening. This is what makes the multi-asset structure load-bearing rather than decorative.

| Peak | Trough | Depth | Days | Equity | Gold | US |
|---|---|---|---|---|---|---|
| 2020-02-28 | 2020-03-23 | -23.8% | 103 | -15.9% | -1.8% | -5.2% |
| 2022-04-25 | 2022-06-20 | -19.6% | 375 | -12.9% | -0.9% | -4.2% |
| 2018-02-02 | 2018-12-26 | -17.6% | 689 | -10.1% | +0.8% | -5.6% |
| 2025-01-10 | 2025-04-07 | -15.2% | 192 | -9.5% | +3.1% | -5.0% |
| 2026-02-02 | 2026-03-23 | -12.3% | 72 | -3.4% | -4.2% | -1.1% |

## 4. Stress tests

| Scenario | System | Nifty50 TRI |
|---|---|---|
| 2018 NBFC / IL&FS | -9.9% | -4.0% |
| COVID crash 2020 | +2.3% | -16.3% |
| 2022 rate shock | -13.6% | -10.0% |
| 2024-25 correction | -1.2% | -6.5% |

Worst rolling 1-year return **-15.8%** (ending 2019-01-30); **12%** of rolling 1-year windows were negative. A holder must be able to sit through both.

