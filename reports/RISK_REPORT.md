# MARK6 — Institutional Risk Report

Deployed configuration on the survivorship-free point-in-time universe (1341 symbols), 2016-01-01 → 2026-07-21. All figures computed from the daily net series; nothing here is illustrative.

## 1. Tail risk

Historical VaR makes no distributional assumption; parametric assumes normality. The **gap between them is the fat tail** — where parametric is smaller, a normal model is understating how bad the bad days get.

| Horizon | Confidence | Historical VaR | Historical CVaR | Parametric VaR |
|---|---|---|---|---|
| 1 day | 95% | -1.38% | -2.30% | -1.42% |
| 1 day | 99% | -2.89% | -3.89% | -2.05% |
| 21 days | 95% | -5.64% | -8.37% | -5.10% |
| 21 days | 99% | -9.51% | -13.38% | -7.97% |

Daily skew **-0.93**, kurtosis **9.8** (normal = 3.0) — fat-tailed and negatively skewed, the usual equity shape. Worst day -7.91%, best +5.35%.

### Full system 50/25/25

Annualised alpha **+10.47%** (t = 3.46), R² = 0.56.

| Factor | Beta | t-stat |
|---|---|---|
| market | +0.509 | 31.10 |
| size | +0.052 | 1.99 |
| momentum | +0.386 | 23.42 |
| lowvol | -0.247 | -17.02 |

### Equity sleeve only

Annualised alpha **+4.42%** (t = 1.19), R² = 0.71.

| Factor | Beta | t-stat |
|---|---|---|
| market | +0.757 | 37.57 |
| size | +0.050 | 1.56 |
| momentum | +0.633 | 31.16 |
| lowvol | -0.535 | -29.97 |

Factors are long/short terciles rebuilt every 21 bars from the same point-in-time universe (size = small minus big by turnover; momentum and low-volatility from the engine's own causal definitions). A high R² with a large momentum beta would mean the book is simply a momentum index; a surviving positive alpha means the ranking adds something the raw factors do not.

## 3. Drawdown attribution

Each drawdown worse than −5%, decomposed into the rupee P&L each sleeve contributed while it was happening. This is what makes the multi-asset structure load-bearing rather than decorative.

| Peak | Trough | Depth | Days | Equity | Gold | US |
|---|---|---|---|---|---|---|
| 2020-02-28 | 2020-03-23 | -22.2% | 95 | -14.3% | -1.8% | -5.2% |
| 2022-04-25 | 2022-06-20 | -20.5% | 385 | -14.5% | -0.9% | -4.2% |
| 2018-02-02 | 2018-12-26 | -17.6% | 689 | -10.1% | +0.8% | -5.6% |
| 2025-01-10 | 2025-04-07 | -15.4% | 192 | -9.7% | +3.3% | -5.2% |
| 2026-02-02 | 2026-03-23 | -12.7% | 72 | -3.9% | -4.1% | -1.1% |

## 4. Stress tests

| Scenario | System | Nifty50 TRI |
|---|---|---|
| 2018 NBFC / IL&FS | -9.9% | -4.0% |
| COVID crash 2020 | +3.5% | -16.3% |
| 2022 rate shock | -12.8% | -10.0% |
| 2024-25 correction | -2.2% | -6.5% |

Worst rolling 1-year return **-15.8%** (ending 2019-01-30); **12%** of rolling 1-year windows were negative. A holder must be able to sit through both.

