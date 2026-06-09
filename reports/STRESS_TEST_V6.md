# MARK5 V2 Baseline (v6 Models) — Comprehensive Stress Test Report
**Date:** 2026-05-24  |  **System:** V2 Baseline (+15.85% net annual OOS)
**Capital:** ₹5 crore  |  **Costs:** 0.29% round-trip + 0.10% slippage
**STCG Tax:** 20% (India Budget 2024)  |  **Paper mode only**

---

## About This Report

Tests the V2 Baseline framework (best proven system: +15.85% net annual after STCG)
across 7 distinct market regimes: 3 real historical windows + 4 synthetic stress scenarios.

**Data transparency:**
- Periods A & B use IN-SAMPLE data (v5 ML models trained 2015-2021). Results inflated.
- Period C (2022-2026) is TRUE OOS — the only unbiased real-world estimate.
- Synthetic scenarios S1-S4 use simulated price data. Labels clearly marked.

**Verification:** Every trade below contains exact date, price, quantity, and P&L.
Cross-check against JSON: `reports/multi_strategy_backtest_v6.json`

---

## Results Summary — All Periods

| Period | Type | Net Annual | WR | Max DD | Sharpe | Calmar | Trades |
|--------|:----:|:----------:|:--:|:------:|:------:|:------:|:------:|
| Period A — Demonetization Era (IS) | 🔬IS | -60.9% | 22% | -99.7% | -0.70 | -0.76 | 37 |
| Period B — COVID Crash & Recovery (IS) | 🔬IS | +51.1% | 50% | -22.3% | 1.91 | 2.86 | 24 |
| Period C — Russia/Ukraine OOS (OOS) | ✅OOS | +12.7% | 50% | -22.3% | 0.65 | 0.71 | 30 |
| S1 — GFC 2008-style (-65% over 12mo) | 🧪SIM | +19.3% | 30% | -66.4% | 0.65 | 0.36 | 40 |
| S2 — Flash Crash (-20% in 10 days) | 🧪SIM | +22.8% | 58% | -15.6% | 1.73 | 1.83 | 12 |
| S3 — Prolonged Bear (-40% over 30mo) | 🧪SIM | -37.6% | 6% | -88.5% | -4.23 | -0.53 | 51 |
| S4 — High-VIX Chop (sideways 24mo) | 🧪SIM | +60.0% | 39% | -14.9% | 2.47 | 5.03 | 28 |

**Type key:** 🔬IS = in-sample (model trained on this data, inflated), 
✅OOS = out-of-sample (valid), 🧪SIM = synthetic simulation

---

## Period A — Demonetization Era (IS)
**Period:** 2015-04-01 → 2019-03-31 (4.0 years)  |  **Data type:** REAL_IS

> ⚠️ **IN-SAMPLE WARNING**: The ML models used here were trained on this data
> period. Win rates and returns are inflated. Use only to understand system
> behavior in this market regime, not as a performance forecast.

### Key Metrics

| Metric | Value | Verification Formula |
|--------|:-----:|---------------------|
| Net Annual (STCG) | **-60.86%** | CAGR × 0.80 (20% STCG) |
| Gross CAGR | -76.07% | (final/initial)^(1/years) - 1 |
| Total Return | -99.67% | (final_eq - 5cr) / 5cr |
| Win Rate | 21.6% | count(pnl>0) / total_trades |
| Max Drawdown | -99.70% | min(eq/cummax(eq) - 1) |
| Sharpe Ratio | -0.700 | (mean(ret) - 6.5%/252) / std × √252 |
| Calmar Ratio | -0.763 | CAGR / |max_dd| (>1.0 = good) |
| Total Trades | 37 | count(trade records) |
| Avg Hold Days | 143 days | mean(exit_date - entry_date) |
| Expected Value | -2.118%/trade | WR×avg_win - (1-WR)×avg_loss |

### Annual Returns

| Year | Return | Signal |
|------|:------:|--------|
| 2015 | -11.7% | 🔴 Loss |
| 2016 | -19.7% | 🔴 Loss |
| 2017 | +29.4% | ✅ Strong |
| 2018 | -11.6% | 🔴 Loss |
| 2019 | -99.6% | 🔴 Loss |

### Equity Curve
![Equity Curve](equity_Period_A_—_Demonetization_Era_(IS).png)

### Complete Trade Log

Every single trade — date, price, shares, P&L — for manual verification.

| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |
|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|
| 1 | **LUPIN** | 2015-04-01 | Wed | ₹1919.37 | 6,512 | ₹125.35L | 2015-04-17 | Fri | ₹1664.09 | 16d | ₹-17.30L | -13.8% | TRAIL_STOP_15% | 0.8006 | ₹1968.36 | 2015-04-07 | ❌ |
| 2 | **TATAELXSI** | 2015-04-01 | Wed | ₹563.38 | 21,557 | ₹121.80L | 2015-04-20 | Mon | ₹509.16 | 19d | ₹-12.36L | -10.2% | TRAIL_STOP_15% | 0.6999 | ₹602.25 | 2015-04-08 | ❌ |
| 4 | **RELIANCE** | 2015-04-01 | Wed | ₹179.08 | 69,799 | ₹125.36L | 2015-08-24 | Mon | ₹179.57 | 145d | ₹-0.39L | -0.3% | TRAIL_STOP_15% | 0.7030 | ₹227.60 | 2015-07-22 | ❌ |
| 7 | **LT** | 2015-04-01 | Wed | ₹970.39 | 12,881 | ₹125.36L | 2015-08-26 | Wed | ₹889.13 | 147d | ₹-11.16L | -8.9% | TRAIL_STOP_15% | 0.7055 | ₹1048.88 | 2015-07-10 | ❌ |
| 3 | **LUPIN** | 2015-04-22 | Wed | ₹1649.01 | 7,580 | ₹125.36L | 2015-07-24 | Fri | ₹1578.27 | 93d | ₹-6.07L | -4.8% | TRAIL_STOP_15% | 0.7956 | ₹1869.11 | 2015-07-20 | ❌ |
| 6 | **COFORGE** | 2015-04-22 | Wed | ₹63.26 | 146,040 | ₹92.65L | 2015-08-25 | Tue | ₹75.55 | 125d | ₹+17.37L | +18.8% | TRAIL_STOP_15% | 0.7574 | ₹92.33 | 2015-08-12 | ✅ |
| 5 | **BHARTIARTL** | 2015-08-05 | Wed | ₹354.82 | 33,393 | ₹118.83L | 2015-08-24 | Mon | ₹297.04 | 19d | ₹-19.93L | -16.8% | TRAIL_STOP_15% | 0.8251 | ₹354.82 | 2015-08-05 | ❌ |
| 8 | **BEL** | 2015-08-26 | Wed | ₹28.24 | 257,152 | ₹72.82L | 2015-09-16 | Wed | ₹27.05 | 21d | ₹-3.47L | -4.8% | ML_EXIT(rc=0.385) | 0.6637 | ₹29.13 | 2015-08-28 | ❌ |
| 9 | **LT** | 2015-08-26 | Wed | ₹890.91 | 14,030 | ₹125.36L | 2015-11-04 | Wed | ₹771.81 | 70d | ₹-17.39L | -13.9% | TRAIL_STOP_15% | 0.7730 | ₹912.21 | 2015-09-10 | ❌ |
| 10 | **LUPIN** | 2015-08-26 | Wed | ₹1712.23 | 7,300 | ₹125.36L | 2015-11-10 | Tue | ₹1689.93 | 76d | ₹-2.35L | -1.9% | TRAIL_STOP_15% | 0.8125 | ₹1992.51 | 2015-10-01 | ❌ |
| 11 | **BHARTIARTL** | 2015-08-26 | Wed | ₹289.86 | 43,124 | ₹125.36L | 2015-12-09 | Wed | ₹265.10 | 105d | ₹-11.37L | -9.1% | TRAIL_STOP_15% | 0.8541 | ₹317.14 | 2015-10-21 | ❌ |
| 12 | **TATAELXSI** | 2015-09-16 | Wed | ₹838.55 | 8,253 | ₹69.41L | 2016-01-13 | Wed | ₹868.79 | 119d | ₹+2.09L | +3.0% | TRAIL_STOP_15% | 0.6317 | ₹1027.50 | 2015-12-23 | ✅ |
| 14 | **RELIANCE** | 2015-11-18 | Wed | ₹197.81 | 52,930 | ₹105.01L | 2016-02-12 | Fri | ₹196.24 | 86d | ₹-1.44L | -1.4% | TRAIL_STOP_15% | 0.6489 | ₹233.34 | 2016-01-13 | ❌ |
| 15 | **LUPIN** | 2015-11-18 | Wed | ₹1686.36 | 7,412 | ₹125.36L | 2016-03-18 | Fri | ₹1467.18 | 121d | ₹-16.92L | -13.5% | TRAIL_STOP_15% | 0.8038 | ₹1800.52 | 2016-02-09 | ❌ |
| 13 | **LT** | 2015-12-09 | Wed | ₹735.25 | 15,437 | ₹113.83L | 2016-01-15 | Fri | ₹625.02 | 37d | ₹-17.63L | -15.5% | TRAIL_STOP_15% | 0.7727 | ₹740.31 | 2015-12-10 | ❌ |
| 18 | **BHARTIARTL** | 2016-01-20 | Wed | ₹264.56 | 47,247 | ₹125.36L | 2016-09-01 | Thu | ₹266.14 | 225d | ₹+0.02L | +0.0% | TRAIL_STOP_15% | 0.8029 | ₹323.46 | 2016-07-15 | ✅ |
| 21 | **LT** | 2016-01-20 | Wed | ₹630.90 | 6,886 | ₹43.57L | 2016-11-22 | Tue | ₹759.28 | 307d | ₹+8.56L | +19.6% | TRAIL_STOP_15% | 0.7340 | ₹904.75 | 2016-07-27 | ✅ |
| 16 | **TATAELXSI** | 2016-03-02 | Wed | ₹818.46 | 12,492 | ₹102.54L | 2016-06-24 | Fri | ₹769.28 | 114d | ₹-6.72L | -6.5% | TRAIL_STOP_15% | 0.6702 | ₹918.55 | 2016-04-27 | ❌ |
| 19 | **LUPIN** | 2016-03-23 | Wed | ₹1441.50 | 7,493 | ₹108.33L | 2016-09-29 | Thu | ₹1392.04 | 190d | ₹-4.32L | -4.0% | TRAIL_STOP_15% | 0.8102 | ₹1651.14 | 2016-07-29 | ❌ |
| 17 | **BEL** | 2016-07-07 | Thu | ₹32.03 | 298,940 | ₹96.04L | 2016-08-18 | Thu | ₹31.23 | 42d | ₹-2.96L | -3.1% | ML_EXIT(rc=0.444) | 0.6675 | ₹32.23 | 2016-07-14 | ❌ |
| 20 | **PERSISTENT** | 2016-08-18 | Thu | ₹298.60 | 31,100 | ₹93.13L | 2016-11-21 | Mon | ₹265.00 | 95d | ₹-10.96L | -11.8% | TRAIL_STOP_15% | 0.6458 | ₹312.50 | 2016-10-05 | ❌ |
| 27 | **BHARTIARTL** | 2016-09-08 | Thu | ₹278.30 | 44,723 | ₹124.83L | 2018-01-24 | Wed | ₹394.41 | 503d | ₹+51.05L | +40.9% | TRAIL_STOP_15% | 0.8430 | ₹466.72 | 2017-11-02 | ✅ |
| 22 | **LUPIN** | 2016-09-29 | Thu | ₹1394.83 | 7,460 | ₹104.36L | 2017-05-02 | Tue | ₹1236.09 | 215d | ₹-12.41L | -11.9% | TRAIL_STOP_15% | 0.8102 | ₹1457.54 | 2016-12-05 | ❌ |
| 24 | **COFORGE** | 2016-12-01 | Thu | ₹75.55 | 13,979 | ₹10.59L | 2017-07-20 | Thu | ₹96.69 | 231d | ₹+2.89L | +27.2% | ML_EXIT(rc=0.329) | 0.6422 | ₹104.49 | 2017-07-11 | ✅ |
| 31 | **LT** | 2016-12-01 | Thu | ₹794.45 | 15,734 | ₹125.36L | 2018-06-27 | Wed | ₹1081.99 | 573d | ₹+44.39L | +35.4% | TRAIL_STOP_15% | 0.7216 | ₹1278.87 | 2018-02-01 | ✅ |
| 23 | **LUPIN** | 2017-05-18 | Thu | ₹1240.51 | 7,278 | ₹90.55L | 2017-05-26 | Fri | ₹1054.77 | 8d | ₹-14.00L | -15.5% | TRAIL_STOP_15% | 0.8123 | ₹1250.99 | 2017-05-19 | ❌ |
| 25 | **LUPIN** | 2017-06-08 | Thu | ₹1109.27 | 6,902 | ₹76.78L | 2017-08-03 | Thu | ₹948.03 | 56d | ₹-11.54L | -15.0% | TRAIL_STOP_15% | 0.7728 | ₹1122.07 | 2017-06-15 | ❌ |
| 33 | **RELIANCE** | 2017-07-20 | Thu | ₹338.02 | 4,296 | ₹14.56L | 2018-10-05 | Fri | ₹465.94 | 442d | ₹+5.40L | +37.0% | TRAIL_STOP_15% | 0.6451 | ₹586.54 | 2018-08-28 | ✅ |
| 26 | **LUPIN** | 2017-08-10 | Thu | ₹895.71 | 7,166 | ₹64.37L | 2017-11-07 | Tue | ₹820.63 | 89d | ₹-5.74L | -8.9% | TRAIL_STOP_15% | 0.8237 | ₹1022.44 | 2017-10-16 | ❌ |
| 28 | **LUPIN** | 2017-11-23 | Thu | ₹791.14 | 7,402 | ₹58.73L | 2018-02-06 | Tue | ₹764.32 | 75d | ₹-2.32L | -4.0% | TRAIL_STOP_15% | 0.8047 | ₹916.16 | 2018-01-24 | ❌ |
| 29 | **BHARTIARTL** | 2018-01-25 | Thu | ₹389.35 | 32,104 | ₹125.36L | 2018-04-09 | Mon | ₹328.55 | 74d | ₹-20.19L | -16.1% | TRAIL_STOP_15% | 0.8197 | ₹389.35 | 2018-01-25 | ❌ |
| 34 | **LUPIN** | 2018-02-15 | Thu | ₹786.31 | 13,455 | ₹106.11L | 2018-12-10 | Mon | ₹780.42 | 298d | ₹-1.40L | -1.3% | TRAIL_STOP_15% | 0.8353 | ₹931.14 | 2018-09-14 | ❌ |
| 30 | **BHARTIARTL** | 2018-04-23 | Mon | ₹348.72 | 30,077 | ₹105.19L | 2018-05-23 | Wed | ₹308.24 | 30d | ₹-12.75L | -12.1% | TRAIL_STOP_15% | 0.7712 | ₹363.08 | 2018-04-25 | ❌ |
| 32 | **BHARTIARTL** | 2018-06-04 | Mon | ₹322.28 | 28,669 | ₹92.66L | 2018-10-01 | Mon | ₹284.13 | 119d | ₹-11.44L | -12.3% | TRAIL_STOP_15% | 0.8596 | ₹339.12 | 2018-09-07 | ❌ |
| 36 | **LT** | 2018-07-16 | Mon | ₹1128.44 | 11,077 | ₹125.36L | 2019-02-14 | Thu | ₹1091.23 | 213d | ₹-4.83L | -3.9% | TRAIL_STOP_15% | 0.6624 | ₹1290.93 | 2018-12-20 | ❌ |
| 35 | **TATAELXSI** | 2018-10-08 | Mon | ₹930.60 | 2,298 | ₹21.45L | 2019-01-25 | Fri | ₹841.90 | 109d | ₹-2.16L | -10.1% | TRAIL_STOP_15% | 0.6637 | ₹992.31 | 2018-11-01 | ❌ |
| 37 | **LUPIN** | 2018-12-10 | Mon | ₹781.98 | 13,168 | ₹103.27L | 2019-03-22 | Fri | ₹713.63 | 102d | ₹-9.57L | -9.3% | TRAIL_STOP_15% | 0.8122 | ₹848.82 | 2019-02-01 | ❌ |

### Per-Ticker Summary

| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |
|--------|:------:|:--:|:--------:|----------:|:------------:|
| COFORGE | 2 | 100% | +23.0% | ₹+20.26L 📈 | [COFORGE chart](ticker_COFORGE_Period_A_—_Demonetization_Era_(IS).png) |
| RELIANCE | 3 | 33% | +11.8% | ₹+3.57L 📈 | [RELIANCE chart](ticker_RELIANCE_Period_A_—_Demonetization_Era_(IS).png) |
| LT | 6 | 33% | +2.2% | ₹+1.94L 📈 | [LT chart](ticker_LT_Period_A_—_Demonetization_Era_(IS).png) |
| BEL | 2 | 0% | -3.9% | ₹-6.43L 📉 | [BEL chart](ticker_BEL_Period_A_—_Demonetization_Era_(IS).png) |
| PERSISTENT | 1 | 0% | -11.8% | ₹-10.96L 📉 | [PERSISTENT chart](ticker_PERSISTENT_Period_A_—_Demonetization_Era_(IS).png) |
| TATAELXSI | 4 | 25% | -5.9% | ₹-19.15L 📉 | [TATAELXSI chart](ticker_TATAELXSI_Period_A_—_Demonetization_Era_(IS).png) |
| BHARTIARTL | 7 | 29% | -3.6% | ₹-24.60L 📉 | [BHARTIARTL chart](ticker_BHARTIARTL_Period_A_—_Demonetization_Era_(IS).png) |
| LUPIN | 12 | 0% | -8.7% | ₹-103.95L 📉 | [LUPIN chart](ticker_LUPIN_Period_A_—_Demonetization_Era_(IS).png) |

---

## Period B — COVID Crash & Recovery (IS)
**Period:** 2019-04-01 → 2021-12-31 (2.8 years)  |  **Data type:** REAL_IS

> ⚠️ **IN-SAMPLE WARNING**: The ML models used here were trained on this data
> period. Win rates and returns are inflated. Use only to understand system
> behavior in this market regime, not as a performance forecast.

### Key Metrics

| Metric | Value | Verification Formula |
|--------|:-----:|---------------------|
| Net Annual (STCG) | **+51.13%** | CAGR × 0.80 (20% STCG) |
| Gross CAGR | +63.92% | (final/initial)^(1/years) - 1 |
| Total Return | +289.54% | (final_eq - 5cr) / 5cr |
| Win Rate | 50.0% | count(pnl>0) / total_trades |
| Max Drawdown | -22.32% | min(eq/cummax(eq) - 1) |
| Sharpe Ratio | 1.912 | (mean(ret) - 6.5%/252) / std × √252 |
| Calmar Ratio | 2.863 | CAGR / |max_dd| (>1.0 = good) |
| Total Trades | 24 | count(trade records) |
| Avg Hold Days | 160 days | mean(exit_date - entry_date) |
| Expected Value | +48.232%/trade | WR×avg_win - (1-WR)×avg_loss |

### Annual Returns

| Year | Return | Signal |
|------|:------:|--------|
| 2019 | +10.6% | 🟡 Mild |
| 2020 | +58.7% | ✅ Strong |
| 2021 | +121.9% | ✅ Strong |

### Equity Curve
![Equity Curve](equity_Period_B_—_COVID_Crash_&_Recovery_(IS).png)

### Complete Trade Log

Every single trade — date, price, shares, P&L — for manual verification.

| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |
|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|
| 1 | **TATAELXSI** | 2019-04-01 | Mon | ₹887.90 | 13,677 | ₹121.79L | 2019-05-14 | Tue | ₹773.60 | 43d | ₹-16.29L | -13.4% | TRAIL_STOP_15% | 0.6845 | ₹915.73 | 2019-04-15 | ❌ |
| 2 | **LUPIN** | 2019-04-01 | Mon | ₹737.19 | 16,956 | ₹125.36L | 2019-05-27 | Mon | ₹713.73 | 56d | ₹-4.69L | -3.7% | TRAIL_STOP_15% | 0.7855 | ₹843.15 | 2019-05-02 | ❌ |
| 3 | **RELIANCE** | 2019-04-01 | Mon | ₹619.55 | 20,175 | ₹125.36L | 2019-07-30 | Tue | ₹524.60 | 120d | ₹-19.83L | -15.8% | TRAIL_STOP_15% | 0.7899 | ₹626.49 | 2019-05-03 | ❌ |
| 9 | **BHARTIARTL** | 2019-04-01 | Mon | ₹299.42 | 41,747 | ₹125.36L | 2020-03-12 | Thu | ₹444.01 | 346d | ₹+59.46L | +47.4% | TRAIL_STOP_15% | 0.7624 | ₹540.09 | 2020-02-14 | ✅ |
| 4 | **LT** | 2019-06-03 | Mon | ₹1388.95 | 7,263 | ₹101.17L | 2019-08-13 | Tue | ₹1182.47 | 71d | ₹-15.54L | -15.4% | TRAIL_STOP_15% | 0.8068 | ₹1404.15 | 2019-07-03 | ❌ |
| 6 | **LUPIN** | 2019-06-03 | Mon | ₹717.03 | 17,432 | ₹125.36L | 2020-02-24 | Mon | ₹650.26 | 266d | ₹-12.33L | -9.8% | TRAIL_STOP_15% | 0.8437 | ₹778.13 | 2019-11-28 | ❌ |
| 7 | **TATAELXSI** | 2019-08-05 | Mon | ₹567.27 | 18,537 | ₹105.46L | 2020-02-28 | Fri | ₹826.73 | 207d | ₹+47.35L | +44.9% | TRAIL_STOP_15% | 0.7140 | ₹1021.00 | 2020-02-13 | ✅ |
| 5 | **COFORGE** | 2019-08-26 | Mon | ₹257.47 | 33,295 | ₹85.97L | 2019-10-07 | Mon | ₹254.69 | 42d | ₹-1.42L | -1.6% | ML_EXIT(rc=0.272) | 0.7802 | ₹280.50 | 2019-08-28 | ❌ |
| 8 | **RELIANCE** | 2019-10-07 | Mon | ₹586.66 | 14,375 | ₹84.58L | 2020-02-28 | Fri | ₹593.78 | 144d | ₹+0.53L | +0.6% | TRAIL_STOP_15% | 0.8285 | ₹720.22 | 2019-12-19 | ✅ |
| 10 | **LT** | 2020-03-03 | Tue | ₹1065.79 | 11,728 | ₹125.36L | 2020-03-16 | Mon | ₹872.23 | 13d | ₹-23.36L | -18.6% | TRAIL_STOP_15% | 0.7190 | ₹1065.79 | 2020-03-03 | ❌ |
| 11 | **PERSISTENT** | 2020-03-03 | Tue | ₹337.25 | 29,552 | ₹99.95L | 2020-03-17 | Tue | ₹282.86 | 14d | ₹-16.61L | -16.6% | TRAIL_STOP_15% | 0.6641 | ₹337.25 | 2020-03-03 | ❌ |
| 12 | **LUPIN** | 2020-03-03 | Tue | ₹629.05 | 19,871 | ₹125.36L | 2020-03-25 | Wed | ₹547.48 | 22d | ₹-16.89L | -13.5% | TRAIL_STOP_15% | 0.8074 | ₹654.49 | 2020-03-05 | ❌ |
| 13 | **BHARTIARTL** | 2020-03-24 | Tue | ₹386.62 | 32,331 | ₹125.36L | 2020-09-08 | Tue | ₹477.37 | 168d | ₹+28.53L | +22.8% | TRAIL_STOP_15% | 0.8571 | ₹572.40 | 2020-05-19 | ✅ |
| 14 | **LT** | 2020-03-24 | Tue | ₹647.62 | 18,339 | ₹119.11L | 2020-09-22 | Tue | ₹788.83 | 182d | ₹+25.13L | +21.1% | TRAIL_STOP_15% | 0.6172 | ₹930.30 | 2020-08-19 | ✅ |
| 21 | **TATAELXSI** | 2020-03-24 | Tue | ₹516.31 | 24,210 | ₹125.36L | 2021-12-20 | Mon | ₹5247.75 | 636d | ₹+1141.43L | +910.5% | TRAIL_STOP_15% | 0.6819 | ₹6399.06 | 2021-11-15 | ✅ |
| 17 | **LUPIN** | 2020-04-15 | Wed | ₹784.77 | 13,805 | ₹108.65L | 2020-10-30 | Fri | ₹884.46 | 198d | ₹+13.09L | +12.1% | TRAIL_STOP_15% | 0.7780 | ₹1057.35 | 2020-09-18 | ✅ |
| 15 | **BHARTIARTL** | 2020-09-09 | Wed | ₹484.24 | 25,813 | ₹125.36L | 2020-09-24 | Thu | ₹401.85 | 15d | ₹-21.93L | -17.5% | TRAIL_STOP_15% | 0.8173 | ₹484.24 | 2020-09-09 | ❌ |
| 16 | **COFORGE** | 2020-09-30 | Wed | ₹436.44 | 28,640 | ₹125.36L | 2020-10-26 | Mon | ₹414.93 | 26d | ₹-6.87L | -5.5% | TRAIL_STOP_15% | 0.7667 | ₹525.25 | 2020-10-13 | ❌ |
| 18 | **BHARTIARTL** | 2020-09-30 | Wed | ₹404.24 | 30,922 | ₹125.36L | 2021-03-25 | Thu | ₹486.62 | 176d | ₹+24.67L | +19.7% | TRAIL_STOP_15% | 0.8401 | ₹584.10 | 2021-02-03 | ✅ |
| 19 | **LT** | 2020-11-11 | Wed | ₹996.63 | 12,542 | ₹125.36L | 2021-04-19 | Mon | ₹1230.28 | 159d | ₹+28.49L | +22.7% | TRAIL_STOP_15% | 0.7344 | ₹1472.09 | 2021-02-09 | ✅ |
| 20 | **LUPIN** | 2020-11-11 | Wed | ₹888.03 | 14,076 | ₹125.36L | 2021-08-11 | Wed | ₹1027.80 | 273d | ₹+18.89L | +15.1% | TRAIL_STOP_15% | 0.8182 | ₹1214.45 | 2021-06-02 | ✅ |
| 22 | **PERSISTENT** | 2021-04-07 | Wed | ₹946.87 | 13,201 | ₹125.36L | 2021-12-31 | Fri | ₹2367.07 | 268d | ₹+186.21L | +148.5% | END_SIM | 0.6450 | ₹2369.44 | 2021-12-31 | ✅ |
| 23 | **BHARTIARTL** | 2021-04-28 | Wed | ₹522.70 | 23,914 | ₹125.36L | 2021-12-31 | Fri | ₹667.65 | 247d | ₹+33.84L | +27.0% | END_SIM | 0.7470 | ₹747.83 | 2021-11-25 | ✅ |
| 24 | **LUPIN** | 2021-08-12 | Thu | ₹956.54 | 13,067 | ₹125.35L | 2021-12-31 | Fri | ₹929.98 | 141d | ₹-4.19L | -3.3% | END_SIM | 0.8146 | ₹968.95 | 2021-09-07 | ❌ |

### Per-Ticker Summary

| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |
|--------|:------:|:--:|:--------:|----------:|:------------:|
| TATAELXSI | 3 | 67% | +314.0% | ₹+1172.49L 📈 | [TATAELXSI chart](ticker_TATAELXSI_Period_B_—_COVID_Crash_&_Recovery_(IS).png) |
| PERSISTENT | 2 | 50% | +66.0% | ₹+169.61L 📈 | [PERSISTENT chart](ticker_PERSISTENT_Period_B_—_COVID_Crash_&_Recovery_(IS).png) |
| BHARTIARTL | 5 | 80% | +19.9% | ₹+124.58L 📈 | [BHARTIARTL chart](ticker_BHARTIARTL_Period_B_—_COVID_Crash_&_Recovery_(IS).png) |
| LT | 4 | 50% | +2.5% | ₹+14.73L 📈 | [LT chart](ticker_LT_Period_B_—_COVID_Crash_&_Recovery_(IS).png) |
| LUPIN | 6 | 33% | -0.5% | ₹-6.11L 📉 | [LUPIN chart](ticker_LUPIN_Period_B_—_COVID_Crash_&_Recovery_(IS).png) |
| COFORGE | 2 | 0% | -3.6% | ₹-8.29L 📉 | [COFORGE chart](ticker_COFORGE_Period_B_—_COVID_Crash_&_Recovery_(IS).png) |
| RELIANCE | 2 | 50% | -7.6% | ₹-19.29L 📉 | [RELIANCE chart](ticker_RELIANCE_Period_B_—_COVID_Crash_&_Recovery_(IS).png) |

---

## Period C — Russia/Ukraine OOS (OOS)
**Period:** 2022-01-01 → 2026-05-20 (4.4 years)  |  **Data type:** REAL_OOS

### Key Metrics

| Metric | Value | Verification Formula |
|--------|:-----:|---------------------|
| Net Annual (STCG) | **+12.69%** | CAGR × 0.80 (20% STCG) |
| Gross CAGR | +15.86% | (final/initial)^(1/years) - 1 |
| Total Return | +90.57% | (final_eq - 5cr) / 5cr |
| Win Rate | 50.0% | count(pnl>0) / total_trades |
| Max Drawdown | -22.32% | min(eq/cummax(eq) - 1) |
| Sharpe Ratio | 0.652 | (mean(ret) - 6.5%/252) / std × √252 |
| Calmar Ratio | 0.711 | CAGR / |max_dd| (>1.0 = good) |
| Total Trades | 30 | count(trade records) |
| Avg Hold Days | 208 days | mean(exit_date - entry_date) |
| Expected Value | +15.511%/trade | WR×avg_win - (1-WR)×avg_loss |

### Annual Returns

| Year | Return | Signal |
|------|:------:|--------|
| 2022 | -8.4% | 🔴 Loss |
| 2023 | +54.3% | ✅ Strong |
| 2024 | +44.6% | ✅ Strong |
| 2025 | +0.6% | 🟡 Mild |
| 2026 | -7.3% | 🔴 Loss |

### Equity Curve
![Equity Curve](equity_Period_C_—_Russia_Ukraine_OOS_(OOS).png)

### Complete Trade Log

Every single trade — date, price, shares, P&L — for manual verification.

| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |
|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|
| 1 | **COFORGE** | 2022-01-03 | Mon | ₹1132.95 | 11,033 | ₹125.36L | 2022-01-24 | Mon | ₹913.34 | 21d | ₹-24.88L | -19.9% | TRAIL_STOP_15% | 0.7036 | ₹1156.88 | 2022-01-04 | ❌ |
| 2 | **LUPIN** | 2022-01-03 | Mon | ₹926.46 | 13,492 | ₹125.36L | 2022-02-07 | Mon | ₹787.12 | 35d | ₹-19.47L | -15.5% | TRAIL_STOP_15% | 0.7948 | ₹939.43 | 2022-01-19 | ❌ |
| 4 | **LT** | 2022-01-03 | Mon | ₹1829.31 | 6,638 | ₹121.78L | 2022-02-24 | Thu | ₹1668.52 | 52d | ₹-11.35L | -9.3% | TRAIL_STOP_15% | 0.7026 | ₹1966.76 | 2022-01-17 | ❌ |
| 9 | **BHARTIARTL** | 2022-01-03 | Mon | ₹676.32 | 18,482 | ₹125.36L | 2022-06-16 | Thu | ₹638.75 | 164d | ₹-7.65L | -6.1% | TRAIL_STOP_15% | 0.7601 | ₹758.09 | 2022-04-06 | ❌ |
| 3 | **PERSISTENT** | 2022-01-24 | Mon | ₹1966.45 | 5,113 | ₹100.84L | 2022-02-14 | Mon | ₹1912.74 | 21d | ₹-3.32L | -3.3% | ML_EXIT(rc=0.392) | 0.6054 | ₹2183.03 | 2022-02-01 | ❌ |
| 6 | **TATAELXSI** | 2022-02-14 | Mon | ₹6688.07 | 1,169 | ₹78.41L | 2022-05-02 | Mon | ₹7409.55 | 77d | ₹+7.96L | +10.2% | TRAIL_STOP_15% | 0.6738 | ₹8734.86 | 2022-03-28 | ✅ |
| 7 | **LUPIN** | 2022-02-14 | Mon | ₹746.21 | 16,751 | ₹125.36L | 2022-05-19 | Thu | ₹620.98 | 94d | ₹-21.64L | -17.3% | TRAIL_STOP_15% | 0.7938 | ₹774.06 | 2022-04-04 | ❌ |
| 5 | **COFORGE** | 2022-03-07 | Mon | ₹898.20 | 12,203 | ₹109.93L | 2022-04-18 | Mon | ₹772.90 | 42d | ₹-15.88L | -14.4% | TRAIL_STOP_15% | 0.6274 | ₹914.49 | 2022-03-08 | ❌ |
| 10 | **RELIANCE** | 2022-04-18 | Mon | ₹1158.64 | 8,116 | ₹94.31L | 2022-07-07 | Thu | ₹1085.53 | 80d | ₹-6.46L | -6.8% | TRAIL_STOP_15% | 0.6879 | ₹1283.07 | 2022-04-28 | ❌ |
| 8 | **BEL** | 2022-05-09 | Mon | ₹71.93 | 119,925 | ₹86.51L | 2022-05-30 | Mon | ₹73.43 | 21d | ₹+1.29L | +1.5% | ML_EXIT(rc=0.432) | 0.6886 | ₹74.87 | 2022-05-20 | ✅ |
| 14 | **LUPIN** | 2022-05-30 | Mon | ₹600.71 | 20,808 | ₹125.36L | 2023-02-23 | Thu | ₹647.98 | 269d | ₹+9.08L | +7.2% | TRAIL_STOP_15% | 0.8249 | ₹764.54 | 2022-12-02 | ✅ |
| 21 | **LT** | 2022-05-30 | Mon | ₹1580.05 | 4,197 | ₹66.51L | 2025-02-03 | Mon | ₹3224.06 | 980d | ₹+68.41L | +102.9% | TRAIL_STOP_15% | 0.7762 | ₹3873.00 | 2024-12-09 | ✅ |
| 11 | **TATAELXSI** | 2022-06-20 | Mon | ₹7056.23 | 1,651 | ₹116.84L | 2022-08-29 | Mon | ₹8610.75 | 70d | ₹+24.91L | +21.3% | TRAIL_STOP_15% | 0.6401 | ₹10381.99 | 2022-08-17 | ✅ |
| 25 | **BHARTIARTL** | 2022-07-11 | Mon | ₹646.24 | 13,630 | ₹88.34L | 2026-03-11 | Wed | ₹1805.19 | 1339d | ₹+157.00L | +177.7% | TRAIL_STOP_15% | 0.7807 | ₹2162.70 | 2025-11-21 | ✅ |
| 12 | **TATAELXSI** | 2022-09-12 | Mon | ₹8642.14 | 1,446 | ₹125.33L | 2022-10-18 | Tue | ₹6993.40 | 36d | ₹-24.50L | -19.6% | TRAIL_STOP_15% | 0.6417 | ₹8831.53 | 2022-09-13 | ❌ |
| 13 | **COFORGE** | 2022-10-24 | Mon | ₹742.53 | 15,679 | ₹116.76L | 2022-12-26 | Mon | ₹723.12 | 63d | ₹-3.71L | -3.2% | ML_EXIT(rc=0.360) | 0.8067 | ₹813.46 | 2022-12-02 | ❌ |
| 15 | **TATAELXSI** | 2022-12-26 | Mon | ₹6081.47 | 1,854 | ₹113.08L | 2024-01-25 | Thu | ₹7500.69 | 395d | ₹+25.58L | +22.6% | TRAIL_STOP_15% | 0.6582 | ₹8898.95 | 2023-12-18 | ✅ |
| 20 | **LUPIN** | 2023-02-27 | Mon | ₹645.58 | 19,362 | ₹125.36L | 2025-01-28 | Tue | ₹2007.84 | 701d | ₹+262.27L | +209.2% | TRAIL_STOP_15% | 0.8144 | ₹2381.29 | 2025-01-02 | ✅ |
| 16 | **BEL** | 2024-01-31 | Wed | ₹182.01 | 68,677 | ₹125.36L | 2024-03-13 | Wed | ₹186.65 | 42d | ₹+2.45L | +2.0% | ML_EXIT(rc=0.424) | 0.7051 | ₹211.29 | 2024-03-07 | ✅ |
| 17 | **RELIANCE** | 2024-03-13 | Wed | ₹1423.07 | 8,783 | ₹125.35L | 2024-09-19 | Thu | ₹1462.36 | 190d | ₹+2.72L | +2.2% | ML_EXIT(rc=0.424) | 0.6864 | ₹1589.14 | 2024-07-08 | ✅ |
| 18 | **COFORGE** | 2024-09-19 | Thu | ₹1365.08 | 9,157 | ₹125.36L | 2025-01-02 | Thu | ₹1915.20 | 105d | ₹+49.50L | +39.5% | ML_EXIT(rc=0.329) | 0.7769 | ₹1941.10 | 2024-12-30 | ✅ |
| 19 | **BEL** | 2025-01-02 | Thu | ₹293.53 | 42,585 | ₹125.36L | 2025-01-23 | Thu | ₹270.39 | 21d | ₹-10.55L | -8.4% | ML_EXIT(rc=0.392) | 0.6742 | ₹293.53 | 2025-01-02 | ❌ |
| 24 | **RELIANCE** | 2025-01-23 | Thu | ₹1259.88 | 9,921 | ₹125.36L | 2026-03-04 | Wed | ₹1343.65 | 405d | ₹+7.56L | +6.0% | TRAIL_STOP_15% | 0.7269 | ₹1592.30 | 2026-01-02 | ✅ |
| 22 | **TATAELXSI** | 2025-02-13 | Thu | ₹6071.80 | 2,058 | ₹125.32L | 2025-03-13 | Thu | ₹5173.00 | 28d | ₹-19.17L | -15.3% | TRAIL_STOP_15% | 0.6283 | ₹6104.94 | 2025-02-14 | ❌ |
| 27 | **LUPIN** | 2025-02-13 | Thu | ₹2044.59 | 6,113 | ₹125.35L | 2026-05-20 | Wed | ₹2283.21 | 461d | ₹+13.82L | +11.0% | END_SIM | 0.8214 | ₹2460.10 | 2026-05-07 | ✅ |
| 23 | **COFORGE** | 2025-03-27 | Thu | ₹1608.05 | 7,773 | ₹125.36L | 2025-04-04 | Fri | ₹1307.98 | 8d | ₹-23.98L | -19.1% | TRAIL_STOP_15% | 0.7002 | ₹1608.05 | 2025-03-27 | ❌ |
| 26 | **LT** | 2025-04-17 | Thu | ₹3189.36 | 3,919 | ₹125.35L | 2026-03-12 | Thu | ₹3679.84 | 329d | ₹+18.44L | +14.7% | TRAIL_STOP_15% | 0.7357 | ₹4375.36 | 2026-02-23 | ✅ |
| 28 | **BHARTIARTL** | 2026-03-20 | Fri | ₹1847.95 | 6,764 | ₹125.36L | 2026-05-20 | Wed | ₹1903.00 | 61d | ₹+2.99L | +2.4% | END_SIM | 0.8454 | ₹1938.10 | 2026-05-18 | ✅ |
| 29 | **TATAELXSI** | 2026-03-20 | Fri | ₹4241.94 | 2,946 | ₹125.33L | 2026-05-20 | Wed | ₹4170.92 | 61d | ₹-2.81L | -2.2% | END_SIM | 0.6474 | ₹4650.70 | 2026-04-21 | ❌ |
| 30 | **RELIANCE** | 2026-03-20 | Fri | ₹1415.81 | 8,828 | ₹125.35L | 2026-05-20 | Wed | ₹1358.34 | 61d | ₹-5.78L | -4.6% | END_SIM | 0.6396 | ₹1463.60 | 2026-05-05 | ❌ |

### Per-Ticker Summary

| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |
|--------|:------:|:--:|:--------:|----------:|:------------:|
| LUPIN | 5 | 60% | +38.9% | ₹+244.06L 📈 | [LUPIN chart](ticker_LUPIN_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |
| BHARTIARTL | 3 | 67% | +58.0% | ₹+152.33L 📈 | [BHARTIARTL chart](ticker_BHARTIARTL_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |
| LT | 3 | 67% | +36.1% | ₹+75.51L 📈 | [LT chart](ticker_LT_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |
| TATAELXSI | 6 | 50% | +2.8% | ₹+11.98L 📈 | [TATAELXSI chart](ticker_TATAELXSI_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |
| RELIANCE | 4 | 50% | -0.8% | ₹-1.97L 📉 | [RELIANCE chart](ticker_RELIANCE_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |
| PERSISTENT | 1 | 0% | -3.3% | ₹-3.32L 📉 | [PERSISTENT chart](ticker_PERSISTENT_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |
| BEL | 3 | 67% | -1.7% | ₹-6.81L 📉 | [BEL chart](ticker_BEL_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |
| COFORGE | 5 | 20% | -3.4% | ₹-18.95L 📉 | [COFORGE chart](ticker_COFORGE_Period_C_—_Russia_Ukraine_OOS_(OOS).png) |

---

## S1 — GFC 2008-style (-65% over 12mo)
**Period:** 2022-01-03 → 2024-04-05 (2.3 years)  |  **Data type:** SYNTHETIC

### Key Metrics

| Metric | Value | Verification Formula |
|--------|:-----:|---------------------|
| Net Annual (STCG) | **+19.29%** | CAGR × 0.80 (20% STCG) |
| Gross CAGR | +24.12% | (final/initial)^(1/years) - 1 |
| Total Return | +62.71% | (final_eq - 5cr) / 5cr |
| Win Rate | 30.0% | count(pnl>0) / total_trades |
| Max Drawdown | -66.36% | min(eq/cummax(eq) - 1) |
| Sharpe Ratio | 0.652 | (mean(ret) - 6.5%/252) / std × √252 |
| Calmar Ratio | 0.363 | CAGR / |max_dd| (>1.0 = good) |
| Total Trades | 40 | count(trade records) |
| Avg Hold Days | 74 days | mean(exit_date - entry_date) |
| Expected Value | +23.251%/trade | WR×avg_win - (1-WR)×avg_loss |

### Annual Returns

| Year | Return | Signal |
|------|:------:|--------|
| 2022 | -60.8% | 🔴 Loss |
| 2023 | +226.9% | ✅ Strong |
| 2024 | +27.1% | ✅ Strong |

### Equity Curve
![Equity Curve](equity_S1_—_GFC_2008-style_(-65%_over_12mo).png)

### Complete Trade Log

Every single trade — date, price, shares, P&L — for manual verification.

| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |
|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|
| 1 | **COFORGE** | 2022-01-03 | Mon | ₹62.60 | 199,682 | ₹125.36L | 2022-01-24 | Mon | ₹65.13 | 21d | ₹+4.31L | +3.4% | ML_EXIT(rc=0.298) | 0.8094 | ₹65.19 | 2022-01-24 | ✅ |
| 2 | **BEL** | 2022-01-03 | Mon | ₹24.48 | 496,160 | ₹121.80L | 2022-03-07 | Mon | ₹29.84 | 63d | ₹+25.83L | +21.2% | ML_EXIT(rc=0.300) | 0.6456 | ₹29.87 | 2022-03-07 | ✅ |
| 3 | **LT** | 2022-01-03 | Mon | ₹840.97 | 14,863 | ₹125.36L | 2022-04-11 | Mon | ₹948.07 | 98d | ₹+15.15L | +12.1% | TRAIL_STOP_15% | 0.7140 | ₹1118.77 | 2022-03-30 | ✅ |
| 4 | **LUPIN** | 2022-01-03 | Mon | ₹1346.34 | 9,284 | ₹125.36L | 2022-04-18 | Mon | ₹1173.30 | 105d | ₹-16.74L | -13.4% | TRAIL_STOP_15% | 0.8012 | ₹1387.63 | 2022-01-07 | ❌ |
| 5 | **RELIANCE** | 2022-01-24 | Mon | ₹188.10 | 66,452 | ₹125.36L | 2022-04-20 | Wed | ₹178.29 | 86d | ₹-7.23L | -5.8% | TRAIL_STOP_15% | 0.7030 | ₹211.67 | 2022-03-28 | ❌ |
| 7 | **COFORGE** | 2022-03-07 | Mon | ₹70.25 | 177,937 | ₹125.36L | 2022-05-26 | Thu | ₹61.46 | 80d | ₹-16.32L | -13.0% | TRAIL_STOP_15% | 0.7444 | ₹73.00 | 2022-03-29 | ❌ |
| 6 | **LUPIN** | 2022-04-18 | Mon | ₹1175.65 | 10,632 | ₹125.36L | 2022-04-29 | Fri | ₹950.59 | 11d | ₹-24.58L | -19.6% | TRAIL_STOP_15% | 0.8189 | ₹1175.65 | 2022-04-18 | ❌ |
| 12 | **LT** | 2022-04-18 | Mon | ₹936.44 | 13,348 | ₹125.36L | 2022-09-15 | Thu | ₹1051.50 | 150d | ₹+14.59L | +11.6% | TRAIL_STOP_15% | 0.7706 | ₹1311.49 | 2022-08-02 | ✅ |
| 8 | **LUPIN** | 2022-05-09 | Mon | ₹817.58 | 15,288 | ₹125.35L | 2022-06-24 | Fri | ₹703.63 | 46d | ₹-18.10L | -14.4% | TRAIL_STOP_15% | 0.7933 | ₹840.54 | 2022-05-10 | ❌ |
| 10 | **TATAELXSI** | 2022-05-09 | Mon | ₹254.49 | 46,465 | ₹118.59L | 2022-08-09 | Tue | ₹235.61 | 92d | ₹-9.44L | -8.0% | TRAIL_STOP_15% | 0.6608 | ₹279.65 | 2022-07-06 | ❌ |
| 9 | **BHARTIARTL** | 2022-05-30 | Mon | ₹282.78 | 38,506 | ₹109.20L | 2022-07-26 | Tue | ₹241.32 | 57d | ₹-16.55L | -15.2% | TRAIL_STOP_15% | 0.8413 | ₹301.55 | 2022-06-13 | ❌ |
| 11 | **LUPIN** | 2022-07-11 | Mon | ₹632.36 | 16,918 | ₹107.29L | 2022-09-08 | Thu | ₹539.93 | 59d | ₹-16.21L | -15.1% | TRAIL_STOP_15% | 0.8219 | ₹636.90 | 2022-08-02 | ❌ |
| 13 | **BHARTIARTL** | 2022-08-01 | Mon | ₹229.65 | 40,337 | ₹92.90L | 2022-09-20 | Tue | ₹211.47 | 50d | ₹-7.85L | -8.4% | TRAIL_STOP_15% | 0.7708 | ₹260.75 | 2022-09-06 | ❌ |
| 14 | **RELIANCE** | 2022-08-22 | Mon | ₹153.13 | 70,897 | ₹108.88L | 2022-09-26 | Mon | ₹129.95 | 35d | ₹-17.01L | -15.6% | TRAIL_STOP_15% | 0.6930 | ₹160.31 | 2022-09-20 | ❌ |
| 15 | **LUPIN** | 2022-09-12 | Mon | ₹565.65 | 16,109 | ₹91.38L | 2022-09-26 | Mon | ₹484.53 | 14d | ₹-13.56L | -14.8% | TRAIL_STOP_15% | 0.8096 | ₹572.08 | 2022-09-21 | ❌ |
| 16 | **RELIANCE** | 2022-10-03 | Mon | ₹134.36 | 93,031 | ₹125.36L | 2022-10-06 | Thu | ₹110.79 | 3d | ₹-22.59L | -18.0% | TRAIL_STOP_15% | 0.8019 | ₹134.36 | 2022-10-03 | ❌ |
| 17 | **LUPIN** | 2022-10-03 | Mon | ₹462.85 | 27,006 | ₹125.36L | 2022-10-13 | Thu | ₹386.75 | 10d | ₹-21.22L | -16.9% | TRAIL_STOP_15% | 0.8003 | ₹478.55 | 2022-10-05 | ❌ |
| 18 | **LT** | 2022-10-03 | Mon | ₹880.80 | 14,191 | ₹125.36L | 2022-10-13 | Thu | ₹700.76 | 10d | ₹-26.20L | -20.9% | TRAIL_STOP_15% | 0.7958 | ₹910.73 | 2022-10-05 | ❌ |
| 20 | **BHARTIARTL** | 2022-10-03 | Mon | ₹211.41 | 9,369 | ₹19.86L | 2022-11-03 | Thu | ₹239.88 | 31d | ₹+2.54L | +12.8% | TRAIL_STOP_15% | 0.7146 | ₹310.57 | 2022-10-31 | ✅ |
| 19 | **LUPIN** | 2022-10-24 | Mon | ₹411.63 | 30,367 | ₹125.36L | 2022-10-26 | Wed | ₹326.97 | 2d | ₹-26.36L | -21.0% | TRAIL_STOP_15% | 0.7843 | ₹411.63 | 2022-10-24 | ❌ |
| 21 | **RELIANCE** | 2022-10-24 | Mon | ₹94.41 | 132,406 | ₹125.36L | 2022-11-07 | Mon | ₹77.93 | 14d | ₹-22.47L | -17.9% | TRAIL_STOP_15% | 0.7667 | ₹94.41 | 2022-10-24 | ❌ |
| 22 | **LT** | 2022-10-24 | Mon | ₹488.22 | 11,179 | ₹54.74L | 2022-11-11 | Fri | ₹466.12 | 18d | ₹-2.78L | -5.1% | TRAIL_STOP_15% | 0.7534 | ₹578.34 | 2022-11-04 | ❌ |
| 23 | **LUPIN** | 2022-11-14 | Mon | ₹270.35 | 46,235 | ₹125.36L | 2022-11-17 | Thu | ₹222.57 | 3d | ₹-22.75L | -18.1% | TRAIL_STOP_15% | 0.8035 | ₹270.35 | 2022-11-14 | ❌ |
| 24 | **LT** | 2022-11-14 | Mon | ₹478.21 | 26,139 | ₹125.36L | 2022-11-28 | Mon | ₹383.76 | 14d | ₹-25.34L | -20.2% | TRAIL_STOP_15% | 0.7935 | ₹478.21 | 2022-11-14 | ❌ |
| 25 | **COFORGE** | 2022-11-14 | Mon | ₹31.92 | 1,399 | ₹0.45L | 2022-11-29 | Tue | ₹31.22 | 15d | ₹-0.01L | -2.8% | TRAIL_STOP_15% | 0.7153 | ₹36.87 | 2022-11-25 | ❌ |
| 26 | **BHARTIARTL** | 2022-11-14 | Mon | ₹215.96 | 12,021 | ₹26.04L | 2022-12-07 | Wed | ₹179.72 | 23d | ₹-4.49L | -17.3% | TRAIL_STOP_15% | 0.7176 | ₹215.96 | 2022-11-14 | ❌ |
| 27 | **LT** | 2022-12-05 | Mon | ₹340.58 | 22,360 | ₹76.37L | 2022-12-09 | Fri | ₹277.50 | 4d | ₹-14.50L | -19.0% | TRAIL_STOP_15% | 0.7768 | ₹340.58 | 2022-12-05 | ❌ |
| 28 | **TATAELXSI** | 2022-12-05 | Mon | ₹123.79 | 1,058 | ₹1.31L | 2022-12-15 | Thu | ₹102.82 | 10d | ₹-0.23L | -17.4% | TRAIL_STOP_15% | 0.6307 | ₹125.75 | 2022-12-06 | ❌ |
| 29 | **LUPIN** | 2022-12-05 | Mon | ₹179.14 | 69,776 | ₹125.36L | 2022-12-30 | Fri | ₹168.45 | 25d | ₹-8.16L | -6.5% | TRAIL_STOP_15% | 0.8376 | ₹209.80 | 2022-12-27 | ❌ |
| 30 | **COFORGE** | 2022-12-26 | Mon | ₹20.32 | 6,999 | ₹1.43L | 2023-02-06 | Mon | ₹18.81 | 42d | ₹-0.11L | -8.0% | ML_EXIT(rc=0.418) | 0.7071 | ₹20.63 | 2023-01-06 | ❌ |
| 32 | **LT** | 2022-12-26 | Mon | ₹200.49 | 41,314 | ₹83.07L | 2023-03-16 | Thu | ₹281.38 | 80d | ₹+32.84L | +39.5% | TRAIL_STOP_15% | 0.7663 | ₹349.48 | 2023-02-27 | ✅ |
| 37 | **TATAELXSI** | 2022-12-26 | Mon | ₹78.75 | 31 | ₹0.02L | 2024-04-05 | Fri | ₹462.21 | 466d | ₹+0.12L | +483.5% | END_SIM | 0.6901 | ₹483.03 | 2024-03-18 | ✅ |
| 38 | **BHARTIARTL** | 2023-01-16 | Mon | ₹109.18 | 105,199 | ₹115.19L | 2024-04-05 | Fri | ₹575.52 | 445d | ₹+488.50L | +424.1% | END_SIM | 0.8380 | ₹592.53 | 2024-04-01 | ✅ |
| 31 | **LUPIN** | 2023-02-06 | Mon | ₹230.42 | 1,413 | ₹3.27L | 2023-03-15 | Wed | ₹186.63 | 37d | ₹-0.64L | -19.5% | TRAIL_STOP_15% | 0.8099 | ₹234.35 | 2023-02-07 | ❌ |
| 33 | **COFORGE** | 2023-03-20 | Mon | ₹20.11 | 9,919 | ₹2.00L | 2023-04-10 | Mon | ₹25.34 | 21d | ₹+0.51L | +25.3% | ML_EXIT(rc=0.302) | 0.7448 | ₹26.99 | 2023-04-04 | ✅ |
| 34 | **LUPIN** | 2023-03-20 | Mon | ₹190.13 | 61,131 | ₹116.57L | 2023-04-20 | Thu | ₹159.72 | 31d | ₹-19.21L | -16.5% | TRAIL_STOP_15% | 0.8002 | ₹192.91 | 2023-03-24 | ❌ |
| 35 | **LT** | 2023-04-10 | Mon | ₹281.16 | 885 | ₹2.50L | 2023-04-21 | Fri | ₹237.88 | 11d | ₹-0.40L | -15.9% | TRAIL_STOP_15% | 0.7696 | ₹281.16 | 2023-04-10 | ❌ |
| 36 | **LT** | 2023-05-01 | Mon | ₹210.48 | 794 | ₹1.68L | 2023-09-14 | Thu | ₹255.16 | 136d | ₹+0.34L | +20.5% | TRAIL_STOP_15% | 0.7177 | ₹301.53 | 2023-08-31 | ✅ |
| 39 | **LUPIN** | 2023-05-01 | Mon | ₹147.35 | 66,177 | ₹97.79L | 2024-04-05 | Fri | ₹309.50 | 340d | ₹+106.43L | +108.8% | END_SIM | 0.8301 | ₹353.16 | 2024-03-07 | ✅ |
| 40 | **LT** | 2023-09-25 | Mon | ₹234.66 | 856 | ₹2.01L | 2024-04-05 | Fri | ₹640.51 | 193d | ₹+3.45L | +171.4% | END_SIM | 0.8159 | ₹641.15 | 2024-04-05 | ✅ |

### Per-Ticker Summary

| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |
|--------|:------:|:--:|:--------:|----------:|:------------:|
| BHARTIARTL | 5 | 40% | +79.2% | ₹+462.15L 📈 | [BHARTIARTL chart](ticker_BHARTIARTL_S1_—_GFC_2008-style_(-65%_over_12mo).png) |
| BEL | 1 | 100% | +21.2% | ₹+25.83L 📈 | [BEL chart](ticker_BEL_S1_—_GFC_2008-style_(-65%_over_12mo).png) |
| LT | 10 | 50% | +17.4% | ₹-2.84L 📉 | [LT chart](ticker_LT_S1_—_GFC_2008-style_(-65%_over_12mo).png) |
| TATAELXSI | 3 | 33% | +152.7% | ₹-9.55L 📉 | [TATAELXSI chart](ticker_TATAELXSI_S1_—_GFC_2008-style_(-65%_over_12mo).png) |
| COFORGE | 5 | 40% | +1.0% | ₹-11.63L 📉 | [COFORGE chart](ticker_COFORGE_S1_—_GFC_2008-style_(-65%_over_12mo).png) |
| RELIANCE | 4 | 0% | -14.3% | ₹-69.30L 📉 | [RELIANCE chart](ticker_RELIANCE_S1_—_GFC_2008-style_(-65%_over_12mo).png) |
| LUPIN | 12 | 8% | -5.6% | ₹-81.09L 📉 | [LUPIN chart](ticker_LUPIN_S1_—_GFC_2008-style_(-65%_over_12mo).png) |

---

## S2 — Flash Crash (-20% in 10 days)
**Period:** 2022-01-03 → 2023-09-22 (1.7 years)  |  **Data type:** SYNTHETIC

### Key Metrics

| Metric | Value | Verification Formula |
|--------|:-----:|---------------------|
| Net Annual (STCG) | **+22.84%** | CAGR × 0.80 (20% STCG) |
| Gross CAGR | +28.55% | (final/initial)^(1/years) - 1 |
| Total Return | +53.90% | (final_eq - 5cr) / 5cr |
| Win Rate | 58.3% | count(pnl>0) / total_trades |
| Max Drawdown | -15.58% | min(eq/cummax(eq) - 1) |
| Sharpe Ratio | 1.725 | (mean(ret) - 6.5%/252) / std × √252 |
| Calmar Ratio | 1.832 | CAGR / |max_dd| (>1.0 = good) |
| Total Trades | 12 | count(trade records) |
| Avg Hold Days | 202 days | mean(exit_date - entry_date) |
| Expected Value | +19.193%/trade | WR×avg_win - (1-WR)×avg_loss |

### Annual Returns

| Year | Return | Signal |
|------|:------:|--------|
| 2022 | +51.1% | ✅ Strong |
| 2023 | +1.8% | 🟡 Mild |

### Equity Curve
![Equity Curve](equity_S2_—_Flash_Crash_(-20%_in_10_days).png)

### Complete Trade Log

Every single trade — date, price, shares, P&L — for manual verification.

| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |
|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|
| 1 | **COFORGE** | 2022-01-03 | Mon | ₹62.60 | 199,682 | ₹125.36L | 2022-01-24 | Mon | ₹60.62 | 21d | ₹-4.67L | -3.7% | ML_EXIT(rc=0.298) | 0.8094 | ₹63.45 | 2022-01-12 | ❌ |
| 2 | **BEL** | 2022-01-03 | Mon | ₹24.48 | 496,160 | ₹121.80L | 2022-03-07 | Mon | ₹24.81 | 63d | ₹+0.92L | +0.8% | ML_EXIT(rc=0.300) | 0.6456 | ₹26.17 | 2022-01-21 | ✅ |
| 4 | **LUPIN** | 2022-01-03 | Mon | ₹1346.34 | 9,284 | ₹125.36L | 2022-09-15 | Thu | ₹1281.36 | 255d | ₹-6.74L | -5.4% | TRAIL_STOP_15% | 0.8012 | ₹1583.02 | 2022-08-11 | ❌ |
| 5 | **LT** | 2022-01-03 | Mon | ₹840.97 | 14,863 | ₹125.36L | 2022-09-16 | Fri | ₹881.58 | 256d | ₹+5.29L | +4.2% | TRAIL_STOP_15% | 0.7140 | ₹1050.91 | 2022-08-03 | ✅ |
| 7 | **RELIANCE** | 2022-01-24 | Mon | ₹188.29 | 63,923 | ₹120.71L | 2022-09-21 | Wed | ₹180.61 | 240d | ₹-5.60L | -4.6% | TRAIL_STOP_15% | 0.7030 | ₹213.22 | 2022-08-03 | ❌ |
| 3 | **COFORGE** | 2022-03-07 | Mon | ₹57.49 | 212,784 | ₹122.69L | 2022-06-20 | Mon | ₹53.33 | 105d | ₹-9.54L | -7.8% | ML_EXIT(rc=0.293) | 0.7444 | ₹58.78 | 2022-03-14 | ❌ |
| 6 | **BHARTIARTL** | 2022-06-20 | Mon | ₹387.07 | 29,190 | ₹113.31L | 2022-09-16 | Fri | ₹327.76 | 88d | ₹-17.92L | -15.8% | TRAIL_STOP_15% | 0.8551 | ₹387.07 | 2022-06-20 | ❌ |
| 8 | **RELIANCE** | 2022-10-03 | Mon | ₹177.36 | 70,479 | ₹125.36L | 2023-07-25 | Tue | ₹245.73 | 295d | ₹+47.33L | +37.8% | TRAIL_STOP_15% | 0.8019 | ₹290.79 | 2023-04-21 | ✅ |
| 9 | **LUPIN** | 2022-10-03 | Mon | ₹1067.23 | 11,712 | ₹125.36L | 2023-09-22 | Fri | ₹2221.39 | 354d | ₹+134.06L | +106.9% | END_SIM | 0.8003 | ₹2328.43 | 2023-09-14 | ✅ |
| 10 | **LT** | 2022-10-03 | Mon | ₹830.64 | 15,048 | ₹125.36L | 2023-09-22 | Fri | ₹1258.26 | 354d | ₹+63.44L | +50.6% | END_SIM | 0.7958 | ₹1435.40 | 2023-05-04 | ✅ |
| 11 | **BHARTIARTL** | 2022-10-03 | Mon | ₹268.38 | 31,286 | ₹84.21L | 2023-09-22 | Fri | ₹411.21 | 354d | ₹+44.07L | +52.3% | END_SIM | 0.7146 | ₹413.91 | 2023-09-20 | ✅ |
| 12 | **TATAELXSI** | 2023-08-14 | Mon | ₹1266.33 | 9,871 | ₹125.36L | 2023-09-22 | Fri | ₹1465.24 | 39d | ₹+18.85L | +15.0% | END_SIM | 0.6281 | ₹1466.71 | 2023-09-22 | ✅ |

### Per-Ticker Summary

| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |
|--------|:------:|:--:|:--------:|----------:|:------------:|
| LUPIN | 2 | 50% | +50.8% | ₹+127.32L 📈 | [LUPIN chart](ticker_LUPIN_S2_—_Flash_Crash_(-20%_in_10_days).png) |
| LT | 2 | 100% | +27.4% | ₹+68.73L 📈 | [LT chart](ticker_LT_S2_—_Flash_Crash_(-20%_in_10_days).png) |
| RELIANCE | 2 | 50% | +16.6% | ₹+41.73L 📈 | [RELIANCE chart](ticker_RELIANCE_S2_—_Flash_Crash_(-20%_in_10_days).png) |
| BHARTIARTL | 2 | 50% | +18.3% | ₹+26.15L 📈 | [BHARTIARTL chart](ticker_BHARTIARTL_S2_—_Flash_Crash_(-20%_in_10_days).png) |
| TATAELXSI | 1 | 100% | +15.0% | ₹+18.85L 📈 | [TATAELXSI chart](ticker_TATAELXSI_S2_—_Flash_Crash_(-20%_in_10_days).png) |
| BEL | 1 | 100% | +0.8% | ₹+0.92L 📈 | [BEL chart](ticker_BEL_S2_—_Flash_Crash_(-20%_in_10_days).png) |
| COFORGE | 2 | 0% | -5.8% | ₹-14.20L 📉 | [COFORGE chart](ticker_COFORGE_S2_—_Flash_Crash_(-20%_in_10_days).png) |

---

## S3 — Prolonged Bear (-40% over 30mo)
**Period:** 2022-01-03 → 2025-04-18 (3.3 years)  |  **Data type:** SYNTHETIC

### Key Metrics

| Metric | Value | Verification Formula |
|--------|:-----:|---------------------|
| Net Annual (STCG) | **-37.55%** | CAGR × 0.80 (20% STCG) |
| Gross CAGR | -46.94% | (final/initial)^(1/years) - 1 |
| Total Return | -87.56% | (final_eq - 5cr) / 5cr |
| Win Rate | 5.9% | count(pnl>0) / total_trades |
| Max Drawdown | -88.50% | min(eq/cummax(eq) - 1) |
| Sharpe Ratio | -4.228 | (mean(ret) - 6.5%/252) / std × √252 |
| Calmar Ratio | -0.530 | CAGR / |max_dd| (>1.0 = good) |
| Total Trades | 51 | count(trade records) |
| Avg Hold Days | 84 days | mean(exit_date - entry_date) |
| Expected Value | -10.726%/trade | WR×avg_win - (1-WR)×avg_loss |

### Annual Returns

| Year | Return | Signal |
|------|:------:|--------|
| 2022 | -51.6% | 🔴 Loss |
| 2023 | -46.7% | 🔴 Loss |
| 2024 | -54.2% | 🔴 Loss |
| 2025 | +5.1% | 🟡 Mild |

### Equity Curve
![Equity Curve](equity_S3_—_Prolonged_Bear_(-40%_over_30mo).png)

### Complete Trade Log

Every single trade — date, price, shares, P&L — for manual verification.

| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |
|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|
| 1 | **COFORGE** | 2022-01-03 | Mon | ₹62.60 | 199,682 | ₹125.36L | 2022-01-24 | Mon | ₹60.99 | 21d | ₹-3.93L | -3.1% | ML_EXIT(rc=0.298) | 0.8094 | ₹62.62 | 2022-01-06 | ❌ |
| 2 | **BEL** | 2022-01-03 | Mon | ₹24.48 | 496,160 | ₹121.80L | 2022-03-07 | Mon | ₹23.96 | 63d | ₹-3.26L | -2.7% | ML_EXIT(rc=0.300) | 0.6456 | ₹25.20 | 2022-01-13 | ❌ |
| 3 | **LUPIN** | 2022-01-03 | Mon | ₹1346.34 | 9,284 | ₹125.36L | 2022-04-01 | Fri | ₹1127.79 | 88d | ₹-20.96L | -16.7% | TRAIL_STOP_15% | 0.8012 | ₹1346.34 | 2022-01-03 | ❌ |
| 6 | **LT** | 2022-01-03 | Mon | ₹840.97 | 14,863 | ₹125.36L | 2022-05-30 | Mon | ₹726.83 | 147d | ₹-17.64L | -14.1% | TRAIL_STOP_15% | 0.7140 | ₹868.28 | 2022-04-08 | ❌ |
| 4 | **RELIANCE** | 2022-01-24 | Mon | ₹182.85 | 66,222 | ₹121.44L | 2022-04-01 | Fri | ₹154.24 | 67d | ₹-19.59L | -16.1% | TRAIL_STOP_15% | 0.7030 | ₹182.85 | 2022-01-24 | ❌ |
| 7 | **COFORGE** | 2022-03-07 | Mon | ₹59.54 | 198,620 | ₹118.59L | 2022-06-15 | Wed | ₹55.59 | 100d | ₹-8.49L | -7.2% | TRAIL_STOP_15% | 0.7444 | ₹65.80 | 2022-04-06 | ❌ |
| 5 | **RELIANCE** | 2022-04-18 | Mon | ₹141.92 | 57,283 | ₹81.53L | 2022-05-23 | Mon | ₹126.64 | 35d | ₹-9.20L | -11.3% | TRAIL_STOP_15% | 0.7639 | ₹149.27 | 2022-04-28 | ❌ |
| 8 | **LUPIN** | 2022-04-18 | Mon | ₹1144.16 | 10,925 | ₹125.36L | 2022-06-24 | Fri | ₹964.23 | 67d | ₹-20.33L | -16.2% | TRAIL_STOP_15% | 0.8189 | ₹1144.16 | 2022-04-18 | ❌ |
| 9 | **BHARTIARTL** | 2022-05-30 | Mon | ₹284.95 | 43,866 | ₹125.36L | 2022-07-18 | Mon | ₹248.53 | 49d | ₹-16.66L | -13.3% | TRAIL_STOP_15% | 0.8413 | ₹293.29 | 2022-06-14 | ❌ |
| 10 | **LT** | 2022-05-30 | Mon | ₹728.28 | 7,550 | ₹55.14L | 2022-07-27 | Wed | ₹618.30 | 58d | ₹-8.60L | -15.6% | TRAIL_STOP_15% | 0.7549 | ₹728.28 | 2022-05-30 | ❌ |
| 12 | **TATAELXSI** | 2022-06-20 | Mon | ₹257.78 | 42,224 | ₹109.16L | 2022-09-05 | Mon | ₹218.62 | 77d | ₹-17.12L | -15.7% | TRAIL_STOP_15% | 0.6626 | ₹257.78 | 2022-06-20 | ❌ |
| 11 | **LUPIN** | 2022-07-11 | Mon | ₹911.15 | 11,502 | ₹105.10L | 2022-08-04 | Thu | ₹744.51 | 24d | ₹-19.72L | -18.8% | TRAIL_STOP_15% | 0.8219 | ₹911.15 | 2022-07-11 | ❌ |
| 13 | **BHARTIARTL** | 2022-08-01 | Mon | ₹262.24 | 47,667 | ₹125.36L | 2022-10-06 | Thu | ₹231.55 | 66d | ₹-15.31L | -12.2% | TRAIL_STOP_15% | 0.7708 | ₹281.13 | 2022-08-08 | ❌ |
| 14 | **LT** | 2022-08-01 | Mon | ₹623.79 | 4,984 | ₹31.18L | 2022-10-13 | Thu | ₹525.10 | 73d | ₹-5.08L | -16.3% | TRAIL_STOP_15% | 0.6766 | ₹623.79 | 2022-08-01 | ❌ |
| 16 | **LUPIN** | 2022-08-22 | Mon | ₹713.97 | 11,795 | ₹84.46L | 2022-10-31 | Mon | ₹608.41 | 70d | ₹-12.90L | -15.3% | TRAIL_STOP_15% | 0.8024 | ₹717.09 | 2022-08-23 | ❌ |
| 15 | **RELIANCE** | 2022-09-12 | Mon | ₹96.70 | 94,775 | ₹91.91L | 2022-10-26 | Wed | ₹84.92 | 44d | ₹-11.66L | -12.7% | TRAIL_STOP_15% | 0.7432 | ₹100.89 | 2022-09-22 | ❌ |
| 17 | **BHARTIARTL** | 2022-10-24 | Mon | ₹216.77 | 57,663 | ₹125.36L | 2022-11-04 | Fri | ₹183.90 | 11d | ₹-19.62L | -15.7% | TRAIL_STOP_15% | 0.7536 | ₹216.77 | 2022-10-24 | ❌ |
| 19 | **LT** | 2022-10-24 | Mon | ₹542.46 | 2,238 | ₹12.18L | 2022-12-15 | Thu | ₹460.14 | 52d | ₹-1.91L | -15.7% | TRAIL_STOP_15% | 0.7534 | ₹542.46 | 2022-10-24 | ❌ |
| 18 | **BHARTIARTL** | 2022-11-14 | Mon | ₹177.79 | 70,308 | ₹125.36L | 2022-12-08 | Thu | ₹153.34 | 24d | ₹-17.87L | -14.2% | TRAIL_STOP_15% | 0.7176 | ₹182.17 | 2022-11-24 | ❌ |
| 20 | **COFORGE** | 2022-11-14 | Mon | ₹38.09 | 18,099 | ₹6.91L | 2023-01-18 | Wed | ₹32.04 | 65d | ₹-1.13L | -16.4% | TRAIL_STOP_15% | 0.7153 | ₹38.09 | 2022-11-14 | ❌ |
| 21 | **LUPIN** | 2022-11-14 | Mon | ₹544.06 | 22,975 | ₹125.36L | 2023-02-07 | Tue | ₹476.08 | 85d | ₹-16.30L | -13.0% | TRAIL_STOP_15% | 0.8035 | ₹563.59 | 2022-11-22 | ❌ |
| 22 | **TATAELXSI** | 2022-12-26 | Mon | ₹199.17 | 996 | ₹1.99L | 2023-02-09 | Thu | ₹173.99 | 45d | ₹-0.26L | -13.2% | TRAIL_STOP_15% | 0.6901 | ₹205.71 | 2022-12-29 | ❌ |
| 23 | **LT** | 2022-12-26 | Mon | ₹459.10 | 25,163 | ₹115.86L | 2023-02-23 | Thu | ₹408.19 | 59d | ₹-13.44L | -11.6% | TRAIL_STOP_15% | 0.7663 | ₹488.27 | 2023-01-25 | ❌ |
| 25 | **BHARTIARTL** | 2023-02-06 | Mon | ₹125.40 | 4,546 | ₹5.72L | 2023-04-07 | Fri | ₹106.14 | 60d | ₹-0.91L | -15.8% | TRAIL_STOP_15% | 0.7601 | ₹125.40 | 2023-02-06 | ❌ |
| 24 | **LUPIN** | 2023-02-27 | Mon | ₹494.52 | 25,277 | ₹125.36L | 2023-03-30 | Thu | ₹406.60 | 31d | ₹-22.88L | -18.2% | TRAIL_STOP_15% | 0.8143 | ₹499.37 | 2023-03-08 | ❌ |
| 26 | **LT** | 2023-02-27 | Mon | ₹413.48 | 20,843 | ₹86.43L | 2023-04-10 | Mon | ₹348.80 | 42d | ₹-13.94L | -16.1% | TRAIL_STOP_15% | 0.7925 | ₹413.48 | 2023-02-27 | ❌ |
| 27 | **COFORGE** | 2023-02-27 | Mon | ₹31.47 | 4,699 | ₹1.48L | 2023-04-10 | Mon | ₹28.37 | 42d | ₹-0.15L | -10.4% | ML_EXIT(rc=0.302) | 0.7914 | ₹31.47 | 2023-02-27 | ❌ |
| 28 | **LUPIN** | 2023-04-10 | Mon | ₹396.74 | 31,507 | ₹125.36L | 2023-05-15 | Mon | ₹333.72 | 35d | ₹-20.52L | -16.4% | TRAIL_STOP_15% | 0.8028 | ₹397.47 | 2023-04-12 | ❌ |
| 29 | **BEL** | 2023-04-10 | Mon | ₹9.00 | 180 | ₹0.02L | 2023-05-22 | Mon | ₹8.90 | 42d | ₹-0.00L | -1.7% | ML_EXIT(rc=0.383) | 0.6973 | ₹9.35 | 2023-04-18 | ❌ |
| 30 | **BHARTIARTL** | 2023-04-10 | Mon | ₹110.26 | 852 | ₹0.94L | 2023-07-05 | Wed | ₹97.04 | 86d | ₹-0.12L | -12.5% | TRAIL_STOP_15% | 0.7294 | ₹115.45 | 2023-05-02 | ❌ |
| 37 | **LT** | 2023-04-10 | Mon | ₹349.50 | 15,638 | ₹54.81L | 2024-01-11 | Thu | ₹322.46 | 276d | ₹-4.53L | -8.3% | TRAIL_STOP_15% | 0.7696 | ₹382.47 | 2023-04-20 | ❌ |
| 31 | **LUPIN** | 2023-05-22 | Mon | ₹315.84 | 32,535 | ₹103.06L | 2023-07-18 | Tue | ₹275.63 | 57d | ₹-13.64L | -13.2% | TRAIL_STOP_15% | 0.8069 | ₹325.85 | 2023-05-23 | ❌ |
| 32 | **TATAELXSI** | 2023-05-22 | Mon | ₹150.77 | 1,171 | ₹1.77L | 2023-09-11 | Mon | ₹143.56 | 112d | ₹-0.09L | -5.3% | TRAIL_STOP_15% | 0.6106 | ₹170.01 | 2023-07-07 | ❌ |
| 33 | **LUPIN** | 2023-07-24 | Mon | ₹285.10 | 532 | ₹1.52L | 2023-09-19 | Tue | ₹250.10 | 57d | ₹-0.19L | -12.8% | TRAIL_STOP_15% | 0.8172 | ₹294.81 | 2023-08-10 | ❌ |
| 34 | **BHARTIARTL** | 2023-07-24 | Mon | ₹91.08 | 97,132 | ₹88.72L | 2023-09-26 | Tue | ₹77.45 | 64d | ₹-13.71L | -15.5% | TRAIL_STOP_15% | 0.8209 | ₹92.38 | 2023-07-25 | ❌ |
| 35 | **LUPIN** | 2023-09-25 | Mon | ₹242.69 | 1,223 | ₹2.98L | 2023-10-24 | Tue | ₹202.00 | 29d | ₹-0.51L | -17.2% | TRAIL_STOP_15% | 0.8341 | ₹242.69 | 2023-09-25 | ❌ |
| 48 | **TATAELXSI** | 2023-09-25 | Mon | ₹144.57 | 36 | ₹0.05L | 2025-04-18 | Fri | ₹196.13 | 571d | ₹+0.02L | +34.9% | END_SIM | 0.7296 | ₹210.49 | 2025-03-11 | ✅ |
| 38 | **COFORGE** | 2023-10-16 | Mon | ₹17.50 | 419,948 | ₹73.72L | 2024-01-16 | Tue | ₹15.31 | 92d | ₹-9.61L | -13.0% | TRAIL_STOP_15% | 0.6398 | ₹18.07 | 2023-11-03 | ❌ |
| 36 | **LUPIN** | 2023-11-06 | Mon | ₹186.55 | 1,970 | ₹3.69L | 2023-12-13 | Wed | ₹157.75 | 37d | ₹-0.59L | -15.9% | TRAIL_STOP_15% | 0.8070 | ₹186.55 | 2023-11-06 | ❌ |
| 44 | **BHARTIARTL** | 2023-12-18 | Mon | ₹49.77 | 6,229 | ₹3.11L | 2024-06-17 | Mon | ₹49.84 | 182d | ₹-0.01L | -0.4% | TRAIL_STOP_15% | 0.8362 | ₹58.79 | 2024-03-20 | ❌ |
| 39 | **LUPIN** | 2024-01-29 | Mon | ₹129.83 | 86,392 | ₹112.48L | 2024-02-16 | Fri | ₹109.73 | 18d | ₹-17.96L | -16.0% | TRAIL_STOP_15% | 0.8057 | ₹129.83 | 2024-01-29 | ❌ |
| 41 | **LT** | 2024-01-29 | Mon | ₹349.86 | 550 | ₹1.93L | 2024-04-04 | Thu | ₹302.82 | 66d | ₹-0.27L | -13.9% | TRAIL_STOP_15% | 0.7280 | ₹361.10 | 2024-02-06 | ❌ |
| 40 | **LUPIN** | 2024-02-19 | Mon | ₹108.88 | 85,106 | ₹92.93L | 2024-03-13 | Wed | ₹93.38 | 23d | ₹-13.69L | -14.7% | TRAIL_STOP_15% | 0.7732 | ₹110.74 | 2024-02-20 | ❌ |
| 43 | **LUPIN** | 2024-04-01 | Mon | ₹86.76 | 91,340 | ₹79.48L | 2024-05-14 | Tue | ₹73.46 | 43d | ₹-12.57L | -15.8% | TRAIL_STOP_15% | 0.8003 | ₹86.76 | 2024-04-01 | ❌ |
| 42 | **PERSISTENT** | 2024-04-22 | Mon | ₹20.91 | 14,286 | ₹3.00L | 2024-05-13 | Mon | ₹19.91 | 21d | ₹-0.16L | -5.3% | ML_EXIT(rc=0.430) | 0.6138 | ₹21.66 | 2024-05-02 | ❌ |
| 45 | **COFORGE** | 2024-05-13 | Mon | ₹12.20 | 23,209 | ₹2.84L | 2024-07-12 | Fri | ₹10.49 | 60d | ₹-0.41L | -14.5% | TRAIL_STOP_15% | 0.7051 | ₹12.67 | 2024-06-12 | ❌ |
| 46 | **LUPIN** | 2024-06-03 | Mon | ₹66.80 | 98,234 | ₹65.81L | 2024-09-04 | Wed | ₹57.76 | 93d | ₹-9.23L | -14.0% | TRAIL_STOP_15% | 0.8099 | ₹68.10 | 2024-06-12 | ❌ |
| 49 | **RELIANCE** | 2024-06-24 | Mon | ₹13.37 | 31,103 | ₹4.17L | 2025-04-18 | Fri | ₹11.99 | 298d | ₹-0.45L | -10.9% | END_SIM | 0.6607 | ₹13.56 | 2024-11-15 | ❌ |
| 50 | **LT** | 2024-07-15 | Mon | ₹220.79 | 1,110 | ₹2.46L | 2025-04-18 | Fri | ₹276.38 | 277d | ₹+0.60L | +24.5% | END_SIM | 0.7900 | ₹293.22 | 2025-03-26 | ✅ |
| 47 | **LUPIN** | 2024-09-16 | Mon | ₹54.18 | 102,419 | ₹55.65L | 2025-01-02 | Thu | ₹49.09 | 108d | ₹-5.52L | -9.9% | TRAIL_STOP_15% | 0.7951 | ₹58.46 | 2024-10-30 | ❌ |
| 51 | **BHARTIARTL** | 2025-01-20 | Mon | ₹64.83 | 77,246 | ₹50.22L | 2025-04-18 | Fri | ₹70.75 | 88d | ₹+4.27L | +8.5% | END_SIM | 0.8599 | ₹76.55 | 2025-03-18 | ✅ |

### Per-Ticker Summary

| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |
|--------|:------:|:--:|:--------:|----------:|:------------:|
| PERSISTENT | 1 | 0% | -5.3% | ₹-0.16L 📉 | [PERSISTENT chart](ticker_PERSISTENT_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |
| BEL | 2 | 0% | -2.2% | ₹-3.26L 📉 | [BEL chart](ticker_BEL_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |
| TATAELXSI | 4 | 25% | +0.2% | ₹-17.46L 📉 | [TATAELXSI chart](ticker_TATAELXSI_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |
| COFORGE | 6 | 0% | -10.8% | ₹-23.73L 📉 | [COFORGE chart](ticker_COFORGE_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |
| RELIANCE | 4 | 0% | -12.7% | ₹-40.91L 📉 | [RELIANCE chart](ticker_RELIANCE_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |
| LT | 9 | 11% | -9.7% | ₹-64.82L 📉 | [LT chart](ticker_LT_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |
| BHARTIARTL | 9 | 11% | -10.1% | ₹-79.94L 📉 | [BHARTIARTL chart](ticker_BHARTIARTL_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |
| LUPIN | 16 | 0% | -15.3% | ₹-207.51L 📉 | [LUPIN chart](ticker_LUPIN_S3_—_Prolonged_Bear_(-40%_over_30mo).png) |

---

## S4 — High-VIX Chop (sideways 24mo)
**Period:** 2022-01-03 → 2023-11-03 (1.8 years)  |  **Data type:** SYNTHETIC

### Key Metrics

| Metric | Value | Verification Formula |
|--------|:-----:|---------------------|
| Net Annual (STCG) | **+60.03%** | CAGR × 0.80 (20% STCG) |
| Gross CAGR | +75.03% | (final/initial)^(1/years) - 1 |
| Total Return | +178.81% | (final_eq - 5cr) / 5cr |
| Win Rate | 39.3% | count(pnl>0) / total_trades |
| Max Drawdown | -14.91% | min(eq/cummax(eq) - 1) |
| Sharpe Ratio | 2.473 | (mean(ret) - 6.5%/252) / std × √252 |
| Calmar Ratio | 5.033 | CAGR / |max_dd| (>1.0 = good) |
| Total Trades | 28 | count(trade records) |
| Avg Hold Days | 87 days | mean(exit_date - entry_date) |
| Expected Value | +25.316%/trade | WR×avg_win - (1-WR)×avg_loss |

### Annual Returns

| Year | Return | Signal |
|------|:------:|--------|
| 2022 | +43.7% | ✅ Strong |
| 2023 | +94.0% | ✅ Strong |

### Equity Curve
![Equity Curve](equity_S4_—_High-VIX_Chop_(sideways_24mo).png)

### Complete Trade Log

Every single trade — date, price, shares, P&L — for manual verification.

| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |
|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|
| 1 | **COFORGE** | 2022-01-03 | Mon | ₹62.60 | 199,682 | ₹125.36L | 2022-01-24 | Mon | ₹67.26 | 21d | ₹+8.55L | +6.8% | ML_EXIT(rc=0.298) | 0.8094 | ₹71.91 | 2022-01-14 | ✅ |
| 2 | **LT** | 2022-01-03 | Mon | ₹840.97 | 14,863 | ₹125.36L | 2022-02-03 | Thu | ₹796.88 | 31d | ₹-7.26L | -5.8% | TRAIL_STOP_15% | 0.7140 | ₹978.39 | 2022-01-10 | ❌ |
| 4 | **BEL** | 2022-01-03 | Mon | ₹24.48 | 496,160 | ₹121.80L | 2022-03-07 | Mon | ₹31.88 | 63d | ₹+35.93L | +29.5% | ML_EXIT(rc=0.300) | 0.6456 | ₹31.92 | 2022-03-07 | ✅ |
| 7 | **LUPIN** | 2022-01-03 | Mon | ₹1346.34 | 9,284 | ₹125.36L | 2022-04-14 | Thu | ₹1304.35 | 101d | ₹-4.61L | -3.7% | TRAIL_STOP_15% | 0.8012 | ₹1540.40 | 2022-01-11 | ❌ |
| 3 | **RELIANCE** | 2022-01-24 | Mon | ₹202.61 | 61,694 | ₹125.36L | 2022-02-21 | Mon | ₹208.53 | 28d | ₹+2.91L | +2.3% | TRAIL_STOP_15% | 0.7030 | ₹249.78 | 2022-02-07 | ✅ |
| 9 | **LT** | 2022-02-14 | Mon | ₹902.10 | 13,856 | ₹125.36L | 2022-06-01 | Wed | ₹1371.15 | 107d | ₹+64.08L | +51.1% | TRAIL_STOP_15% | 0.7713 | ₹1625.53 | 2022-05-18 | ✅ |
| 5 | **RELIANCE** | 2022-03-07 | Mon | ₹212.28 | 58,883 | ₹125.36L | 2022-04-12 | Tue | ₹252.08 | 36d | ₹+22.64L | +18.1% | TRAIL_STOP_15% | 0.7451 | ₹299.55 | 2022-04-04 | ✅ |
| 6 | **COFORGE** | 2022-03-07 | Mon | ₹66.22 | 188,759 | ₹125.36L | 2022-04-13 | Wed | ₹57.69 | 37d | ₹-16.79L | -13.4% | TRAIL_STOP_15% | 0.7444 | ₹68.86 | 2022-03-09 | ❌ |
| 8 | **RELIANCE** | 2022-04-18 | Mon | ₹244.54 | 51,117 | ₹125.36L | 2022-05-02 | Mon | ₹207.92 | 14d | ₹-19.39L | -15.5% | TRAIL_STOP_15% | 0.7639 | ₹246.54 | 2022-04-22 | ❌ |
| 10 | **LUPIN** | 2022-04-18 | Mon | ₹1236.86 | 10,106 | ₹125.36L | 2022-06-01 | Wed | ₹1083.26 | 44d | ₹-16.20L | -12.9% | TRAIL_STOP_15% | 0.8189 | ₹1305.38 | 2022-04-22 | ❌ |
| 12 | **BHARTIARTL** | 2022-04-18 | Mon | ₹544.63 | 22,951 | ₹125.36L | 2022-07-15 | Fri | ₹517.73 | 88d | ₹-6.88L | -5.5% | TRAIL_STOP_15% | 0.7240 | ₹623.98 | 2022-07-01 | ❌ |
| 11 | **COFORGE** | 2022-05-09 | Mon | ₹56.26 | 222,201 | ₹125.36L | 2022-06-08 | Wed | ₹48.10 | 30d | ₹-18.80L | -15.0% | TRAIL_STOP_15% | 0.6955 | ₹56.66 | 2022-05-18 | ❌ |
| 13 | **LT** | 2022-06-20 | Mon | ₹1242.74 | 10,058 | ₹125.36L | 2022-08-24 | Wed | ₹1537.95 | 65d | ₹+28.88L | +23.0% | TRAIL_STOP_15% | 0.7335 | ₹1899.18 | 2022-08-11 | ✅ |
| 16 | **LUPIN** | 2022-06-20 | Mon | ₹974.54 | 12,826 | ₹125.36L | 2022-12-20 | Tue | ₹1000.39 | 183d | ₹+2.58L | +2.1% | TRAIL_STOP_15% | 0.7899 | ₹1196.29 | 2022-09-12 | ✅ |
| 27 | **TATAELXSI** | 2022-06-20 | Mon | ₹474.39 | 26,349 | ₹125.36L | 2023-11-03 | Fri | ₹3665.00 | 501d | ₹+837.53L | +668.1% | END_SIM | 0.6626 | ₹4014.20 | 2023-10-03 | ✅ |
| 21 | **BHARTIARTL** | 2022-08-01 | Mon | ₹523.86 | 23,861 | ₹125.36L | 2023-04-07 | Fri | ₹649.00 | 249d | ₹+29.05L | +23.2% | TRAIL_STOP_15% | 0.7708 | ₹799.93 | 2023-01-27 | ✅ |
| 14 | **RELIANCE** | 2022-09-12 | Mon | ₹206.60 | 60,504 | ₹125.36L | 2022-10-18 | Tue | ₹173.96 | 36d | ₹-20.42L | -16.3% | TRAIL_STOP_15% | 0.7432 | ₹206.60 | 2022-09-12 | ❌ |
| 15 | **RELIANCE** | 2022-10-24 | Mon | ₹191.51 | 65,269 | ₹125.36L | 2022-11-17 | Thu | ₹158.74 | 24d | ₹-22.06L | -17.6% | TRAIL_STOP_15% | 0.7667 | ₹195.29 | 2022-11-04 | ❌ |
| 17 | **LT** | 2022-12-05 | Mon | ₹1519.18 | 8,228 | ₹125.36L | 2023-01-05 | Thu | ₹1332.30 | 31d | ₹-16.06L | -12.8% | TRAIL_STOP_15% | 0.7768 | ₹1585.76 | 2022-12-06 | ❌ |
| 19 | **LUPIN** | 2022-12-26 | Mon | ₹1020.53 | 12,248 | ₹125.36L | 2023-03-31 | Fri | ₹942.02 | 95d | ₹-10.31L | -8.2% | TRAIL_STOP_15% | 0.8156 | ₹1111.42 | 2023-03-03 | ❌ |
| 18 | **LT** | 2023-01-16 | Mon | ₹1550.79 | 8,060 | ₹125.36L | 2023-03-14 | Tue | ₹1292.62 | 57d | ₹-21.47L | -17.1% | TRAIL_STOP_15% | 0.7580 | ₹1598.19 | 2023-02-01 | ❌ |
| 20 | **COFORGE** | 2023-03-20 | Mon | ₹12.88 | 909,948 | ₹117.52L | 2023-04-06 | Thu | ₹10.89 | 17d | ₹-18.68L | -15.9% | TRAIL_STOP_15% | 0.7448 | ₹13.20 | 2023-03-21 | ❌ |
| 22 | **LUPIN** | 2023-04-10 | Mon | ₹966.03 | 12,939 | ₹125.36L | 2023-07-06 | Thu | ₹879.16 | 87d | ₹-11.93L | -9.5% | TRAIL_STOP_15% | 0.8028 | ₹1037.35 | 2023-05-15 | ❌ |
| 25 | **LT** | 2023-04-10 | Mon | ₹1103.88 | 11,323 | ₹125.35L | 2023-10-26 | Thu | ₹1403.63 | 199d | ₹+33.12L | +26.4% | TRAIL_STOP_15% | 0.7696 | ₹1658.36 | 2023-08-25 | ✅ |
| 28 | **BHARTIARTL** | 2023-04-10 | Mon | ₹663.16 | 17,679 | ₹117.58L | 2023-11-03 | Fri | ₹1121.86 | 207d | ₹+80.18L | +68.2% | END_SIM | 0.7294 | ₹1163.71 | 2023-10-27 | ✅ |
| 23 | **LUPIN** | 2023-07-24 | Mon | ₹991.48 | 11,414 | ₹113.50L | 2023-08-30 | Wed | ₹833.60 | 37d | ₹-18.62L | -16.4% | TRAIL_STOP_15% | 0.8172 | ₹997.67 | 2023-08-02 | ❌ |
| 24 | **LUPIN** | 2023-09-04 | Mon | ₹815.16 | 11,643 | ₹95.18L | 2023-10-03 | Tue | ₹751.20 | 29d | ₹-7.98L | -8.4% | TRAIL_STOP_15% | 0.8232 | ₹893.28 | 2023-09-14 | ❌ |
| 26 | **LUPIN** | 2023-10-16 | Mon | ₹707.41 | 12,312 | ₹87.35L | 2023-10-27 | Fri | ₹597.87 | 11d | ₹-13.95L | -16.0% | TRAIL_STOP_15% | 0.8224 | ₹709.91 | 2023-10-19 | ❌ |

### Per-Ticker Summary

| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |
|--------|:------:|:--:|:--------:|----------:|:------------:|
| TATAELXSI | 1 | 100% | +668.1% | ₹+837.53L 📈 | [TATAELXSI chart](ticker_TATAELXSI_S4_—_High-VIX_Chop_(sideways_24mo).png) |
| BHARTIARTL | 3 | 67% | +28.6% | ₹+102.34L 📈 | [BHARTIARTL chart](ticker_BHARTIARTL_S4_—_High-VIX_Chop_(sideways_24mo).png) |
| LT | 6 | 50% | +10.8% | ₹+81.29L 📈 | [LT chart](ticker_LT_S4_—_High-VIX_Chop_(sideways_24mo).png) |
| BEL | 1 | 100% | +29.5% | ₹+35.93L 📈 | [BEL chart](ticker_BEL_S4_—_High-VIX_Chop_(sideways_24mo).png) |
| RELIANCE | 5 | 40% | -5.8% | ₹-36.30L 📉 | [RELIANCE chart](ticker_RELIANCE_S4_—_High-VIX_Chop_(sideways_24mo).png) |
| COFORGE | 4 | 25% | -9.4% | ₹-45.71L 📉 | [COFORGE chart](ticker_COFORGE_S4_—_High-VIX_Chop_(sideways_24mo).png) |
| LUPIN | 8 | 12% | -9.1% | ₹-81.03L 📉 | [LUPIN chart](ticker_LUPIN_S4_—_High-VIX_Chop_(sideways_24mo).png) |

---

## Vulnerability Analysis

Based on stress test results across all periods:

| Vulnerability | Evidence | Severity | Mitigation |
|--------------|---------|:--------:|-----------|
| Macro black swan (Ukraine-style) | Period A — Demonetization Era (IS): -99.7% max DD | 🔴 HIGH | FII gate + trail stops |
| Prolonged drawdown recovery | CB lockout risk in V6 Full | 🔴 HIGH | Add CB Recovery Protocol (V7) |
| No 2008 real data available | Synthetic only — model behavior unknown on pre-2015 data | 🟡 MED | Need pre-2015 data fetch |
| IS inflation | Periods A/B show inflated WR due to model trained on same data | 🟡 MED | Trust only OOS period C |
| Tail risk (gap openings) | Trailing stop assumes next-bar fill — gaps exceed 15% occasionally | 🟡 MED | Consider options hedge |

---

## How to Manually Verify

1. **Pick any trade** from the log above
2. **Pull the ticker's historical data** from any NSE source (NSE website, Bloomberg)
3. **Verify entry date**: Confirm the trading day, price ± 0.1% slippage
4. **Verify exit trigger**: If TRAIL_STOP_15%, price fell 15% below peak price
5. **Verify P&L**: `(exit_price × shares - 0.29% tx) - (entry_price × shares + 0.29% tx)`
6. **Verify CAGR**: `(final_equity / 5,00,00,000)^(1/n_years) - 1`

All inputs are in `reports/multi_strategy_backtest_v6.json`.

---
*Paper mode only. Capital: ₹5 crore. Never switch to LIVE.*