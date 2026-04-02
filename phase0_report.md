# MARK5 Phase 0 Validation Report
**Generated:** 2026-03-20 05:49:30

## Executive Summary

| Check | Result |
|-------|--------|
| Dead features (IC < 0.02) | **9 of 9** |
| Non-stationary features | **2 of 11** |
| Bias tests | ✅ PASS |
| CPCV gate P(Sharpe>1.5) > 70% | ❌ FAIL |
| Critical bugs | **1** |
| High severity bugs | **3** |

> Fix all CRITICAL and HIGH bugs before Phase 1. Do not add new data or models on top of broken foundations.

## 1. Feature IC Analysis

- Forward return horizon: 5 trading days
- IC kill threshold: 0.02
- ICIR target: > 0.50 (stable signal)

| Feature | Mean IC | Std IC | ICIR | p-value | n_tickers | Verdict |
|---------|---------|--------|------|---------|-----------|---------|
| `dist_52w_high` | +0.0741 | 0.0563 | +1.315 | 0.1490 | 19 | ❌ KILL — not statistically significant |
| `amihud_illiquidity` | +0.0591 | 0.0895 | +0.660 | 0.1273 | 19 | ❌ KILL — not statistically significant |
| `gap_significance` | +0.0356 | 0.0363 | +0.980 | 0.4117 | 19 | ❌ KILL — not statistically significant |
| `relative_strength_nifty` | +0.0067 | 0.0797 | +0.084 | 0.2723 | 19 | ❌ KILL — IC below threshold (noise) |
| `lower_wick_ratio` | -0.0013 | 0.0434 | -0.031 | 0.4519 | 19 | ❌ KILL — IC below threshold (noise) |
| `wick_asymmetry` | -0.0014 | 0.0388 | -0.036 | 0.5335 | 19 | ❌ KILL — IC below threshold (noise) |
| `ofi_proxy` | -0.0164 | 0.0332 | -0.494 | 0.4616 | 19 | ❌ KILL — IC below threshold (noise) |
| `price_vwap_deviation` | -0.0214 | 0.0620 | -0.345 | 0.3655 | 19 | ❌ KILL — not statistically significant |
| `efficiency_ratio` | -0.0560 | 0.0562 | -0.995 | 0.2099 | 19 | ❌ KILL — not statistically significant |

## 2. Stationarity Tests (ADF)

- ADF p-value threshold: 0.05
- Non-stationary features must have fractional differentiation applied (d ≈ 0.3–0.5)
- `optimize_frac_diff.py` already exists — use it on flagged features

| Feature | Mean ADF p-value | % Stationary | Verdict |
|---------|-----------------|--------------|---------|
| `fii_flow_3d` | 1.0000 | 0% | ⚠️ NON-STATIONARY → apply fractional differentiation (d ≈ 0.3–0.5) |
| `relative_strength_nifty` | 0.0306 | 74% | ✅ STATIONARY |
| `dist_52w_high` | 0.0057 | 95% | ✅ STATIONARY |
| `efficiency_ratio` | 0.0000 | 100% | ✅ STATIONARY |
| `ofi_proxy` | 0.0000 | 100% | ✅ STATIONARY |
| `delivery_pct` | 1.0000 | 0% | ⚠️ NON-STATIONARY → apply fractional differentiation (d ≈ 0.3–0.5) |
| `amihud_illiquidity` | 0.0040 | 100% | ✅ STATIONARY |
| `lower_wick_ratio` | 0.0000 | 100% | ✅ STATIONARY |
| `wick_asymmetry` | 0.0000 | 100% | ✅ STATIONARY |
| `gap_significance` | 0.0000 | 100% | ✅ STATIONARY |
| `price_vwap_deviation` | 0.0000 | 100% | ✅ STATIONARY |

## 3. Bias Tests

### 3.1 Lookahead Bias

- Original Sharpe: **1.3947**
- Shifted Sharpe (1-day lag): **0.0533**
- Degradation ratio: **0.0382** (must be < 0.70)
- Result: ✅ PASS


### 3.2 Feature Importance Stability

- Mean rank correlation across time folds: **0.7353** (target: >0.7)
- Folds computed: 5
- Top features (average importance): ['dist_52w_high', 'amihud_illiquidity', 'relative_strength_nifty', 'price_vwap_deviation', 'efficiency_ratio']
- Result: ✅ PASS


### 3.3 Signal Correlation

- Avg pairwise signal correlation: **0.1603** (target: <0.4)
- Tickers: 19
- Result: ✅ PASS


## 4. CPCV Validation

- Configuration: C(8,2) = 28 test combinations per ticker
- Embargo: 10 bars between train and test
- Production gate: P(Sharpe > 1.5) > 70% AND worst-5% ≥ 0.0

| Metric | Value |
|--------|-------|
| Folds computed | 532 |
| Tickers used | 19 |
| Mean Sharpe | **0.0961** |
| Std Sharpe | 2.8478 |
| Worst-5% Sharpe | **-4.4434** |
| P(Sharpe > 1.5) | **31.8%** |
| Mean AUC | 0.5328 |
| Production gate | ❌ FAIL |

Sharpe distribution: min=-26.66, p25=-1.46, median=0.43, p75=1.87, max=10.09

## 5. Known Bugs

### BUG-1 — 🔴 CRITICAL
**amihud_illiquidity p99 normalization uses full-dataset quantile — lookahead bias**

- **Location:** `features.py → engineer_all_features() → Feature 7 (amihud_illiquidity)`
- **Impact:** p99 is computed on train+test combined. Training features 'know' the future volatility distribution. Out-of-sample performance will be measurably worse than backtest. Every backtest result is optimistically biased.
- **Fix:** Add training_end_date: Optional[pd.Timestamp] = None parameter to engineer_all_features(). Compute: amihud_p99 = amihud_20.loc[:training_end_date].quantile(0.99). Pass training_end_date from trainer.py (= end of training window per fold).

### BUG-2 — 🟠 HIGH
**delivery_pct is a constant 0.5 — dead feature occupying 1/11 of model capacity**

- **Location:** `features.py → engineer_all_features() → Feature 6 (delivery_pct)`
- **Impact:** The feature is always 0.5 for every stock every day. It contributes pure noise to training. The model wastes capacity trying to learn from a constant. After normalization it becomes a zero vector anyway.
- **Fix:** Either: (a) Remove delivery_pct from FEATURE_COLS list until bhav copy data is integrated via MarketDataProvider, OR (b) integrate NSE bhav copy delivery_volume column — available at: https://www.nseindia.com/market-data/live-market-indices (bhav copy download)

### BUG-3 — 🟠 HIGH
**trainer.py uses walk-forward — cpcv.py exists but is never called**

- **Location:** `trainer.py → train_advanced_ensemble() — CPCV not imported or used`
- **Impact:** Walk-forward only tests the most recent period. If the recent period is anomalous (post-COVID rally, rate hike cycle), fold selection is biased. CPCV tests all C(8,2)=28 time-period combinations — far more honest and statistically robust. Current backtest Sharpes are likely optimistic.
- **Fix:** In trainer.py: from core.models.cpcv import CombinatorialPurgedKFold. Replace the walk-forward loop with cpcv.split(X, y). Selection criterion: highest F-beta across folds where P(Sharpe>1.5) ≥ 70%.

### BUG-4 — 🟠 HIGH
**Slippage is fixed 0.05% — unrealistic for midcap stocks (true cost: 12–25bps round trip)**

- **Location:** `backtester.py (slippage_pct=0.0005), position_sizer.py`
- **Impact:** For a midcap stock (₹20–100cr daily turnover), realistic round-trip cost is 12–25bps. At fixed 5bps, the backtest overstates net P&L by 7–20bps per trade. At 100 trades/year, this is 7–20% of total annual P&L — the difference between a profitable and unprofitable strategy.
- **Fix:** Implement Almgren-Chriss model: slippage = 0.0025 * sqrt(trade_value / adv_20d) + spread_cost where spread_cost = 15bps if adv<₹5cr, 8bps if adv<₹50cr, 3bps otherwise. Add ADV lookup to MarketDataProvider.

### BUG-5 — 🟡 MEDIUM
**STRONG_BULL regime threshold (ret_20d > 15%) never triggers on NSE**

- **Location:** `regime_detector.py → detect_market_regime() → STRONG_BULL check`
- **Impact:** 15% return in 20 days = ~189% annualized. This happens perhaps once per decade on NSE. RULE 88 (Sharpe whitelist for STRONG_BULL) is permanently dead code. The regime_multipliers for STRONG_BULL are never applied.
- **Fix:** Change STRONG_BULL_RET_THRESHOLD from 0.15 → 0.07 (7% in 20 days ≈ 88% ann). Change STRONG_BULL_ADX_THRESHOLD from 40 → 28 (28 = established trend on daily bars). This will trigger 1–3 times/year in real bull markets.

### BUG-6 — 🟡 MEDIUM
**Random Forest is weakest ensemble member — CatBoost outperforms on financial tabular data**

- **Location:** `trainer.py → train_advanced_ensemble() → RandomForestClassifier`
- **Impact:** RF has high variance on daily bar financial data. It consistently underperforms gradient boosting on tabular data in published benchmarks. It contributes noise to the ensemble rather than signal, especially when n_features is small (11).
- **Fix:** Replace RandomForestClassifier with CatBoostClassifier: CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, eval_metric='Logloss', early_stopping_rounds=50, class_weights=[1.0, 2.0], verbose=False, random_seed=42). Also replace arithmetic mean ensemble with stacking meta-learner (LogisticRegression).

### BUG-7 — 🔵 LOW
**wick_asymmetry is likely redundant with lower_wick_ratio (pairwise corr expected >0.75)**

- **Location:** `features.py → Feature 9 (wick_asymmetry) vs Feature 8 (lower_wick_ratio)`
- **Impact:** Both features measure buyer/seller wick dominance from the same candle geometry. Redundant features bloat the model and reduce effective feature count. Run IC analysis — if |IC(wick_asymmetry)| < IC_KILL_THRESHOLD or corr(wick_asymmetry, lower_wick_ratio) > 0.75, kill wick_asymmetry.
- **Fix:** After IC analysis: if wick_asymmetry IC < 0.02 OR spearmanr(wick_asymmetry, lower_wick_ratio) > 0.75 → remove wick_asymmetry from FEATURE_COLS in features.py.

## 6. Priority Actions (Before Phase 1)

Complete in this order. Do not skip ahead.

1. **Fix BUG-1**: Add training_end_date: Optional[pd.
2. **Kill dead features**: Remove `dist_52w_high`, `amihud_illiquidity`, `gap_significance`, `relative_strength_nifty`, `lower_wick_ratio`, `wick_asymmetry`, `ofi_proxy`, `price_vwap_deviation`, `efficiency_ratio` from `FEATURE_COLS` in `features.py`. They have IC < 0.02 and contribute noise.
3. **Fix non-stationary features**: Apply fractional differentiation (use `optimize_frac_diff.py`) to: `fii_flow_3d`, `delivery_pct`.
4. **Fix BUG-2**: Either: (a) Remove delivery_pct from FEATURE_COLS list until bhav copy data is integrated via MarketDataProvider, OR (b) integrate NSE bhav copy delivery_volume column — available at: https://www.
5. **Fix BUG-3**: In trainer.
6. **Fix BUG-4**: Implement Almgren-Chriss model: slippage = 0.
7. **Switch to CPCV**: Replace walk-forward in `trainer.py` with `CombinatorialPurgedKFold`. Current P(Sharpe > 1.5) = 32% — need > 70% before adding new features or data.

---
*Report generated by `phase0_validation.py` at 2026-03-20 05:49:30. Do NOT proceed to Phase 1 until all CRITICAL and HIGH items are resolved.*