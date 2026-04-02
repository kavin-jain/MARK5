# MARK5 System Architecture Map

Complete documentation of the MARK5 trading system, including data flow, component interactions, and exact calculation methodologies.

---

## Architecture Flow

### 1. Training Pipeline (Offline/Periodic)
```text
[Historical OHLCV Data]
      │
      ├─► [AdvancedFeatureEngine] ──► (60+ Technical & Micro-structural Features)
      │
      └─► [FinancialEngineer] ─────► (Triple Barrier Meta-Labels)
                                        │
                                        ▼
                                [MARK5MLTrainer]
                                  • Time-Decay Sample Weights
                                  • Purged Cross-Validation (Embargo)
                                  • XGBoost/LightGBM/RF/CatBoost Ensemble
                                  • SHAP Feature Pruning
                                  • Isotonic Calibration
                                        │
                                        ▼
                                [Model Registry] ◄── Models & Thresholds
```

### 2. Inference & Execution Pipeline (Live)
```text
[Live Market Tick] ──► [AdvancedFeatureEngine] ──► (Feature Vector)
                                                         │
                                                         ▼
[Model Registry] ────► (Ensemble Models) ────────► [MARK5Predictor]
                                                         │
                                                         ▼
                                                [TradingSignalGenerator]
                                                  • Applies Probability Hurdle (≥0.42)
                                                  • Checks Entropy & Regime
                                                         │
                                                         ▼
                                  [RiskManager & VolatilityAwarePositionSizer]
                                                  • Core Rules (9-18) Enforcement
                                                         │
                                                         ▼
                                                  [OrderManager]
```

---

## Component Details & Calculation Methodologies

### 1. FinancialEngineer (Label Generation)
**File**: `core/models/training/financial_engineer.py`
**Purpose**: Creates targets for ML training using real-world constraints.
**Calculations**:
- **Dynamic Volatility**: Calculates 14-period True Range (ATR) to measure current market expansion.
- **Triple Barrier Method**: 
  - *Upper Barrier (Profit Target)*: `Close + (2.0 × ATR)`
  - *Lower Barrier (Stop Loss)*: `Close - (2.0 × ATR)` (symmetric configurations)
  - *Vertical Barrier (Time)*: `15 bars` (Max Holding Period)
- **Labeling Logic**: 
  - `BUY` (2): Upper barrier touched first.
  - `SELL` (0): Lower barrier touched first.
  - `WAIT` (1): Vertical barrier touched first (Timeout) or neither constraint hit.

### 2. MARK5MLTrainer (Model Training)
**File**: `core/models/training/trainer.py`
**Purpose**: Trains the predictive ensemble with strict leakage prevention.
**Calculations**:
- **Sample Weighting**: Exponential time-decay weights to prioritize recent market dynamics.
- **Cross-Validation**: Purged Walk-Forward Cross Validation with an **embargo period** ensures no look-ahead bias spanning the holding period (30 bars embargo).
- **Ensemble Weights**: Base models (XGBoost, LightGBM, Random Forest, CatBoost) are weighted inversely by their log-loss on the validation set.
- **Calibration**: Uses Isotonic Regression to map raw model outputs to true probabilities.
- **Feature Pruning**: Uses SHAP values to dynamically drop the bottom 15% noisest features during each fold.

### 3. AdvancedFeatureEngine
**File**: `core/models/features.py`
**Purpose**: Transforms raw OHLCV into stationary, predictive ML features.
**Calculations**:
- **Multi-Scale Interactions**:
  - `Momentum_Micro_Macro_Div`: `RSI(5) - RSI(20)`
  - `MACD_Hist_Trend`: Normalised difference of MACD lines over ATR.
- **Volume & Liquidity**:
  - `Volume_Surge`: `Current Volume / SMA(Volume, 20)`
  - `OBV_ZScore`: `(OBV - RollingMean(OBV, 20)) / RollingStd(OBV, 20)`
- **Volatility**:
  - `Norm_ATR`: `ATR(14) / Close`
  - `Roll_Measure_Spread`: `RollingMax(High, 14) - RollingMin(Low, 14)`
- **Return Series**: Multi-period raw returns (5d, 10d, 20d).
- **Deduplication**: Systematically drops features with absolute correlation `> 0.95`.

### 4. MARK5Predictor
**File**: `core/models/predictor.py`
**Purpose**: Generates real-time probabilistic forecasts.
**Calculations**:
- **Probability Hurdle**: Computes required break-even probability `p = (SL + TC) / (PT + SL + 2×TC)`. Placed floor at minimum `0.42` for 3-class models to beat noise and account for market frictions.
- **Information Entropy**: Calculates predictability `H = -Σ(p × log(p))`. High entropy means flat uncertain distributions; predictions are rejected if the model is confused (e.g. `[0.34, 0.33, 0.33]`).
- **Ensemble Fallback**: Weighted averages probabilities across base models to ensure robustness.

### 5. RiskManager & VolatilityAwarePositionSizer
**File**: `core/trading/risk_manager.py` & `core/trading/position_sizer.py`
**Purpose**: Capital preservation and bet sizing (Financial Rules 9-18).
**Calculations**:
- **Base Position Size**: `(Capital × RiskPerTrade%) / (EntryPrice - StopLossPrice)`
- **Dynamic Multipliers**:
  - *Volatility Mult*: If `VIX > 25`, reduce max exposure from 75% to 40%.
  - *Regime Mult*: Ranges from 1.2x (Trending, Low Vol) to 0.3x/0.0x (Crisis).
- **Circuit Breakers**: Hard daily loss limits (`-2%` halt new entries, `-3%` emergency liquidate).

### 6. TradingSignalGenerator
**File**: `core/trading/signals.py`
**Purpose**: Translates probabilities into actionable orders.
**Filters**:
- Minimum Return:Risk must exceed actual market spread cost (Rule 27).
- Hard filtering restricting trade placement during extreme volume spike times (9:15-9:30 or 3:20-3:30).

### 7. TradeJournal & PerformanceTracker
**File**: `core/analytics/journal.py` & `core/analytics/performance.py`
**Calculations**:
- **NSE Taxes & Charges**: Brokerage (₹20), STT (0.025% intraday, 0.1% delivery), Exchange (0.00325%), GST (18%), Stamp Duty (0.015%).
- **Expectancy & System Quality Number (SQN)**: `SQN = (Expectancy / StdDev(R)) × sqrt(Trades)`. Triggers automatic model retraining if SQN drops significantly.
- **Rolling Sharpe**: Assessed dynamically to freeze specific assets (Rule 90 `Sharpe > 0.5`).
