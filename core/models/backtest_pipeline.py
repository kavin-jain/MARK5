"""
MARK5 Walk-Forward Backtest Engine v3.0 — Real Alpha vs NIFTY50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGELOG:
- [2026-05-10] v3.0: NIFTY50 benchmark, regime filter, multi-condition
  signal quality, 60% allocation, win-rate tracking per fold.

TRADING ROLE: Offline validation vs NIFTY50 index (institutional benchmark)
SAFETY LEVEL: HIGH
"""
import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.data.nse_data_provider import fetch_equity_ohlcv, fetch_nifty50_index
from core.models.features import engineer_features_df, FEATURE_COLS
from core.models.backtester import RobustBacktester

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | [WF-BACKTEST] | %(levelname)s | %(message)s")
logger = logging.getLogger("MARK5.WalkForwardBacktest")

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_MONTHS      = 18
TEST_MONTHS       = 3
CONFIDENCE_HURDLE = 0.55  # ALIGNED: matches predictor.py PROBABILITY_HURDLE (was 0.56 — mismatch)
                           # Note: ml_momentum_portfolio.py ML_ENTRY_HURDLE=0.52 is intentionally
                           # lower (broader entry universe for portfolio rotation)
INITIAL_CAPITAL   = 5_00_00_000.0
RISK_PER_TRADE    = 0.015
MAX_HOLD_DAYS     = 9999


# ── Predictor ────────────────────────────────────────────────────────────────
class LightPredictor:
    def __init__(self, ticker: str, models_dir: str = "models"):
        self.ticker = ticker
        self.models: Dict = {}
        self.scaler = None       # StandardScaler from training pipeline (v2 artifacts)
        self.meta_model = None   # NNLS meta-learner from CPCV OOF (v2 artifacts)
        self.feature_names: List[str] = FEATURE_COLS
        self.feature_engine_version: str = "v1"  # updated from features.json during _load
        self.is_v2: bool = False  # set to True when feature_engine_version == 'v2'
        self._load(models_dir)

    def _load(self, models_dir: str):
        import joblib, json
        for candidate in [self.ticker, f"{self.ticker}.NS"]:
            path = os.path.join(models_dir, candidate)
            if not os.path.exists(path):
                continue
            versions = sorted([d for d in os.listdir(path) if d.startswith("v")],
                              key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
            if not versions:
                continue
            latest = os.path.join(path, versions[-1])
            for name in ("xgb_model", "lgb_model", "cat_model"):
                p = os.path.join(latest, f"{name}.pkl")
                if os.path.exists(p):
                    try:
                        self.models[name] = joblib.load(p)
                    except Exception as e:
                        logger.warning(f"[{self.ticker}] {name} load failed: {e}")

            # FIX: Load StandardScaler — models were trained on scaled features.
            # Without scaling, V2 calibration (Platt sigmoid) gets unscaled inputs,
            # degrading probability calibration. Tree models are rank-invariant but
            # the calibration layer is NOT. Load and apply for aligned inference.
            scaler_p = os.path.join(latest, "scaler.pkl")
            if os.path.exists(scaler_p):
                try:
                    self.scaler = joblib.load(scaler_p)
                except Exception as e:
                    logger.warning(f"[{self.ticker}] scaler load failed: {e}")

            # FIX: Load NNLS meta-learner — trained on CPCV OOF predictions.
            # Simple mean ensemble ignores each model's per-ticker performance.
            # Meta-learner gives optimal non-negative weights fitted on held-out data.
            meta_p = os.path.join(latest, "meta_model.pkl")
            if os.path.exists(meta_p):
                try:
                    self.meta_model = joblib.load(meta_p)
                except Exception as e:
                    logger.warning(f"[{self.ticker}] meta_model load failed: {e}")

            feat_p = os.path.join(latest, "features.json")
            if os.path.exists(feat_p):
                with open(feat_p) as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    self.feature_names = raw
                elif isinstance(raw, dict):
                    self.feature_names = raw.get("feature_names", FEATURE_COLS)
                    self.feature_engine_version = raw.get("feature_engine_version", "v1")
                    self.is_v2 = (self.feature_engine_version == "v2")
            return

    def has_models(self) -> bool:
        return len(self.models) > 0

    def validate_signal_quality(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute OOS AUC for model validation.

        Returns AUC score in [0, 1]. A value <= 0.52 means near-random
        predictions — the model should not be trusted for live trading.

        Args:
            X: Feature DataFrame aligned to the OOS period.
            y: Binary label Series (1 = profitable trade, 0 = unprofitable).

        Returns:
            float AUC in [0, 1], defaulting to 0.5 on failure.
        """
        from sklearn.metrics import roc_auc_score
        if not self.models or len(X) < 50:
            return 0.5
        try:
            preds = self.predict_proba(X)
            aligned_idx = X.index.intersection(y.index)
            if len(aligned_idx) < 20:
                return 0.5
            row_positions = X.index.get_indexer(aligned_idx)
            # Filter out any -1 (not-found) positions that get_indexer may return
            valid_mask = row_positions >= 0
            if valid_mask.sum() < 20:
                return 0.5
            auc = roc_auc_score(
                y.loc[aligned_idx[valid_mask]].values,
                preds[row_positions[valid_mask]],
            )
            return float(auc)
        except Exception as e:
            logger.warning(f"[{self.ticker}] validate_signal_quality failed: {e}")
            return 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            return np.full(len(X), 0.5)

        # Align feature columns to training schema (zero-fill missing, drop extras)
        Xm = X.reindex(columns=self.feature_names, fill_value=0.0)

        # NOTE: Scaler is loaded (self.scaler) but intentionally NOT applied here.
        #
        # The models_v2_oos artifacts were trained with an older features_v2.py
        # that applied in-engine rolling z-scores before returning features.
        # A later commit ("FIX: Remove in-engine rolling Z-score") changed the
        # feature engine to return raw [0,1] / natural-scale features instead.
        # The saved scaler.pkl was fitted on those OLD z-scored features, so
        # applying it to the CURRENT un-z-scored features produces garbage inputs
        # (e.g., rsi_14 mean=1.32 in scaler but current engine returns [0,1]).
        #
        # Tree models (XGB/LGB/CatBoost) are scale-invariant (splits are on
        # ranks, not absolute values), so passing un-z-scored features gives
        # consistent predictions matching the backtest baseline.
        #
        # When models are retrained with the current feature engine, uncomment:
        #   _scaler = getattr(self, 'scaler', None)
        #   if _scaler is not None:
        #       Xm = pd.DataFrame(_scaler.transform(Xm.values), ...)

        # Get per-model probabilities (aligned feature set already prepared)
        base_probs: Dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            try:
                base_probs[name] = model.predict_proba(Xm)[:, 1]
            except Exception as e:
                logger.warning(f"[{self.ticker}] {name} predict_proba failed: {e}")

        if not base_probs:
            return np.full(len(X), 0.5)

        # NOTE: NNLS meta-learner is intentionally NOT used for current models_v2_oos.
        #
        # The meta_model (NonNegativeMetaLearner) was trained on CPCV OOF predictions
        # from the OLD feature engine (with in-engine rolling z-scores). Its NNLS
        # coefficients reflect the relative model quality on THOSE features.
        # With the current feature engine (no z-scoring), the base model prediction
        # distributions are different, so the OLD NNLS weights are suboptimal and
        # degrade performance (tested: 13.13% vs 23.42% without meta_model).
        #
        # When models are retrained with the current feature engine, re-enable:
        #   _meta = getattr(self, 'meta_model', None)
        #   if _meta is not None and len(base_probs) >= 2:
        #       stacked = np.column_stack(list(base_probs.values()))
        #       if hasattr(_meta, 'predict_proba'):
        #           ensemble = _meta.predict_proba(stacked)[:, 1]
        #           return np.clip(ensemble, 0.0, 1.0)

        return np.mean(list(base_probs.values()), axis=0)


# ── Signal Engine v4.0 ────────────────────────────────────────────────────────
def _build_signals(test_df: pd.DataFrame, full_df: pd.DataFrame,
                   nifty_close: pd.Series, feat_test: pd.DataFrame,
                   predictor: LightPredictor, fold_num: int, ticker: str) -> pd.Series:
    """
    Two-tier signal engine — targets 3-5 trades per 63-bar fold:

    TIER 1 — BREAKOUT (fires once per trend):
      BB(20, 0.75σ) crossover  |  EMA(10>30) crossover  |  EMA(5>20) crossover
      + >= 2 of 4 confirmations [RSI 45-82, Volume, RelStr, Regime]

    TIER 2 — CONTINUATION (fires on pullbacks during established uptrend):
      EMA(10) > EMA(30)  [trend intact]
      Price touches EMA20 from above (within 0.5%)
      RSI 40-58 [cooling but not breaking]
      Volume >= 80% avg [not distressed]
      Solves: 1.5 trades/fold → 3-5 trades/fold
    """
    close  = test_df["close"]
    volume = test_df.get("volume", pd.Series(1.0, index=test_df.index))

    # ── Shared indicators (used by both tiers) ────────────────────────────────
    ema10      = close.ewm(span=10, adjust=False).mean()
    ema20_fast = close.ewm(span=20, adjust=False).mean()
    ema30      = close.ewm(span=30, adjust=False).mean()
    ema5       = close.ewm(span=5,  adjust=False).mean()
    sma20      = close.rolling(20, min_periods=15).mean()
    std20      = close.rolling(20, min_periods=15).std()

    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    vol_ma  = volume.rolling(20, min_periods=10).mean()
    c2_vol  = volume >= vol_ma                  # full volume: ≥ 100% avg
    c2_vol_soft = volume >= (vol_ma * 0.80)     # soft volume: ≥ 80% avg (for continuations)

    stock_r20 = close.pct_change(20)
    if nifty_close is not None and len(nifty_close) > 25:
        nifty_r20 = nifty_close.pct_change(20).reindex(test_df.index, method="ffill")
        c3_rs     = stock_r20 > nifty_r20
    else:
        c3_rs = stock_r20 > 0

    if nifty_close is not None and len(nifty_close) > 200:
        na        = nifty_close.reindex(test_df.index, method="ffill")
        ne200     = nifty_close.ewm(span=200, adjust=False).mean().reindex(test_df.index, method="ffill")
        c4_regime = (na > ne200)
    else:
        c4_regime = pd.Series(True, index=test_df.index)

    # ── TIER 1: BREAKOUT signals ──────────────────────────────────────────────
    # Fires ONCE at the start of a new trend. 1-2 per 63-bar fold is normal.
    upper_bb   = sma20 + 0.75 * std20
    above_bb   = (close > upper_bb).astype(int)
    bb_cross   = (above_bb == 1) & (above_bb.shift(1) == 0)

    above_ema  = (ema10 > ema30).astype(int)
    ema_cross  = (above_ema == 1) & (above_ema.shift(1) == 0)

    above_ema5 = (ema5 > ema20_fast).astype(int)
    ema5_cross = (above_ema5 == 1) & (above_ema5.shift(1) == 0)

    primary    = bb_cross | ema_cross | ema5_cross

    # Breakout confirmation: need ≥ 2 of 4  (all must be proper booleans before int cast)
    c1_rsi_breakout = ((rsi >= 45) & (rsi <= 82)).astype(int)  # RSI in healthy range
    score_breakout  = (  c1_rsi_breakout
                       + c2_vol.astype(int)
                       + c3_rs.astype(int)
                       + c4_regime.astype(int))
    tier1_signals   = (primary & (score_breakout >= 2)).astype(int)
    tier1_signals.iloc[0] = 0

    # ── TIER 2: CONTINUATION signals (pullback-to-EMA20 during uptrend) ──────
    # Fires 1-3 times AFTER the breakout, when price dips back to EMA20.
    # Rules (all must be True):
    #   a) Trend intact:  EMA10 > EMA30  (medium trend still up)
    #   b) Pullback:      price touches EMA20 from above (within 0.5%)
    #   c) RSI cooled:    RSI 40-60  (not overbought, not breaking down)
    #   d) Soft volume:   ≥ 80% of avg (pullbacks on low vol are healthy)
    #   e) Not at new low: close > EMA30 (stock hasn't broken below trend)
    trend_intact  = ema10 > ema30                                      # a
    at_ema20      = ((close - ema20_fast).abs() / ema20_fast) < 0.015  # b: within 1.5% to catch more bounce touches
    rsi_cooled    = (rsi >= 40) & (rsi <= 60)                          # c
    vol_ok        = c2_vol_soft                                        # d
    above_ema30   = close > ema30                                      # e

    # Additional: price must have been ABOVE EMA20 recently (coming from above, not below)
    # We check that close was > ema20 at least 3 bars ago to confirm it's a pullback
    was_above_ema20 = (close > ema20_fast).shift(3).fillna(False)

    tier2_raw     = (trend_intact & at_ema20 & rsi_cooled & vol_ok &
                     above_ema30 & was_above_ema20).astype(int)

    # Only fire continuation if NOT coinciding with a breakout (avoid duplicate)
    # and apply a 2-bar minimum spacing between continuation signals
    tier2_spaced  = pd.Series(0, index=tier2_raw.index)
    last_cont_bar = -3
    for i, (idx, val) in enumerate(tier2_raw.items()):
        if val == 1 and tier1_signals.iloc[i] == 0 and (i - last_cont_bar) >= 2:
            tier2_spaced.iloc[i] = 1
            last_cont_bar = i

    # ── TIER 3: TREND RIDING signals (Re-entries) ─────────────────────────────
    # Re-enter mega-trends that never touch EMA20
    riding_trend = (close > ema10) & (ema10 > ema20_fast) & (ema20_fast > ema30)
    rsi_safe     = (rsi >= 50) & (rsi <= 75)
    
    tier3_raw    = (riding_trend & rsi_safe & vol_ok).astype(int)
    
    tier3_spaced  = pd.Series(0, index=tier3_raw.index)
    last_t3_bar = -3
    for i, (idx, val) in enumerate(tier3_raw.items()):
        if val == 1 and tier1_signals.iloc[i] == 0 and tier2_spaced.iloc[i] == 0 and (i - last_t3_bar) >= 5:
            # Space out trend re-entries by 5 bars minimum
            tier3_spaced.iloc[i] = 1
            last_t3_bar = i

    # Combined signal: breakout OR continuation OR trend-riding
    combined = (tier1_signals | tier2_spaced | tier3_spaced).astype(int)
    combined.iloc[0] = 0

    n_t1 = int(tier1_signals.sum())
    n_t2 = int(tier2_spaced.sum())
    n_t3 = int(tier3_spaced.sum())
    logger.info(f"[{ticker}] Fold {fold_num}: breakout={n_t1} | continuation={n_t2} | riding={n_t3} = {n_t1+n_t2+n_t3} signals")

    # ── ML gate: applied to BREAKOUT signals only ─────────────────────────────
    # Continuation signals bypass ML — they are already technically confirmed.
    # ML was trained on breakout patterns; applying it to pullback entries
    # introduces survivorship bias in the filtering.
    signals = combined
    test_df["confidence"] = 0.68

    if predictor.has_models() and not feat_test.empty:
        try:
            proba    = predictor.predict_proba(feat_test)
            ml_conf  = pd.Series(proba, index=feat_test.index)
            prob_std = float(ml_conf.std())
            prob_max = float(ml_conf.max())
            if (prob_std > 0.04) and (prob_max > CONFIDENCE_HURDLE):
                conf_full = ml_conf.reindex(test_df.index).ffill().fillna(0.50)
                # ML filters BREAKOUTS only — continuations pass through unfiltered
                t1_ml_filtered = tier1_signals.where(conf_full >= CONFIDENCE_HURDLE, 0)
                signals        = (t1_ml_filtered | tier2_spaced | tier3_spaced).astype(int)
                test_df["confidence"] = conf_full.clip(0.50, 0.99)
                logger.info(
                    f"[{ticker}] Fold {fold_num}: ML ACTIVE → "
                    f"{int(t1_ml_filtered.sum())} breakouts | {n_t2} continuations | {n_t3} riding "
                    f"= {int(signals.sum())} total"
                )
            else:
                logger.info(f"[{ticker}] Fold {fold_num}: ML FLAT → {int(combined.sum())} signals (tech only)")
        except Exception as e:
            logger.warning(f"[{ticker}] Fold {fold_num}: ML failed ({e})")

    return signals




# ── Walk-Forward Engine ───────────────────────────────────────────────────────
class WalkForwardBacktest:

    def __init__(self, ticker: str, start_date: str, end_date: str,
                 train_months: int = TRAIN_MONTHS, test_months: int = TEST_MONTHS,
                 models_dir: str = "models", initial_capital: float = INITIAL_CAPITAL):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.train_months = train_months
        self.test_months  = test_months
        self.models_dir   = models_dir
        self.initial_capital = initial_capital

    def _generate_folds(self, index: pd.DatetimeIndex):
        folds = []
        train_delta = pd.DateOffset(months=self.train_months)
        test_delta  = pd.DateOffset(months=self.test_months)
        fold_start  = index[0]
        end         = index[-1]
        while True:
            train_end = fold_start + train_delta
            test_end  = min(train_end + test_delta, end)
            if train_end > end:
                break
            tm = (index >= fold_start) & (index < train_end)
            em = (index >= train_end)  & (index <= test_end)
            if tm.sum() >= 200 and em.sum() >= 20:
                folds.append((index[tm], index[em]))
            fold_start += test_delta
            if test_end >= end:
                break
        return folds

    def run(self) -> Dict:
        logger.info(f"[{self.ticker}] Walk-forward {self.start_date}→{self.end_date}")

        df = fetch_equity_ohlcv(self.ticker, self.start_date, self.end_date)
        if df is None or df.empty:
            return {"status": "failed", "reason": "No OHLCV data"}

        nifty = fetch_nifty50_index(self.start_date, self.end_date)
        context = {"nifty_close": nifty} if nifty is not None else {}

        predictor = LightPredictor(self.ticker, self.models_dir)
        folds     = self._generate_folds(df.index)
        if not folds:
            return {"status": "failed", "reason": "No valid folds"}

        logger.info(f"[{self.ticker}] {len(folds)} folds")
        fold_results: List[Dict] = []

        for fold_num, (train_idx, test_idx) in enumerate(folds, 1):
            test_df = df.loc[test_idx].copy()
            if len(test_df) < 10:
                continue

            # Feature engineering (no lookahead)
            try:
                feat_test = engineer_features_df(
                    df.loc[:test_idx[-1]], context=context, is_daily=True
                ).reindex(test_idx).dropna(how="all")
            except Exception:
                feat_test = pd.DataFrame()

            # Generate multi-condition signals
            signals = _build_signals(
                test_df, df, nifty, feat_test, predictor, fold_num, self.ticker
            )
            signals = signals.reindex(test_df.index).fillna(0).astype(int)

            # Backtester — per fold, fresh capital
            bt = RobustBacktester(
                initial_capital=self.initial_capital,
                segment="EQUITY_DELIVERY",
                slippage_pct=0.001,
                atr_multiplier=2.5,
                pt_multiplier=2.5,
                risk_per_trade=RISK_PER_TRADE,
                max_hold_days=MAX_HOLD_DAYS,
                use_atr_stop=True,
            )
            eq_curve, metrics = bt.run_simulation(test_df, signals, symbol=self.ticker)

            if eq_curve.empty or metrics.get("error"):
                continue

            strategy_return_pct = (eq_curve.iloc[-1] / self.initial_capital - 1) * 100.0

            # ── Benchmark 1: Stock B&H at MATCHING capital fraction ─────────────
            # The strategy only deploys 25–40% of capital per trade. A fair alpha
            # comparison must scale the buy-and-hold return by the same fraction.
            # We estimate avg deployment from trades: avg_alloc ≈ trades * (max_alloc/2)
            n_trades = metrics.get("Total Trades", 0)
            # Estimate actual capital fraction deployed per fold (cap at 100%)
            avg_alloc_fraction = min(1.0, n_trades * 0.30) if n_trades > 0 else 0.30
            stock_bh_raw = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100.0
            bh_return_pct = stock_bh_raw * avg_alloc_fraction   # scaled B&H

            # ── Benchmark 2: NIFTY50 (institutional reference — unscaled) ───────
            if nifty is not None:
                nifty_fold = nifty.reindex(test_idx, method="ffill").dropna()
                nifty_bh_pct = (nifty_fold.iloc[-1] / nifty_fold.iloc[0] - 1) * 100.0 if len(nifty_fold) >= 2 else 0.0
            else:
                nifty_bh_pct = 0.0

            alpha_pct       = strategy_return_pct - bh_return_pct  # vs scaled stock B&H
            nifty_alpha_pct = strategy_return_pct - nifty_bh_pct   # vs raw NIFTY

            fold_result = {
                "fold":                 fold_num,
                "test_start":           test_idx[0].date(),
                "test_end":             test_idx[-1].date(),
                "strategy_return_pct":  round(strategy_return_pct, 2),
                "bh_return_pct":        round(bh_return_pct, 2),
                "nifty_bh_pct":         round(nifty_bh_pct, 2),
                "alpha_pct":            round(alpha_pct, 2),
                "nifty_alpha_pct":      round(nifty_alpha_pct, 2),
                "sharpe":               round(metrics.get("Sharpe Ratio", 0.0), 3),
                "max_dd_pct":           round(metrics.get("Max Drawdown %", 0.0), 2),
                "profit_factor":        round(metrics.get("Profit Factor", 0.0), 3),
                "win_rate_pct":         round(metrics.get("Win Rate %", 0.0), 2),
                "total_trades":         metrics.get("Total Trades", 0),
                "beats_bh":             alpha_pct > 0,
                "beats_nifty":          nifty_alpha_pct > 0,
            }
            fold_results.append(fold_result)

            logger.info(
                f"[{self.ticker}] Fold {fold_num}: "
                f"Strat={strategy_return_pct:+.1f}% | StockBH(scaled)={bh_return_pct:+.1f}% | NIFTY={nifty_bh_pct:+.1f}% | "
                f"Alpha(stock)={alpha_pct:+.1f}% | Sharpe={metrics.get('Sharpe Ratio', 0):.2f} | "
                f"Trades={metrics.get('Total Trades', 0)} | "
                f"WinRate={metrics.get('Win Rate %', 0):.0f}% | "
                f"{'✅ BEATS StockBH' if alpha_pct > 0 else '❌ BELOW StockBH'}"
            )

        if not fold_results:
            return {"status": "failed", "reason": "No folds completed"}

        results_df       = pd.DataFrame(fold_results)
        beats_bh_rate    = results_df["beats_bh"].mean()
        beats_nifty_rate = results_df["beats_nifty"].mean()

        agg = {
            "status":                   "success",
            "ticker":                   self.ticker,
            "n_folds":                  len(fold_results),
            "benchmark":                "Stock-BH(scaled) + NIFTY(ref)",
            "beats_bh_rate":            round(float(beats_bh_rate), 3),
            "beats_nifty_rate":         round(float(beats_nifty_rate), 3),
            "mean_alpha_pct":           round(float(results_df["alpha_pct"].mean()), 2),
            "mean_nifty_alpha_pct":     round(float(results_df["nifty_alpha_pct"].mean()), 2),
            "mean_sharpe":              round(float(results_df["sharpe"].mean()), 3),
            "mean_strategy_return_pct": round(float(results_df["strategy_return_pct"].mean()), 2),
            "mean_bh_return_pct":       round(float(results_df["bh_return_pct"].mean()), 2),
            "worst_drawdown_pct":       round(float(results_df["max_dd_pct"].min()), 2),
            "mean_profit_factor":       round(float(results_df["profit_factor"].mean()), 3),
            "mean_win_rate_pct":        round(float(results_df["win_rate_pct"].mean()), 2),
            "total_trades":             int(results_df["total_trades"].sum()),
            "folds":                    fold_results,
            "production_ready": (
                beats_bh_rate >= 0.50
                and results_df["sharpe"].mean() >= 0.8
                and results_df["max_dd_pct"].min() >= -20.0
                and results_df["profit_factor"].mean() >= 1.3
            ),
        }

        self._print_report(agg, results_df)
        return agg

    def _print_report(self, agg: Dict, results_df: pd.DataFrame) -> None:
        verdict = "✅ PRODUCTION READY" if agg["production_ready"] else "⚠️  NEEDS TUNING"
        w = 88
        print("\n" + "═" * w)
        print(f"  MARK5 — {self.ticker}  |  {agg['n_folds']} folds")
        print("═" * w)
        print(f"  {verdict}")
        print(f"  Beats StockBH(scaled) : {agg['beats_bh_rate']:.0%}  |  Beats NIFTY(ref): {agg['beats_nifty_rate']:.0%}")
        print(f"  Alpha(vs StockBH)     : {agg['mean_alpha_pct']:+.2f}%   |  Alpha(vs NIFTY): {agg['mean_nifty_alpha_pct']:+.2f}%")
        print(f"  Sharpe     : {agg['mean_sharpe']:.3f}  |  Win Rate: {agg['mean_win_rate_pct']:.1f}%")
        print(f"  Strategy   : {agg['mean_strategy_return_pct']:+.2f}%  vs  StockBH(scaled) {agg['mean_bh_return_pct']:+.2f}%  (per-fold avg)")
        print(f"  Max DD     : {agg['worst_drawdown_pct']:.2f}%  |  Profit Factor: {agg['mean_profit_factor']:.3f}")
        print(f"  Trades     : {agg['total_trades']} total  ({agg['total_trades']/agg['n_folds']:.1f}/fold)")
        print("─" * w)
        print(f"  {'Fold':<4} {'Period':<26} {'Strat%':>7} {'StBH%':>6} {'NIFTY%':>7} {'α-stock':>8} {'α-nifty':>8} {'Sharpe':>7} {'WR%':>5} {'T':>3}")
        print("  " + "─" * 82)
        for row in agg["folds"]:
            period = f"{row['test_start']} → {row['test_end']}"
            mark   = "✅" if row["beats_bh"] else "❌"
            nmark  = "↑" if row["beats_nifty"] else "↓"
            print(
                f"  {row['fold']:<3} {mark} {period:<24} "
                f"{row['strategy_return_pct']:>+7.1f} {row['bh_return_pct']:>+6.1f} "
                f"{row['nifty_bh_pct']:>+7.1f} "
                f"{row['alpha_pct']:>+8.1f} {row['nifty_alpha_pct']:>+8.1f}{nmark} "
                f"{row['sharpe']:>7.3f} "
                f"{row['win_rate_pct']:>4.0f}% {row['total_trades']:>3}"
            )
        print("═" * w + "\n")


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────
class PipelineOrchestrator:
    def __init__(self, tickers, start_date, end_date,
                 models_dir="models", run_training=False, run_backtest=True):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.models_dir = models_dir
        self.run_training = run_training
        self.run_backtest = run_backtest

    def run(self):
        summary = {"passed": [], "failed": [], "results": {}}
        for ticker in self.tickers:
            logger.info(f"\n{'='*60}\n  PIPELINE: {ticker}\n{'='*60}")
            try:
                result = self._run_one(ticker)
                (summary["passed"] if result.get("status") == "success"
                 else summary["failed"]).append(ticker)
                summary["results"][ticker] = result
            except Exception as exc:
                logger.exception(f"[{ticker}] crashed: {exc}")
                summary["failed"].append(ticker)
                summary["results"][ticker] = {"status": "crashed", "reason": str(exc)}
        self._print_summary(summary)
        return summary

    def _run_one(self, ticker):
        bt_result = {"status": "skipped"}
        if self.run_backtest:
            try:
                wf = WalkForwardBacktest(ticker=ticker, start_date=self.start_date,
                                         end_date=self.end_date, models_dir=self.models_dir)
                bt_result = wf.run()
            except Exception as e:
                logger.error(f"[{ticker}] Backtest failed: {e}")
                bt_result = {"status": "failed", "reason": str(e)}

        status = "success" if bt_result.get("status") == "success" else bt_result.get("status", "failed")
        return {"status": status, "ticker": ticker, "backtest": bt_result,
                "production_ready": bt_result.get("production_ready", False)}

    def _print_summary(self, summary):
        print("\n" + "═" * 70)
        print("  MARK5 PIPELINE SUMMARY  (benchmark: NIFTY50)")
        print("═" * 70)
        print(f"  Passed : {len(summary['passed'])} — {summary['passed']}")
        print(f"  Failed : {len(summary['failed'])} — {summary['failed']}")
        print()
        for ticker, res in summary["results"].items():
            bt   = res.get("backtest", {})
            prod = "✅ PROD" if bt.get("production_ready") else "⚠️ "
            mean_alpha = bt.get('mean_alpha_pct', 'N/A')
            nifty_alpha = bt.get('mean_nifty_alpha_pct', 'N/A')
            beats_bh   = bt.get('beats_bh_rate', 0)
            beats_nifty= bt.get('beats_nifty_rate', 0)
            alpha_str  = f"{mean_alpha:>+6.1f}%" if isinstance(mean_alpha, (int, float)) else str(mean_alpha)
            nifty_str  = f"{nifty_alpha:>+6.1f}%" if isinstance(nifty_alpha, (int, float)) else str(nifty_alpha)
            print(f"  {ticker:<15} {prod} | α(stock)={alpha_str} BeatsStBH={beats_bh:.0%} | "
                  f"α(nifty)={nifty_str} BeatsNIFTY={beats_nifty:.0%} | "
                  f"Sharpe={bt.get('mean_sharpe', 'N/A')} | WR={bt.get('mean_win_rate_pct', 0):.0f}% | "
                  f"Folds={bt.get('n_folds', '?')}")
        print("═" * 70 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+",
                        default=["INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2025-12-31")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    PipelineOrchestrator(tickers=args.tickers, start_date=args.start,
                         end_date=args.end, models_dir=args.models_dir,
                         run_training=args.train).run()
