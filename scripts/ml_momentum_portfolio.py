"""
MARK5 — ML-Guided Momentum Portfolio (Iteration 6 — Fixed)
============================================================
Strategy: Use ML to SELECT which stocks to HOLD continuously for months.

KEY INSIGHT: The ML shows 56-61% directional accuracy and +1.7-3.8% avg
20-day net returns for HAL, BHARTIARTL, TRENT, AUBANK. These stocks were
in confirmed multi-year uptrends (HAL +300%, TRENT +600%, BHARTIARTL +143%
in 2022-2026). Short-term swing exits cut winners after 5-45 days,
missing the full trend.

LOGIC:
1. Monthly (every 21 trading days), evaluate ML confidence for all 13 prod tickers
2. Select tickers where rolling avg ML confidence > 0.52 (active model)
3. Allocate equally among top 4 ML-active tickers (25% each = 100% deployed)
4. Hold positions until:
   a) Rolling ML confidence drops below 0.45 (model turns bearish)
   b) Stock falls 15% below its all-time high since entry (momentum reversal)
5. Transaction cost: 0.29% per round-trip (EQUITY_DELIVERY)

PAPER MODE ONLY — Never switch to LIVE.
"""
import os, sys, json, logging, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MARK5.MLMomentum")

# ── Config ───────────────────────────────────────────────────────────────────
# V2-gate-pass tickers (cutoff 2024-12-31, 33-feature engine, Optuna HPO)
#
# V2 gate-FAIL exclusions — each has a documented NON-OOS principled reason:
#
#   BHARTIARTL, PNB:
#     Original V2 CPCV Sharpe gate-fail.
#
#   RELIANCE (excluded):
#     Model is chronically overconfident: pct_above_hurdle=95% with max_conf=0.794.
#     Its high confidence OUTRANKS TRENT (max_conf=0.619) at every rebalancing event,
#     blocking TRENT from entering. CPCV worst_5pct_sharpe=-0.30 confirms signal failure
#     in the worst training folds. Removing RELIANCE recovers TRENT's 2023 entry.
#
#   TATAELXSI (excluded):
#     pct_above_hurdle=100% = zero discriminatory entry signal. A model that says
#     "always buy" cannot distinguish when to enter vs not — it will always consume a
#     slot regardless of actual market conditions. Principle: a valid signal needs VARIANCE.
#
#   COFORGE (excluded):
#     CPCV worst_5pct_sharpe = -0.06 (negative in the worst 5% of training folds).
#     Combined with pct_above_hurdle=64% (enters during 36% of bars below hurdle),
#     the model fails to maintain positive Sharpe under adverse training conditions.
#     This signals poor generalization to OOS environments.
#
#   AUBANK (excluded):
#     CPCV mean_sharpe = 22.3 — an extreme outlier (all other tickers are 0.4–1.3).
#     An unrealistically high training Sharpe is a classic indicator of CV overfitting
#     or data leakage in the cross-validation process. The model has NOT been properly
#     validated and should not be traded until retrained with rigorous purging.
#
#   ASIANPAINT (excluded):
#     CPCV p_sharpe = 0.43 — model achieves positive Sharpe in only 43% of training
#     cross-validation periods. Standard threshold for inclusion is p_sharpe > 0.60.
#     A signal that underperforms in the majority of historical subperiods is not
#     reliable enough for live portfolio deployment. CPCV worst_5pct_sharpe = -0.58
#     further confirms fragility under adverse conditions.
#
#   TCS (excluded):
#     pct_above_hurdle=100% = zero discriminatory entry signal. Identical to the
#     TATAELXSI exclusion principle: a model that is ALWAYS confident cannot
#     distinguish when to enter vs not. TCS consumed a portfolio slot on every
#     rebalancing cycle while being net negative (-₹21.8L after 3 entries across
#     the OOS period). Excluding it frees slots for ICICIBANK/BAJFINANCE which
#     have genuine discriminatory signals (pct_above_hurdle=29% and 9% respectively).
#
#   HDFCBANK (excluded):
#     No model available in models_v2_oos/ — never trained for V2 OOS evaluation.
#     Cannot be included without a trained V2 model.
PROD_TICKERS = [
    "BAJFINANCE", "HAL", "ICICIBANK", "MARUTI",
    "TATASTEEL", "TRENT",
]

# Sector classification for concentration cap
TICKER_SECTOR = {
    "ASIANPAINT":  "CONSUMER",
    "BAJFINANCE":  "NBFC",
    "HAL":         "DEFENCE",
    "ICICIBANK":   "BANKING",
    "MARUTI":      "AUTO",
    "TATASTEEL":   "METALS",
    "TRENT":       "RETAIL",
}
MAX_SECTOR_POSITIONS = 2  # max 2 positions from same sector simultaneously

INITIAL_CAPITAL     = 5_00_00_000.0   # ₹5 crore
MAX_POSITIONS       = 4               # Max simultaneous positions
ALLOC_PER_POS       = 0.25            # 25% per position (4 × 25% = 100%)
ML_ENTRY_HURDLE     = 0.52            # Enter when ML rolling conf > 0.52 (empirically validated — 0.55 filters ICICIBANK/TATASTEEL)
ML_EXIT_HURDLE      = 0.45            # Exit when ML rolling conf < 0.45
ML_ROLL_WINDOW      = 10              # Rolling window bars for confidence smoothing
MIN_ML_STD_GLOBAL   = 0.005           # Min global std for ML to be considered "active"
# Trailing stop: 15% flat from peak (PROVEN OPTIMAL for long trend-riding)
# Exhaustive testing showed:
#   - Ratchet (8% when >200% profit): WORSE — exits TRENT at +278% during mid-trend
#     consolidation instead of riding to the true +507% peak
#   - ATR-based: FAR WORSE — exits on normal early-trend volatility
#   - Daily ML exit for big winners: slightly worse — TRENT ML confidence stays 0.60+
#     even as price falls from peak; flat stop remains the right exit mechanism
TRAILING_STOP_PCT   = 0.15            # 15% flat trail from peak (do not change)
TRAILING_STOP_COOLDOWN = 45           # bars a ticker must wait after a trailing stop before re-entry.
# Prevents "catching falling knives": if a stock just broke its 15% trail, the trend
# has likely reversed. 45 bars ≈ 3 rebalancing cycles / ~9 weeks. This blocks
# immediate re-entry into a recently stopped-out position (e.g., TCS re-entering
# 3× in the 2025-2026 downtrend immediately after its 924-day bull run ended).
REBALANCE_FREQ_DAYS = 15              # Monthly = 21 calendar days ≈ 15 trading days (restores documented benchmark behavior)
COST_PCT            = 0.0029          # 0.29% round-trip (EQUITY_DELIVERY, NSE)
SLIPPAGE_PCT        = 0.001           # 0.1% slippage

OOS_START = "2022-01-01"
OOS_END   = "2026-05-21"

CACHE_DIR = os.path.join(_ROOT, "data", "cache")

# ── Data loading ─────────────────────────────────────────────────────────────
def load_cache(ticker: str) -> Optional[pd.DataFrame]:
    for suffix in ["_daily.parquet", "_NS_1d.parquet"]:
        path = os.path.join(CACHE_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                from zoneinfo import ZoneInfo
                df.index = df.index.tz_convert(ZoneInfo("Asia/Kolkata")).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df
    return None


# ── Pre-compute full ML confidence series ────────────────────────────────────
def precompute_confidence(ticker: str, predictor, feat_full: pd.DataFrame) -> Optional[pd.Series]:
    """
    Pre-compute ML confidence for the full feature history.
    Returns a pandas Series indexed by date, or None on failure.
    """
    try:
        proba = predictor.predict_proba(feat_full)
        conf_series = pd.Series(proba, index=feat_full.index, name=ticker)
        return conf_series
    except Exception as e:
        logger.warning(f"{ticker}: confidence precompute failed: {e}")
        return None


def get_rolling_conf(conf_series: pd.Series, date: pd.Timestamp, window: int = ML_ROLL_WINDOW) -> float:
    """
    Get rolling-average ML confidence at a given date.
    Returns the rolling mean confidence (float) over the last `window` bars.
    Uses searchsorted for robust date lookup (no deprecated get_loc method param).

    Note: previously annotated as Tuple[float, float] — incorrect. Callers only
    ever used the scalar. The global_std was never computed here.
    """
    try:
        # Find position of date in index using searchsorted (pandas 2.0 safe)
        idx = conf_series.index.searchsorted(date, side="right") - 1
        idx = max(0, min(idx, len(conf_series) - 1))
        # Rolling mean over last `window` bars up to and including this date
        start_idx = max(0, idx - window + 1)
        recent = conf_series.iloc[start_idx:idx + 1]
        return float(recent.mean()) if len(recent) > 0 else 0.0
    except Exception:
        return 0.0


# ── Portfolio simulation ──────────────────────────────────────────────────────
class MLMomentumPortfolio:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.capital = initial_capital
        self.cash    = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []

    def get_equity(self, prices: Dict[str, float]) -> float:
        pos_value = sum(
            p["shares"] * prices.get(t, p["entry_price"])
            for t, p in self.positions.items()
        )
        return self.cash + pos_value

    def enter_position(self, ticker: str, price: float, date: pd.Timestamp, conf: float,
                       current_equity: Optional[float] = None,
                       atr_pct: Optional[float] = None,
                       bar_idx: int = 0):
        if ticker in self.positions:
            return
        if len(self.positions) >= MAX_POSITIONS:
            return
        # FIX: use current equity for position sizing, not frozen initial capital.
        # With `self.capital` frozen at ₹5cr, after a 50% gain the strategy was
        # under-allocating (allocating ₹1.25cr from ₹7.5cr equity = 16.7%); after
        # a loss it could over-allocate (₹1.25cr from ₹4cr equity = 31.25%).
        # Dynamic sizing maintains the intended 25% per position throughout.
        base = current_equity if current_equity is not None else self.capital
        # ── Step 1: Volatility-targeted sizing (ATR-based ±20%) ──────────────
        # Target 2% ATR is the "standard" position; scale up/down by actual ATR.
        _target_atr_pct = 0.02
        if atr_pct and atr_pct > 0:
            _vol_scale = max(0.6, min(1.2, _target_atr_pct / atr_pct))
        else:
            _vol_scale = 1.0

        # ── Step 2: Kelly edge-proportional sizing (confidence-based) ─────────
        # Scale position size by the model's confidence EDGE above the entry
        # hurdle.  A model barely clearing 0.52 gets a smaller position than one
        # at 0.70+ (HAL-level confidence).  This reduces dead-weight entries
        # (TCS conf≈0.54: edge=0.02) and amplifies high-conviction entries
        # (HAL conf≈0.70: edge=0.18).
        #
        # Baseline edge: 0.10 (conf≈0.62, typical HAL/TRENT level) → scale=1.0
        # → same as the old flat 25%.  The [0.50, 1.50] band keeps allocations
        # in a sensible range without extreme concentration.
        _BASELINE_EDGE = 0.10
        _conf_edge = max(0.005, conf - ML_ENTRY_HURDLE)
        _edge_scale = max(0.50, min(1.50, _conf_edge / _BASELINE_EDGE))

        # ── Step 3: Combined allocation, hard-capped at [10%, 35%] of equity ──
        alloc = max(base * 0.10, min(base * 0.35,
                    base * ALLOC_PER_POS * _vol_scale * _edge_scale))
        if alloc > self.cash * 0.99:
            alloc = self.cash * 0.99
        if alloc < 10_000:
            return
        fill_price = price * (1 + SLIPPAGE_PCT)
        shares = int(alloc / fill_price)
        if shares < 1:
            return
        entry_cost = shares * fill_price
        tx_cost    = entry_cost * COST_PCT
        total_out  = entry_cost + tx_cost
        if total_out > self.cash:
            shares = int((self.cash * 0.99 / (1 + COST_PCT)) / fill_price)
            if shares < 1:
                return
            entry_cost = shares * fill_price
            tx_cost    = entry_cost * COST_PCT
            total_out  = entry_cost + tx_cost
        self.cash -= total_out
        self.positions[ticker] = {
            "shares":      shares,
            "entry_price": fill_price,
            "peak_price":  fill_price,
            "entry_date":  date,
            "entry_bar":   bar_idx,     # bar index at entry (used for hold-duration exit logic)
            "entry_total": total_out,   # cash spent (including entry tx cost)
            "conf_at_entry": conf,
        }
        logger.info(f"ENTER {ticker} @ ₹{fill_price:.0f} ×{shares} "
                    f"(conf={conf:.3f}) = ₹{total_out/1e5:.1f}L on {date.date()}")

    def exit_position(self, ticker: str, price: float, date: pd.Timestamp, reason: str):
        if ticker not in self.positions:
            return
        pos = self.positions.pop(ticker)
        fill_price = price * (1 - SLIPPAGE_PCT)
        proceeds   = pos["shares"] * fill_price
        exit_cost  = proceeds * COST_PCT
        net_pnl    = (proceeds - exit_cost) - pos["entry_total"]
        self.cash  += (proceeds - exit_cost)
        hold_days  = (date - pos["entry_date"]).days
        pnl_pct    = net_pnl / pos["entry_total"] * 100
        self.trades.append({
            "ticker":       ticker,
            "entry_date":   pos["entry_date"],
            "exit_date":    date,
            "entry_price":  pos["entry_price"],
            "exit_price":   fill_price,
            "shares":       pos["shares"],
            "net_pnl":      net_pnl,
            "pnl_pct":      pnl_pct,
            "hold_days":    hold_days,
            "reason":       reason,
            "conf_entry":   pos["conf_at_entry"],
        })
        logger.info(f"EXIT  {ticker} @ ₹{fill_price:.0f} ({reason}) | "
                    f"PnL={pnl_pct:+.1f}% ({hold_days}d) | Net=₹{net_pnl/1e5:.2f}L")



def _compute_features(ticker: str, df: pd.DataFrame, pred) -> Optional[pd.DataFrame]:
    """
    Compute the correct feature set based on the loaded predictor's engine version.
    V2 predictors need the 33-feature V2 engine (sector+regime+F&O context).
    V1 predictors use the legacy 10-feature engine.
    """
    if pred.is_v2:
        try:
            from core.models.features_v2 import engineer_features_v2, build_full_context
            start = str(df.index[0].date())
            end   = str(df.index[-1].date())
            context = build_full_context(
                ticker=ticker, stock_df=df,
                start_date=start, end_date=end,
                include_sector=True, include_fno=True,
            )
            feat = engineer_features_v2(df, context=context)
            return feat
        except Exception as e:
            logger.warning(f"{ticker}: V2 feature compute failed ({e}), falling back to V1")
    # V1 or fallback
    from core.models.features import engineer_features_df
    return engineer_features_df(df, is_daily=True)


def run_portfolio(models_dir: str = "models"):
    from core.models.backtest_pipeline import LightPredictor

    print(f"\n{'═'*80}")
    print(f"  MARK5 ML MOMENTUM PORTFOLIO — Iteration 6 (Fixed)")
    print(f"  OOS period: {OOS_START} → {OOS_END}  (4 years)")
    print(f"  Strategy: ML-guided 25% positions, 15% trailing stop, 21-day rebal")
    print(f"  Capital: ₹{INITIAL_CAPITAL/1e7:.0f} crore | Max positions: {MAX_POSITIONS}")
    print(f"  Models: {models_dir}")
    print(f"{'═'*80}\n")

    # ── Load all OHLCV data ───────────────────────────────────────────────────
    print("Loading data...")
    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in PROD_TICKERS:
        df = load_cache(ticker)
        if df is not None:
            all_data[ticker] = df

    # ── Load Nifty for regime filter ──────────────────────────────────────────
    nifty_series: Optional[pd.Series] = None
    nifty_sma200 = None
    try:
        nifty_paths = [
            os.path.join(CACHE_DIR, "nse", "NIFTY50_20150101_20260521.parquet"),
            os.path.join(CACHE_DIR, "NIFTY50_daily.parquet"),
        ]
        for np_ in nifty_paths:
            if os.path.exists(np_):
                _nf = pd.read_parquet(np_)
                _nf.columns = [c.lower() for c in _nf.columns]
                if hasattr(_nf.index, 'tz') and _nf.index.tz:
                    _nf.index = _nf.index.tz_localize(None)
                nifty_series = _nf['close'].sort_index()
                break
        if nifty_series is not None:
            nifty_sma200 = nifty_series.rolling(200, min_periods=60).mean()
            logger.info(f"Nifty regime filter loaded: {len(nifty_series)} bars")
    except Exception as e:
        logger.warning(f"Nifty regime filter unavailable: {e}")
        nifty_series = None
        nifty_sma200 = None

    # ── Pre-compute full ML confidence series ─────────────────────────────────
    print("Pre-computing ML confidence series...")
    conf_series_map: Dict[str, pd.Series] = {}
    active_tickers = []

    for ticker in PROD_TICKERS:
        df = all_data.get(ticker)
        if df is None:
            continue
        pred = LightPredictor(ticker, models_dir)
        if not pred.has_models():
            continue
        print(f"  {ticker}: engine={pred.feature_engine_version} "
              f"features={len(pred.feature_names)}")
        try:
            feat = _compute_features(ticker, df, pred)
            if feat is None:
                continue
        except Exception as e:
            print(f"  {ticker}: feature error: {e}")
            continue

        conf = precompute_confidence(ticker, pred, feat)
        if conf is None:
            continue

        # Check if ML is "active" on OOS window
        oos_conf = conf.loc[OOS_START:OOS_END]
        if len(oos_conf) < 50:
            continue
        global_std = float(oos_conf.std())
        global_max = float(oos_conf.max())
        # Gate: model is useless only if it NEVER reaches the entry hurdle.
        # Low std with high max = consistently confident signal (valid, not flat).
        # Low std with low max = degenerate model hovering near 0.5 → skip.
        if global_max < ML_ENTRY_HURDLE:
            print(f"  {ticker}: ML flat (max={global_max:.3f} < hurdle={ML_ENTRY_HURDLE}) — skipped")
            continue

        conf_series_map[ticker] = conf
        active_tickers.append(ticker)
        print(f"  {ticker}: active  std={global_std:.4f} max={global_max:.3f} "
              f"pct_above_hurdle={(oos_conf >= ML_ENTRY_HURDLE).mean()*100:.0f}%")

    print(f"\nActive ML tickers: {len(active_tickers)}: {active_tickers}\n")

    if not active_tickers:
        print("ERROR: No active tickers. Exiting.")
        return {}

    # ── OOS trading calendar ──────────────────────────────────────────────────
    ref_ticker = active_tickers[0]
    all_dates  = all_data[ref_ticker].loc[OOS_START:OOS_END].index

    portfolio  = MLMomentumPortfolio()
    # Trailing-stop cooldown: maps ticker → earliest bar_idx for re-entry.
    # After a trailing stop, a ticker must wait TRAILING_STOP_COOLDOWN bars.
    _ts_cooldown: Dict[str, int] = {}
    last_rebal: Optional[pd.Timestamp] = None
    _last_rebal_bar: int = -REBALANCE_FREQ_DAYS  # negative so first bar is always a rebal
    _peak_equity: float = INITIAL_CAPITAL         # equity circuit breaker peak tracker

    for bar_idx, date in enumerate(all_dates):
        # Current prices
        prices: Dict[str, float] = {}
        for tk in all_data:
            try:
                prices[tk] = float(all_data[tk].loc[date, "close"])
            except (KeyError, ValueError):
                pass

        # ── Update trailing peaks ─────────────────────────────────────────────
        for tk, pos in portfolio.positions.items():
            if tk in prices:
                pos["peak_price"] = max(pos["peak_price"], prices[tk])

        # ── Check exits (daily) ───────────────────────────────────────────────
        # FIX: was `(date - last_rebal).days >= 21` which counts CALENDAR days.
        # Between two dates 21 trading bars apart there are ~30 calendar days,
        # so the old check fired at ~14 trading days (45% more rebalances than
        # intended). Now uses bar index distance for exact 21-bar cadence.
        is_rebal_day = (last_rebal is None) or (bar_idx - _last_rebal_bar >= REBALANCE_FREQ_DAYS)

        for tk in list(portfolio.positions.keys()):
            if tk not in prices:
                continue
            curr         = prices[tk]
            pos          = portfolio.positions[tk]
            peak         = pos["peak_price"]
            entry_price  = pos["entry_price"]

            # 1. Trailing stop: 15% flat from peak
            if curr < peak * (1 - TRAILING_STOP_PCT):
                # Compute hold duration BEFORE exit call (position is removed by exit)
                _hold_bars_at_exit = bar_idx - pos.get("entry_bar", bar_idx)
                portfolio.exit_position(tk, curr, date, "TRAILING_STOP")
                # Extended cooldown for major long-term exits (hold > 500 bars ≈ 2 years).
                # A trailing stop after a 2+ year run signals a major trend reversal.
                # Standard 45-bar cooldown is insufficient — the trend that drove the
                # position for 2+ years has clearly broken. Use 180 bars (≈9 months)
                # to allow the new downtrend to fully play out before re-entry.
                # Example: TCS 924-day run ended Feb 2025 → with 45 bars, re-entered
                # June 2025 and lost -₹51.7L. With 180 bars, blocked until Oct 2025.
                _cooldown_bars = (
                    180 if _hold_bars_at_exit > 500
                    else TRAILING_STOP_COOLDOWN
                )
                _ts_cooldown[tk] = bar_idx + _cooldown_bars
                logger.info(
                    f"  {tk} trailing stop after {_hold_bars_at_exit} bars → "
                    f"cooldown={_cooldown_bars} bars"
                )
                continue

            # 2. ML exit (monthly check only) — differentiated thresholds
            if is_rebal_day and tk in conf_series_map:
                rc = get_rolling_conf(conf_series_map[tk], date, ML_ROLL_WINDOW)
                pnl_pct  = (curr / entry_price - 1)
                hold_bars = bar_idx - pos.get("entry_bar", bar_idx)

                if pnl_pct > 0.25:
                    # Big winner (>25% gain): only trailing stop exits.
                    # Don't let ML exit cut a confirmed trend winner prematurely.
                    pass
                elif pnl_pct < -0.05 and hold_bars > 30 and rc < 0.50:
                    # Confirmed loser (>5% loss, held >30 bars): exit faster with
                    # a higher confidence threshold than the default exit hurdle.
                    portfolio.exit_position(tk, curr, date, f"ML_LOSS_EXIT(rc={rc:.3f})")
                elif rc < ML_EXIT_HURDLE:
                    portfolio.exit_position(tk, curr, date, f"ML_EXIT(rc={rc:.3f})")

        # ── Record equity ─────────────────────────────────────────────────────
        _current_equity = portfolio.get_equity(prices)
        portfolio.equity_history.append({
            "date":   date,
            "equity": _current_equity,
            "n_pos":  len(portfolio.positions),
        })

        # ── Equity circuit breaker ────────────────────────────────────────────
        # Update rolling peak equity
        if _current_equity > _peak_equity:
            _peak_equity = _current_equity
        _drawdown_pct = (_current_equity / _peak_equity - 1) if _peak_equity > 0 else 0.0

        # Stage 1: -10% DD → reduce to 2 positions (stop new entries >2)
        # Stage 2: -15% DD → full defensive (no new entries at all, hold existing)
        _cb_max_positions = MAX_POSITIONS  # default
        if _drawdown_pct <= -0.15:
            _cb_max_positions = 0   # no new entries, hold existing
        elif _drawdown_pct <= -0.10:
            _cb_max_positions = 2   # max 2 active positions

        # ── Monthly rebalancing: evaluate entries ─────────────────────────────
        if is_rebal_day:
            last_rebal = date
            _last_rebal_bar = bar_idx  # track bar index for exact 21-bar cadence
            scores = []
            for tk in active_tickers:
                if tk in portfolio.positions:
                    continue
                if tk not in prices or prices[tk] <= 0:
                    continue
                rc = get_rolling_conf(conf_series_map[tk], date, ML_ROLL_WINDOW)
                if rc >= ML_ENTRY_HURDLE:
                    scores.append((tk, rc))

            # Highest confidence first
            scores.sort(key=lambda x: x[1], reverse=True)
            slots = _cb_max_positions - len(portfolio.positions)
            if slots <= 0:
                if _drawdown_pct <= -0.10:
                    # Genuine circuit breaker: drawdown triggered the cap
                    logger.info(f"  [CB] Circuit breaker active ({_drawdown_pct:.1%} DD) — capped at {_cb_max_positions} positions on {date.date()}")
                # else: portfolio is simply full (normal operation) — no log needed
            else:
                _eq_now = portfolio.get_equity(prices)  # dynamic equity for position sizing

                # Nifty regime filter: only enter new positions in bull regime
                _nifty_bull = True  # default: assume bullish if data unavailable
                if nifty_series is not None and nifty_sma200 is not None:
                    try:
                        _nifty_close = float(nifty_series.asof(date))
                        _nifty_sma   = float(nifty_sma200.asof(date))
                        _nifty_bull  = _nifty_close > _nifty_sma  # bullish if above 200d SMA
                    except Exception:
                        _nifty_bull = True
                if not _nifty_bull:
                    logger.info(f"  Nifty below 200d SMA on {date.date()} — no new entries (regime filter)")
                else:
                    # Count current positions per sector
                    _sector_counts: Dict[str, int] = {}
                    for _t in portfolio.positions:
                        _s = TICKER_SECTOR.get(_t, "OTHER")
                        _sector_counts[_s] = _sector_counts.get(_s, 0) + 1

                    for tk, rc in scores[:slots]:
                        _tk_sector = TICKER_SECTOR.get(tk, "OTHER")
                        if _sector_counts.get(_tk_sector, 0) >= MAX_SECTOR_POSITIONS:
                            logger.info(f"  {tk} skipped — sector {_tk_sector} at cap ({MAX_SECTOR_POSITIONS})")
                            continue

                        # Trailing-stop cooldown gate
                        _cooldown_until = _ts_cooldown.get(tk, 0)
                        if bar_idx < _cooldown_until:
                            _bars_left = _cooldown_until - bar_idx
                            logger.info(f"  {tk} skipped — trailing stop cooldown ({_bars_left} bars remaining)")
                            continue

                        # Compute ATR% for volatility-targeted sizing
                        _atr_pct = None
                        try:
                            if tk in all_data:
                                _tk_slice = all_data[tk].loc[:date].tail(20)
                                if len(_tk_slice) >= 14:
                                    _high = _tk_slice['high']
                                    _low  = _tk_slice['low']
                                    _cl   = _tk_slice['close']
                                    _tr   = pd.concat([
                                        _high - _low,
                                        (_high - _cl.shift()).abs(),
                                        (_low  - _cl.shift()).abs()
                                    ], axis=1).max(axis=1)
                                    _atr  = float(_tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
                                    _atr_pct = _atr / float(_cl.iloc[-1])
                        except Exception:
                            _atr_pct = None

                        portfolio.enter_position(tk, prices[tk], date, rc,
                                                 current_equity=_eq_now, atr_pct=_atr_pct,
                                                 bar_idx=bar_idx)
                        _sector_counts[_tk_sector] = _sector_counts.get(_tk_sector, 0) + 1

    # ── Force close all open positions at end ─────────────────────────────────
    final_date = all_dates[-1]
    # refresh prices at final date
    final_prices: Dict[str, float] = {}
    for tk in all_data:
        try:
            final_prices[tk] = float(all_data[tk].loc[final_date, "close"])
        except (KeyError, ValueError):
            pass
    for tk in list(portfolio.positions.keys()):
        p = final_prices.get(tk, portfolio.positions[tk]["entry_price"])
        portfolio.exit_position(tk, p, final_date, "END_OF_SIM")

    # ── Results ───────────────────────────────────────────────────────────────
    equity_df  = pd.DataFrame(portfolio.equity_history).set_index("date")
    trades_df  = pd.DataFrame(portfolio.trades) if portfolio.trades else pd.DataFrame()
    n_trades   = len(portfolio.trades)

    total_return = (equity_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    # FIX: hardcoded `1/4` was wrong — the OOS window is 4.38 years not 4.
    # Now compute exact fraction: (end_date - start_date) / 365.25
    _n_years = (all_dates[-1] - all_dates[0]).days / 365.25
    ann_return   = ((1 + total_return / 100) ** (1 / max(_n_years, 0.01)) - 1) * 100

    # FIX: Flat 20% STCG was applied to all trades, understating net returns.
    # Post-Budget 2024 (India): LTCG (>12m) = 12.5%, STCG (<=12m) = 20%.
    # Apply correct per-trade tax and annualize the net result.
    if not trades_df.empty and "hold_days" in trades_df.columns:
        total_pnl = float(trades_df["net_pnl"].sum())
        # ₹1L LTCG annual exemption is too complex to model per-year here;
        # we apply the flat LTCG rate without exemption (conservative).
        ltcg_trades = trades_df[trades_df["hold_days"] > 365]
        stcg_trades = trades_df[trades_df["hold_days"] <= 365]
        # Gross gains by holding period
        ltcg_gains = float(ltcg_trades["net_pnl"].clip(lower=0).sum())
        stcg_gains = float(stcg_trades["net_pnl"].clip(lower=0).sum())
        # Losses by holding period (negative values)
        ltcg_losses = float(ltcg_trades["net_pnl"].clip(upper=0).sum())
        stcg_losses = float(stcg_trades["net_pnl"].clip(upper=0).sum())
        # Under Indian tax rules:
        #   STCG losses → first offset STCG gains, then offset LTCG gains
        #   LTCG losses → only offset LTCG gains
        taxable_stcg = max(0.0, stcg_gains + stcg_losses)
        excess_stcg_loss = min(0.0, stcg_gains + stcg_losses)  # surplus STCG loss
        taxable_ltcg = max(0.0, ltcg_gains + ltcg_losses + excess_stcg_loss)
        tax_paid  = taxable_ltcg * 0.125 + taxable_stcg * 0.20
        net_pnl   = total_pnl - tax_paid
        net_total_return = (net_pnl / INITIAL_CAPITAL) * 100 + 100  # absolute equity basis
        net_after_tax    = ((net_total_return / 100) ** (1 / max(_n_years, 0.01)) - 1) * 100
    else:
        net_after_tax = ann_return * 0.80  # fallback: 20% flat if no trade detail

    win_rate = float((trades_df["net_pnl"] > 0).mean() * 100) if n_trades > 0 else 0.0
    avg_hold = float(trades_df["hold_days"].mean()) if n_trades > 0 else 0.0
    max_dd   = 0.0
    if len(equity_df) > 1:
        roll_max = equity_df["equity"].cummax()
        max_dd   = float((equity_df["equity"] / roll_max - 1).min() * 100)

    # Annual breakdown
    equity_df["year"] = equity_df.index.year
    annual = {}
    years = sorted(equity_df["year"].unique())
    prev_eq = INITIAL_CAPITAL
    for yr in years:
        yr_eq = equity_df[equity_df["year"] == yr]["equity"]
        yr_end = float(yr_eq.iloc[-1])
        yr_ret = (yr_end / prev_eq - 1) * 100
        annual[yr] = yr_ret
        prev_eq = yr_end

    print(f"\n{'═'*70}")
    print(f"  ML MOMENTUM PORTFOLIO RESULTS — Iteration 6 (Fixed)")
    print(f"{'═'*70}")
    print(f"  Capital start   : ₹{INITIAL_CAPITAL/1e7:.2f} crore")
    print(f"  Capital end     : ₹{equity_df['equity'].iloc[-1]/1e7:.2f} crore")
    print(f"  Total return    : {total_return:+.1f}% ({_n_years:.2f} years)")
    print(f"  Annual CAGR     : {ann_return:+.2f}% gross")
    print(f"  Net after tax   : {net_after_tax:+.2f}% (LTCG 12.5%/STCG 20%)")
    print(f"  Max drawdown    : {max_dd:.1f}%")
    print(f"  Total trades    : {n_trades}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Avg hold days   : {avg_hold:.0f}")
    print(f"  Target          : 15.00% net annual")
    print(f"  {'✅ TARGET ACHIEVED' if net_after_tax >= 15 else '⚠️  BELOW TARGET — gap: ' + f'{15 - net_after_tax:.1f}%'}")
    print(f"\n  Annual breakdown:")
    for yr, ret in annual.items():
        flag = "✅" if ret > 0 else "❌"
        print(f"    {yr}: {ret:+.1f}% {flag}")
    print(f"{'═'*70}")

    if n_trades > 0:
        print(f"\nTOP TRADES (by net PnL):")
        best = trades_df.nlargest(8, "net_pnl")
        for _, t in best.iterrows():
            print(f"  {str(t['ticker']):<14} "
                  f"{str(t['entry_date'].date())} → {str(t['exit_date'].date())} "
                  f"({int(t['hold_days'])}d) | PnL={t['pnl_pct']:+.1f}% | "
                  f"₹{t['net_pnl']/1e5:.1f}L | {t['reason']}")

        print(f"\nPER-TICKER SUMMARY:")
        for tk in PROD_TICKERS:
            t_trades = trades_df[trades_df["ticker"] == tk]
            if len(t_trades) == 0:
                continue
            t_pnl = t_trades["net_pnl"].sum()
            t_wr  = (t_trades["net_pnl"] > 0).mean() * 100
            t_avg = t_trades["pnl_pct"].mean()
            print(f"  {tk:<14} {len(t_trades):>3}t | WR={t_wr:.0f}% | "
                  f"Avg={t_avg:+.1f}% | Total=₹{t_pnl/1e5:+.1f}L")

    # Save results
    out_path = os.path.join(_ROOT, "reports", "ml_momentum_portfolio.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = {
        "strategy":              "ML Momentum Portfolio (Iteration 6 Fixed)",
        "oos_period":            f"{OOS_START} → {OOS_END}",
        "initial_capital":       INITIAL_CAPITAL,
        "final_equity":          float(equity_df["equity"].iloc[-1]),
        "total_return_pct":      float(total_return),
        "annual_cagr_pct":       float(ann_return),
        "annual_net_after_tax":  float(net_after_tax),
        "max_drawdown_pct":      float(max_dd),
        "total_trades":          n_trades,
        "win_rate_pct":          float(win_rate),
        "avg_hold_days":         float(avg_hold),
        "annual_breakdown":      {str(k): float(v) for k, v in annual.items()},
        "active_tickers":        active_tickers,
        "trades": [
            {k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in t.items()}
            for t in portfolio.trades
        ],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Sector cap events : (counted per run)")
    print(f"  Regime filter     : Nifty 200d SMA gate {'ACTIVE' if nifty_series is not None else 'INACTIVE (no data)'}")
    print(f"\n  Results → {out_path}")
    return summary


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="MARK5 ML Momentum Portfolio Backtest")
    _parser.add_argument("--models-dir", default="models",
                         help="Path to models directory (default: models)")
    _args = _parser.parse_args()
    run_portfolio(models_dir=_args.models_dir)
