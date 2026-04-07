#!/usr/bin/env python3
"""
MARK5 HISTORICAL BACKTEST RUNNER v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fetches historical OHLCV from Kite, runs the MARK5 ML predictor
day-by-day, and simulates through the CHANAKYA backtesting engine
with full NSE tax modeling.

Usage:
    python3 scripts/run_historical_backtest.py
    python3 scripts/run_historical_backtest.py --symbols COFORGE HAL IDFCFIRSTB
    python3 scripts/run_historical_backtest.py --years 2 --capital 200000
"""

import os
import sys
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load .env before any os.getenv() calls
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
except ImportError:
    pass

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# MARK5 High-Fidelity Context Providers
from core.models.features import AdvancedFeatureEngine
from core.data.market_data import MarketDataProvider
from core.data.fii_data import FIIDataProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MARK5.Backtest")

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT UNIVERSE — stocks with trained models in /models/
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    # IT Midcap
    "COFORGE", "PERSISTENT", "MPHASIS", "KPITTECH", "LTTS",
    # Capital Goods / Defence
    "HAL", "BEL", "POLYCAB", "DIXON", "ABB",
    # Financials Midcap
    "IDFCFIRSTB", "LICHSGFIN", "MUTHOOTFIN", "CHOLAFIN", "ABCAPITAL",
    # Consumer / Retail
    "IRCTC", "JUBLFOOD", "PAGEIND", "MARICO", "COLPAL",
    # Chemicals / Pharma
    "PIIND", "DEEPAKNTR", "AARTIIND", "LAURUSLABS", "GRANULES",
    # Real Estate / Infra
    "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "CONCOR", "CUMMINSIND",
]


def parse_args():
    p = argparse.ArgumentParser(description="MARK5 Historical Backtest")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="NSE symbols to test (default: 15-stock universe)")
    p.add_argument("--years", type=float, default=3.0,
                   help="Years of history to fetch (default: 3)")
    p.add_argument("--capital", type=float, default=100_000.0,
                   help="Starting capital in ₹ (default: 1,00,000)")
    p.add_argument("--segment", default="EQUITY_DELIVERY",
                   choices=["EQUITY_DELIVERY", "EQUITY_INTRADAY", "FUTURES"],
                   help="NSE tax segment (default: EQUITY_DELIVERY)")
    p.add_argument("--confidence-threshold", type=float, default=0.55,
                   help="Min model confidence to take a signal (default: 0.55)")
    p.add_argument("--no-kite", action="store_true",
                   help="Skip Kite data fetch; use cached CSVs in data/cache/")
    p.add_argument("--save-csv", action="store_true",
                   help="Save per-symbol results to reports/backtest/")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_or_load(symbol: str, years: float, adapter=None) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV from Kite, or load from cache if adapter is None.
    Kite-only policy — no yfinance fallback.
    """
    cache_path = Path(PROJECT_ROOT) / "data" / "cache" / f"{symbol}_daily.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Kite first
    if adapter is not None:
        try:
            df = adapter.fetch_ohlcv(symbol, period=f"{int(years * 365)}d", interval="day")
            if df is not None and len(df) >= 200:
                df.to_parquet(cache_path)
                logger.info(f"{symbol}: {len(df)} bars from Kite ✅")
                return df
        except Exception as e:
            logger.warning(f"{symbol}: Kite fetch failed ({e})")

    # Try cache (from a previous Kite session)
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"{symbol}: {len(df)} bars from cache 📂")
            return df
        except Exception:
            pass

    # Kite-only policy — no yfinance
    logger.error(
        f"{symbol}: No data available. Kite not connected and no cache found. "
        f"Run generate_kite_token.py to refresh your access token."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION — MARK5 ML Predictor
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(
    symbol: str, 
    df: pd.DataFrame, 
    min_history: int = 50, 
    confidence_threshold: float = 0.55,
    context: Optional[Dict] = None
) -> pd.Series:
    """
    Run MARK5Predictor over rolling windows of historical data to generate
    a daily signal Series.

    Returns pd.Series with values:
        1  = BUY signal
        0  = HOLD
    """
    from core.models.predictor import MARK5Predictor

    signals = pd.Series(0, index=df.index, dtype=int)

    # Models on disk use .NS suffix — resolve the right ticker name
    models_root = Path(PROJECT_ROOT) / "models"
    ns_symbol   = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    bare_symbol  = symbol.replace(".NS", "")

    ticker_name = None
    for candidate in [bare_symbol, ns_symbol]:
        if (models_root / candidate).exists():
            ticker_name = candidate
            break
    if ticker_name is None:
        logger.warning(f"{symbol}: No model artifacts found in {models_root} — using HOLD")
        return signals

    # Try loading the predictor for this symbol
    try:
        predictor = MARK5Predictor(ticker=ticker_name)
        if predictor._container is None:
            logger.warning(f"{symbol}: Predictor loaded but has no models (no .pkl files?) — using HOLD")
            return signals
    except Exception as e:
        logger.warning(f"{symbol}: Predictor init failed ({e}) — using HOLD")
        return signals

    logger.info(f"{symbol}: Using model '{ticker_name}' — running walk-forward signals")

    # ── FAST BATCH MODE ──────────────────────────────────────────────────────
    # ── HIGH-FIDELITY FEATURE ENGINEERING ─────────────────────────────────────
    # Use the official AdvancedFeatureEngine to ensure 100% training alignment.
    # We pass the pre-fetched NIFTY and FII data as context.
    try:
        from core.models.features import AdvancedFeatureEngine
        fe = AdvancedFeatureEngine()
        
        # Build features for the entire window at once
        feature_df = fe.engineer_all_features(df, context=context)
        
        # Assemble the exact 12-feature DataFrame the model expects
        c = predictor._container
        
        # Re-index and fill NaNs to match model schema
        df_aligned = feature_df.reindex(columns=c.schema).fillna(0)
        
        missing_after = [col for col in c.schema if col not in feature_df.columns]
        if missing_after:
            logger.warning(f"{symbol}: Still missing after re-index: {missing_after}")
        else:
            logger.info(f"{symbol}: Contextual features engineered (NIFTY + FII) ✅")

        X = c.scaler.transform(df_aligned)

        n = len(X)
        # Increase minimum history for 252-day warmup features (52w high, etc)
        effective_min_hist = max(min_history, 500)
        
        for i in range(effective_min_hist, n):
            x_row = X[i].reshape(1, -1)
            try:
                model_probs = []
                for name, model in c.models.items():
                    w = c.weights.get(name, 1.0)
                    if w <= 0:
                        continue
                    p = model.predict_proba(x_row)[0]
                    model_probs.append(float(p[1]) if len(p) > 1 else float(p[0]))

                if not model_probs:
                    continue

                if c.meta_model is not None and len(model_probs) == 3:
                    meta_X = np.array(model_probs).reshape(1, -1)
                    confidence = float(c.meta_model.predict_proba(meta_X)[0, 1])
                else:
                    confidence = float(np.mean(model_probs))

                if confidence >= confidence_threshold:
                    signals.iloc[i] = 1
            except Exception:
                pass

    except Exception as e:
        import traceback
        logger.warning(f"{symbol}: Batch inference failed ({e}) — falling back to slow mode")
        logger.debug(traceback.format_exc())
        
        # Fallback: slow per-row predict()
        # Ensure slow mode also uses sufficient history
        effective_min_hist = max(min_history, 500)
        for i in range(effective_min_hist, len(df)):
            # Pass all history up to i to ensure feature engineering has depth
            window = df.iloc[:i+1] 
            try:
                # predictor.predict uses engineer_all_features internally
                result = predictor.predict(window)
                if "BUY" in result.get("signal", "") and result.get("confidence", 0) >= confidence_threshold:
                    signals.iloc[i] = 1
            except Exception:
                pass

    buy_count = (signals == 1).sum()
    effective_eligible = len(df) - max(min_history, 500)
    logger.info(f"{symbol}: {buy_count} BUY signals out of {effective_eligible} eligible days "
                f"(conf≥{confidence_threshold:.0%})")
                
    # DEBUG: Track max confidence to see how close we are getting
    try:
        max_conf = 0.0
        # Only check the last few hundred rows to avoid slow mode over whole df
        for i in range(max(effective_eligible, len(df)-200), len(df)):
            window = df.iloc[:i+1]
            try:
                result = predictor.predict(window)
                c = result.get('confidence', 0)
                max_conf = max(max_conf, c)
            except:
                pass
        logger.info(f"💡 {symbol}: Maximum confidence observed in recent history: {max_conf:.1%}")
    except:
        pass
        
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(symbol: str, metrics: dict):
    trades = metrics.get("trades", [])
    n = metrics.get("Total Trades", 0)
    print(f"\n{'─'*60}")
    print(f"  {symbol}")
    print(f"{'─'*60}")
    print(f"  Total Return    : {metrics.get('Total Return %', 0):+.2f}%")
    print(f"  Win Rate        : {metrics.get('Win Rate %', 0):.1f}%")
    print(f"  Profit Factor   : {metrics.get('Profit Factor', 0):.2f}")
    print(f"  Max Drawdown    : {metrics.get('Max Drawdown %', 0):.2f}%")
    print(f"  Sharpe Ratio    : {metrics.get('Sharpe Ratio', 0):.2f}")
    print(f"  Total Trades    : {n}")
    print(f"  Taxes Paid      : ₹{metrics.get('Total Taxes Paid', 0):,.2f}")


def save_results(symbol: str, equity_curve: pd.Series, metrics: dict):
    out_dir = Path(PROJECT_ROOT) / "reports" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    ec_path = out_dir / f"{symbol}_equity_{stamp}.csv"
    equity_curve.to_csv(ec_path, header=["equity"])

    trades = metrics.pop("trades", [])
    trades_df = pd.DataFrame([vars(t) for t in trades]) if trades else pd.DataFrame()
    if not trades_df.empty:
        tr_path = out_dir / f"{symbol}_trades_{stamp}.csv"
        trades_df.to_csv(tr_path, index=False)
        logger.info(f"{symbol}: saved trades → {tr_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS

    print(f"\n{'═'*60}")
    print(f"  MARK5 HISTORICAL BACKTEST")
    print(f"{'═'*60}")
    print(f"  Symbols  : {symbols}")
    print(f"  History  : {args.years} year(s)")
    print(f"  Capital  : ₹{args.capital:,.0f}")
    print(f"  Segment  : {args.segment}")
    print(f"  Conf. Thr: {args.confidence_threshold:.0%}")
    print(f"{'═'*60}\n")

    # ── Connect to Kite ──────────────────────────────────────────────────────
    adapter = None
    if not args.no_kite:
        api_key      = os.getenv("KITE_API_KEY", "")
        access_token = os.getenv("KITE_ACCESS_TOKEN", "")
        if api_key and access_token:
            try:
                from core.data.adapters.kite_adapter import KiteFeedAdapter
                adapter = KiteFeedAdapter({"api_key": api_key, "access_token": access_token,
                                           "api_secret": os.getenv("KITE_API_SECRET", "")})
                connected = adapter.connect()
                if connected:
                    logger.info("✅ Kite connected — using live historical data")
                else:
                    logger.warning("⚠️  Kite failed to connect — falling back to cache")
                    adapter = None
            except Exception as e:
                logger.warning(f"⚠️  Kite adapter error: {e} — using cache")
        else:
            logger.info("ℹ️  No Kite credentials — using cache")

    # ── Fetch Context Data (NIFTY50 + FII) ───────────────────────────────────
    context = {}
    if not args.no_kite:
        try:
            logger.info("📡 Fetching NIFTY50 and FII context for backtest...")
            from core.data.market_data import MarketDataProvider
            from core.data.fii_data import FIIDataProvider
            
            mp = MarketDataProvider()
            fp = FIIDataProvider()
            
            # Use extra 1 year to ensure Nifty/FII history covers the backtest warmup
            start_date = (datetime.now() - timedelta(days=int((args.years + 1) * 365))).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            nifty_df = mp.get_nifty50_data(start_date, end_date, kite_adapter=adapter)
            if nifty_df is not None:
                context['nifty_close'] = nifty_df['close']
                
            fii_series = fp.get_fii_flow(start_date, end_date)
            # Per Phase 3 Plan: Real-only. No synthetic fallback.
            context['fii_net'] = fii_series
                
            logger.info(f"✅ Context features initialized: {list(context.keys())}")
            if 'nifty_close' in context:
                logger.info(f"   - NIFTY50: {len(context['nifty_close'])} bars")
            if 'fii_net' in context:
                logger.info(f"   - FII Flow: {len(context['fii_net'])} days")
        except Exception as e:
            logger.warning(f"⚠️ Context fetch error: {e} — using stock-only features")

    # ── Import Backtester ────────────────────────────────────────────────────
    from core.models.tcn.backtester import RobustBacktester

    # ── Per-symbol run ───────────────────────────────────────────────────────
    all_metrics = {}

    for symbol in symbols:
        logger.info(f"\n{'━'*50}\n  Processing: {symbol}\n{'━'*50}")

        # 1. Fetch data
        df = fetch_or_load(symbol, args.years, adapter)
        if df is None or len(df) < 250:
            logger.warning(f"{symbol}: insufficient data ({len(df) if df is not None else 0} bars) — skipping")
            continue

        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                logger.warning(f"{symbol}: missing '{col}' column — skipping")
                break
        else:
            pass

        # 2. Generate ML signals
        signals = generate_signals(symbol, df,
                                   min_history=50,
                                   confidence_threshold=args.confidence_threshold,
                                   context=context)

        # 3. Run simulation
        backtester = RobustBacktester(
            initial_capital=args.capital,
            segment=args.segment,
            slippage_pct=0.0005,     # 0.05% (NIFTY50 per RULE 8)
            use_atr_stop=True,
            atr_period=14,
            atr_multiplier=2.0,      # RULE 15
            risk_per_trade=0.015,    # RULE 10: 1.5% risk per trade
        )

        try:
            equity_curve, metrics = backtester.run_simulation(df, signals)
        except Exception as e:
            logger.error(f"{symbol}: simulation failed — {e}")
            continue

        all_metrics[symbol] = metrics
        print_metrics(symbol, metrics)

        if args.save_csv:
            save_results(symbol, equity_curve, metrics)

    # ── Portfolio summary ────────────────────────────────────────────────────
    if all_metrics:
        print(f"\n{'═'*60}")
        print(f"  PORTFOLIO SUMMARY  ({len(all_metrics)} symbols)")
        print(f"{'═'*60}")
        returns   = [m.get("Total Return %", 0) for m in all_metrics.values()]
        sharpes   = [m.get("Sharpe Ratio", 0) for m in all_metrics.values()]
        drawdowns = [m.get("Max Drawdown %", 0) for m in all_metrics.values()]
        win_rates = [m.get("Win Rate %", 0) for m in all_metrics.values()]
        taxes     = [m.get("Total Taxes Paid", 0) for m in all_metrics.values()]

        print(f"  Avg Return   : {np.mean(returns):+.2f}%")
        print(f"  Avg Sharpe   : {np.mean(sharpes):.2f}")
        print(f"  Avg Max DD   : {np.mean(drawdowns):.2f}%")
        print(f"  Avg Win Rate : {np.mean(win_rates):.1f}%")
        print(f"  Total Taxes  : ₹{sum(taxes):,.2f}")
        winners = [s for s, m in all_metrics.items() if m.get("Total Return %", 0) > 0]
        print(f"  Profitable   : {len(winners)}/{len(all_metrics)} symbols")
        print(f"    → {winners}")
        print(f"{'═'*60}\n")

    if adapter:
        try:
            adapter.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
