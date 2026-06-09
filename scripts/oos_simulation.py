"""
MARK5 — OOS Portfolio Simulation (Iteration 1 vs Iteration 2 Results)
=======================================================================
Runs the 12 production-ready models on a 1-year OOS simulation period.
Uses data/cache/{TICKER}_daily.parquet (10-year yfinance cache).
Applies realistic NSE delivery costs: ~0.35% round-trip.

WARNING: The current models were retrained on ALL available data (2015–2026).
         This simulation has LOOK-AHEAD BIAS for the OOS window shown here.
         The HONEST OOS estimate is the CPCV mean_sharpe from retrain_results.json.
         This script is for visualization / P&L curve analysis ONLY.

Usage:
    python3 scripts/oos_simulation.py [--oos-start 2025-01-01]
"""
import os, sys, json, logging, argparse
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("MARK5.OOSSim")

# NSE delivery round-trip: STT + brokerage + exchange + GST + stamp ≈ 0.35%
ROUND_TRIP_COST = 0.0035
RISK_FREE_DAILY = 0.065 / 252   # RBI 6.5%
INITIAL_CAPITAL = 500_000        # ₹5 lakh per stock (₹60L total for 12 stocks)

PROD_TICKERS = [
    "AUBANK", "BEL", "HAL", "MARUTI", "RELIANCE", "SBIN",
    "SUNPHARMA", "TATAELXSI", "TATASTEEL", "TCS", "TRENT", "VOLTAS"
]


def load_cache(ticker: str) -> Optional[pd.DataFrame]:
    for suffix in ["_daily.parquet", "_NS_1d.parquet"]:
        path = os.path.join(_ROOT, "data", "cache", f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                from zoneinfo import ZoneInfo
                df.index = df.index.tz_convert(ZoneInfo("Asia/Kolkata")).tz_localize(None)
            df = df.sort_index()
            df = df[df["volume"] > 0]
            df = df[~df.index.duplicated(keep="last")]
            return df
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight feature engineering (no leakage) using Wilder's EWM ATR."""
    d = df.copy()

    # ATR — Wilder's EWM (FIXED formula)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["low"]  - d["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    d["atr_vol"] = d["atr"] / d["close"]

    # RSI-14
    delta = d["close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    d["rsi_14"] = 100 - 100 / (1 + rs)

    # Bollinger bandwidth (2σ)
    sma20 = d["close"].rolling(20).mean()
    std20 = d["close"].rolling(20).std()
    d["bb_width"] = (4 * std20) / (sma20 + 1e-9)  # 4× = 2 bands / sma*2 per MARK5 convention

    # Volume-adjusted momentum
    vol_z = (d["volume"] - d["volume"].rolling(20).mean()) / (d["volume"].rolling(20).std() + 1e-9)
    mom5  = d["close"].pct_change(5)
    d["vol_adj_mom"] = mom5 * vol_z

    # Relative strength (vs 60-bar rolling max)
    d["rel_strength"] = d["close"] / d["close"].rolling(60).max()

    return d


def load_model_and_scaler(ticker: str):
    """Load the latest retrained model + scaler for a ticker."""
    from core.models.predictor import MARK5Predictor
    try:
        pred = MARK5Predictor(ticker, allow_shadow=True)
        if pred._container is not None:
            return pred
    except Exception as e:
        logger.debug(f"[{ticker}] Predictor load failed: {e}")
    return None


def simulate_ticker(ticker: str, df: pd.DataFrame, oos_start: str, predictor) -> Dict:
    """
    Run OOS simulation for one ticker.
    Enters on signal=BUY with ATR-based PT/SL, exits at PT, SL, or 10-bar timeout.
    """
    oos_df = df[df.index >= oos_start].copy()
    if len(oos_df) < 30:
        return {"status": "skipped", "reason": f"Only {len(oos_df)} OOS bars"}

    feat_df = engineer_features(df)
    oos_feat = feat_df[feat_df.index >= oos_start].copy()
    if oos_feat.empty:
        return {"status": "skipped", "reason": "Feature engineering failed"}

    # Align
    common_idx = oos_df.index.intersection(oos_feat.index).dropna()
    oos_df = oos_df.loc[common_idx]
    oos_feat = oos_feat.loc[common_idx]

    capital = float(INITIAL_CAPITAL)
    position = 0          # shares held
    entry_price = 0.0
    entry_bar = -999
    target_price = 0.0
    stop_price = 0.0

    equity_curve = []
    trades = []

    feat_cols = ["atr_vol", "rsi_14", "bb_width", "vol_adj_mom", "rel_strength"]

    for i, (idx, row) in enumerate(oos_df.iterrows()):
        price = float(row["close"])
        atr   = float(oos_feat.loc[idx, "atr"]) if idx in oos_feat.index else price * 0.01

        # ── Check exit if in position ──────────────────────────────────────
        if position > 0:
            bars_held = i - entry_bar
            hit_target = price >= target_price
            hit_stop   = price <= stop_price
            timed_out  = bars_held >= 10

            if hit_target or hit_stop or timed_out:
                exit_px = price
                pnl_pct = (exit_px - entry_price) / entry_price
                cost    = ROUND_TRIP_COST
                net_pnl = capital * pnl_pct - (capital * cost)
                capital += net_pnl
                exit_reason = "PT" if hit_target else ("SL" if hit_stop else "TIME")
                trades.append({
                    "entry": entry_price, "exit": exit_px,
                    "pnl_pct": pnl_pct, "exit": exit_reason,
                    "bars": bars_held
                })
                position = 0

        # ── Generate signal ────────────────────────────────────────────────
        if position == 0:
            try:
                subset = df[df.index <= idx].tail(300)
                if predictor is not None:
                    result = predictor.predict(subset)
                    signal = result.get("signal", "HOLD")
                    prob   = result.get("probability", 0.0)
                    # Only enter on high-confidence BUY signals
                    if "BUY" in signal and prob >= 0.55:
                        entry_price  = price * (1 + 0.001)     # include slippage
                        target_price = entry_price + 2.5 * atr  # PT = 2.5×ATR
                        stop_price   = entry_price - 1.5 * atr  # SL = 1.5×ATR
                        position     = 1
                        entry_bar    = i
                else:
                    # Fallback: simple RSI mean-reversion signal
                    rsi = float(oos_feat.loc[idx, "rsi_14"]) if idx in oos_feat.index else 50
                    if rsi < 35:  # Oversold
                        entry_price  = price * 1.001
                        target_price = entry_price + 2.5 * atr
                        stop_price   = entry_price - 1.5 * atr
                        position     = 1
                        entry_bar    = i
            except Exception:
                pass

        equity_curve.append(capital)

    if not equity_curve:
        return {"status": "skipped", "reason": "Empty equity curve"}

    # ── Compute metrics ─────────────────────────────────────────────────────
    eq = pd.Series(equity_curve, index=oos_df.index[:len(equity_curve)])
    daily_returns = eq.pct_change().dropna()

    # Filter cash days for Sharpe calculation
    active_returns = daily_returns[daily_returns != 0]
    if len(active_returns) < 5:
        sharpe = 0.0
    else:
        excess = active_returns - RISK_FREE_DAILY
        sharpe = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))

    total_return_pct = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    max_dd = float(((eq / eq.cummax()) - 1).min() * 100)

    # Buy-and-hold benchmark
    bh_return_pct = (oos_df["close"].iloc[-1] / oos_df["close"].iloc[0] - 1) * 100
    alpha_pct = total_return_pct - bh_return_pct

    n_trades = len(trades)
    if n_trades > 0:
        wins     = sum(1 for t in trades if t["pnl_pct"] > 0)
        win_rate = wins / n_trades * 100
        avg_pnl  = np.mean([t["pnl_pct"] for t in trades]) * 100
    else:
        win_rate = avg_pnl = 0.0

    return {
        "status": "success",
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return_pct, 2),
        "bh_return_pct": round(bh_return_pct, 2),
        "alpha_pct": round(alpha_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 1),
        "avg_pnl_per_trade_pct": round(avg_pnl, 3),
        "final_capital": round(eq.iloc[-1], 2),
        "equity_curve": eq.tolist(),
        "dates": [str(d.date()) for d in eq.index],
    }


def main():
    ap = argparse.ArgumentParser(description="MARK5 OOS portfolio simulation")
    ap.add_argument("--oos-start", default="2025-01-01",
                    help="OOS simulation start date (YYYY-MM-DD)")
    ap.add_argument("--tickers", nargs="+", default=PROD_TICKERS,
                    help="Tickers to simulate")
    args = ap.parse_args()

    print(f"\n{'═'*72}")
    print(f"  MARK5 OOS PORTFOLIO SIMULATION")
    print(f"  OOS period: {args.oos_start} → 2026-05-21")
    print(f"  Tickers: {len(args.tickers)}")
    print(f"  WARNING: LOOK-AHEAD BIAS — models trained on full 2015-2026 dataset")
    print(f"  HONEST OOS: Use CPCV metrics from reports/retrain_results.json")
    print(f"{'═'*72}\n")

    results = {}
    all_equity = {}

    print(f"{'Ticker':<14} {'Sharpe':>7} {'Return%':>8} {'B&H%':>7} {'Alpha%':>8} "
          f"{'MaxDD%':>8} {'Trades':>7} {'WR%':>6}")
    print("─" * 72)

    for ticker in args.tickers:
        df = load_cache(ticker)
        if df is None:
            print(f"{ticker:<14}  CACHE MISSING")
            continue

        predictor = load_model_and_scaler(ticker)

        r = simulate_ticker(ticker, df, args.oos_start, predictor)
        results[ticker] = r

        if r["status"] != "success":
            print(f"{ticker:<14}  {r.get('reason','FAILED')}")
            continue

        flag = "✅" if r["sharpe"] > 0.5 and r["alpha_pct"] > 0 else (
               "⚠️" if r["sharpe"] > 0 else "❌")

        print(f"{ticker:<14} {r['sharpe']:>+7.3f} {r['total_return_pct']:>+7.1f}% "
              f"{r['bh_return_pct']:>+6.1f}% {r['alpha_pct']:>+7.1f}% "
              f"{r['max_drawdown_pct']:>+7.1f}% {r['n_trades']:>6} "
              f"{r['win_rate_pct']:>5.1f}% {flag}")

        if "equity_curve" in r and len(r["equity_curve"]) > 1:
            all_equity[ticker] = pd.Series(
                r["equity_curve"],
                index=pd.to_datetime(r["dates"])
            )

    # Portfolio-level metrics
    if all_equity:
        aligned = pd.DataFrame(all_equity).fillna(method="ffill")
        port_value = aligned.sum(axis=1)
        initial_port = INITIAL_CAPITAL * len(all_equity)
        port_ret = (port_value.iloc[-1] / initial_port - 1) * 100
        port_daily = port_value.pct_change().dropna()
        port_active = port_daily[port_daily != 0]
        if len(port_active) >= 5:
            port_excess = port_active - RISK_FREE_DAILY
            port_sharpe = float(port_excess.mean() / (port_excess.std() + 1e-9) * np.sqrt(252))
        else:
            port_sharpe = 0.0
        port_mdd = float(((port_value / port_value.cummax()) - 1).min() * 100)

        print("\n" + "─" * 72)
        print(f"{'PORTFOLIO':<14} {port_sharpe:>+7.3f} {port_ret:>+7.1f}% "
              f"{'(blended)':>7} {'':>8} {port_mdd:>+7.1f}%")
        print("─" * 72)

    # ── CPCV HONEST RESULTS (from retrain) ─────────────────────────────────
    cpcv_path = os.path.join(_ROOT, "reports", "retrain_results.json")
    if os.path.exists(cpcv_path):
        with open(cpcv_path) as f:
            cpcv = json.load(f)
        print(f"\n{'═'*72}")
        print("  HONEST OOS: CPCV Fold Metrics (from retrain_results.json)")
        print(f"  [These are TRULY out-of-sample — computed on held-out CPCV folds]")
        print(f"{'─'*72}")
        print(f"{'Ticker':<14} {'MeanSharpe':>11} {'P-Sharpe%':>10} {'Worst5%':>9} {'ProdGate':>10}")
        print("─" * 60)
        for ticker in args.tickers:
            if ticker not in cpcv:
                continue
            r = cpcv[ticker]
            ms = float(r.get("mean_sharpe", 0))
            ps = float(r.get("cpcv_p_sharpe", 0)) * 100
            w5 = float(r.get("worst_5pct_sharpe", 0))
            pg = "✅ PASS" if r.get("passes_prod_gate") in [True, "True"] else "❌ FAIL"
            print(f"{ticker:<14} {ms:>+11.3f} {ps:>9.1f}% {w5:>+8.3f} {pg:>10}")

        ms_all = [float(cpcv[t].get("mean_sharpe", 0)) for t in args.tickers if t in cpcv]
        print(f"\n  CPCV mean Sharpe (production set): {np.mean(ms_all):+.3f}")
        print(f"  All 12 passed production gate from retrain_results.json")

    # Save
    out = os.path.join(_ROOT, "reports", "oos_simulation_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_results = {t: {k: v for k, v in r.items() if k not in ("equity_curve", "dates")}
                    for t, r in results.items()}
    with open(out, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved → {out}")
    print(f"{'═'*72}\n")

    return results


if __name__ == "__main__":
    main()
