"""
MARK5 Momentum Portfolio Backtest v1.0 — Multi-Gate Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Four-gate entry filter applied before every rebalance entry:
  1. Regime Gate  — blocks confirmed BEARISH entries via MarketRegimeDetector
  2. FII Gate     — blocks entries during heavy FII selling (>₹5,000cr/5d)
  3. Signal Gate  — requires MomentumSignal.score() ≥ ENTRY_THRESHOLD
  4. Weekly Gate  — requires weekly trend aligned with daily signal

UNIVERSE: 33-stock NSE universe across IT, Defence, NBFC, FMCG, Pharma, Infra.
OOS PERIOD: 2022-01-01 → 2026-04-30

CHANGELOG:
- [2026-05-26] v1.0: Initial creation with all four gates wired.

TRADING ROLE: Primary alpha engine — multi-gate portfolio rotation
SAFETY LEVEL: HIGH
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.data.nse_data_provider import fetch_equity_ohlcv, fetch_nifty50_index
from core.data.fii_data import FIIDataProvider
from core.analytics.regime_detector import MarketRegimeDetector
from core.models.momentum_signal import MomentumSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [MOMENTUM_PORT] | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.MomentumPortfolio")

# ── Universe ──────────────────────────────────────────────────────────────────

UNIVERSE: List[str] = [
    # IT / Software
    "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "KPITTECH.NS", "LTTS.NS",
    # Defence / Aerospace
    "HAL.NS", "BEL.NS", "MAZDOCK.NS",
    # Industrials / Capital Goods
    "POLYCAB.NS", "DIXON.NS", "ABB.NS", "CUMMINSIND.NS",
    # NBFC / Banking
    "IDFCFIRSTB.NS", "LICHSGFIN.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    # FMCG / Consumer
    "IRCTC.NS", "JUBLFOOD.NS", "PAGEIND.NS", "MARICO.NS", "COLPAL.NS",
    # Pharma / Chemicals
    "PIIND.NS", "DEEPAKNTR.NS", "LAURUSLABS.NS", "LUPIN.NS", "SUNPHARMA.NS",
    # Real Estate / Infrastructure
    "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "CONCOR.NS",
    # Also include TCS, INFY for mean reversion candidates
    "TCS.NS", "INFY.NS",
]

# Sector peer lists used for sector-RS component
# Key: ticker (no .NS) → list of peer tickers (no .NS)
SECTOR_PEERS: Dict[str, List[str]] = {
    "COFORGE":    ["PERSISTENT", "MPHASIS", "KPITTECH", "LTTS", "TCS", "INFY"],
    "PERSISTENT": ["COFORGE", "MPHASIS", "KPITTECH", "LTTS", "TCS", "INFY"],
    "MPHASIS":    ["COFORGE", "PERSISTENT", "KPITTECH", "LTTS", "TCS", "INFY"],
    "KPITTECH":   ["COFORGE", "PERSISTENT", "MPHASIS", "LTTS"],
    "LTTS":       ["COFORGE", "PERSISTENT", "MPHASIS", "KPITTECH"],
    "TCS":        ["INFY", "COFORGE", "PERSISTENT", "MPHASIS"],
    "INFY":       ["TCS", "COFORGE", "PERSISTENT", "MPHASIS"],
    "HAL":        ["BEL", "MAZDOCK"],
    "BEL":        ["HAL", "MAZDOCK"],
    "MAZDOCK":    ["HAL", "BEL"],
    "HDFCBANK":   ["ICICIBANK", "KOTAKBANK"],
    "ICICIBANK":  ["HDFCBANK", "KOTAKBANK"],
    "KOTAKBANK":  ["HDFCBANK", "ICICIBANK"],
    "LUPIN":      ["SUNPHARMA", "LAURUSLABS"],
    "SUNPHARMA":  ["LUPIN", "LAURUSLABS"],
}

# ── Config ────────────────────────────────────────────────────────────────────

START_DATE       = "2020-01-01"   # data start (includes warmup)
OOS_START        = "2022-01-01"   # out-of-sample start
END_DATE         = "2026-04-30"

INITIAL_CAPITAL  = 5_00_00_000.0  # ₹5 crore
TOP_N            = 3              # simultaneous positions
ALLOC_PER_STOCK  = 0.33           # 33% each
ENTRY_THRESHOLD  = 0.55           # minimum daily momentum score
REBAL_DAYS       = 63             # quarterly rebalance (~63 trading days)
OOS_WARMUP_BARS  = 220            # require 220+ bars before first entry
SLIPPAGE_PCT     = 0.001          # 0.10% per side
BROKERAGE_FLAT   = 20.0           # ₹20 per order
STT_PCT          = 0.001          # sell-side STT

# Exit parameters
ATR_PERIOD       = 14
ATR_TRAIL_MULT   = 2.5            # trailing stop: 2.5x ATR
ATR_TIGHTEN_MULT = 1.0            # tighten to 1.0x ATR after 2.5x gain
SMART_TIME_STOP  = 10             # bars: exit if still negative after N bars


@dataclass
class _MockConfig:
    """Minimal config proxy for MarketRegimeDetector."""
    cache_ttl: int           = 300
    sma_short_window: int    = 20
    sma_long_window: int     = 50
    sma_trend_window: int    = 200
    volatility_short_window: int = 20
    volatility_long_window: int  = 60
    momentum_short_window: int   = 10
    momentum_long_window: int    = 30
    volume_window: int           = 20
    min_data_rows: int           = 60
    regime_lookback: int         = 400


# ── Regime / FII helpers ──────────────────────────────────────────────────────

def _build_regime_detector() -> MarketRegimeDetector:
    return MarketRegimeDetector(config=_MockConfig(), db_manager=None)


def _load_fii_series(start: str, end: str) -> pd.Series:
    """Load FII net flow series. Returns empty Series on failure (gate skipped)."""
    try:
        provider = FIIDataProvider()
        series = provider.get_fii_flow(start_date=start, end_date=end)
        return series if series is not None else pd.Series(dtype=float, name="fii_net")
    except Exception as e:
        logger.warning(f"FII data load failed — FII gate will be skipped: {e}")
        return pd.Series(dtype=float, name="fii_net")


def _regime_blocks_entry(detector: MarketRegimeDetector, nifty_slice: pd.DataFrame) -> bool:
    """
    Returns True if regime gate should block a new long entry.
    Condition: trend_regime == BEARISH AND regime_confidence > 0.7.
    Safe: returns False (allow entry) if detection fails.
    """
    try:
        if nifty_slice is None or len(nifty_slice) < 220:
            return False
        regime = detector.detect_market_regime("NIFTY50", nifty_slice)
        return (
            regime.get("trend_regime") == "BEARISH"
            and regime.get("regime_confidence", 0.0) > 0.7
        )
    except Exception as e:
        logger.debug(f"Regime gate error (allowing entry): {e}")
        return False


def _fii_blocks_entry(fii_series: pd.Series, date: pd.Timestamp) -> bool:
    """
    Returns True if FII gate should block a new long entry.
    Condition: 5-day sum of FII net < -₹5,000cr.
    Safe: returns False if data unavailable.
    """
    try:
        if fii_series.empty:
            return False
        fii_before_date = fii_series.loc[fii_series.index <= date]
        if len(fii_before_date) < 1:
            return False
        fii_5d = float(fii_before_date.tail(5).sum())
        return fii_5d < -5_000.0
    except Exception as e:
        logger.debug(f"FII gate error (allowing entry): {e}")
        return False


# ── Position simulator ────────────────────────────────────────────────────────

def simulate_position(
    prices: pd.DataFrame,
    entry_bar: int,
    capital: float,
) -> Tuple[float, Dict]:
    """
    Simulate one stock position from bar `entry_bar` forward.
    Uses ATR-trailing stop + smart time stop.
    """
    close  = prices["close"]
    high   = prices["high"]
    low    = prices["low"]
    opens  = prices["open"]

    tr   = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr  = tr.rolling(ATR_PERIOD, min_periods=5).mean()

    trades = wins = 0
    trade_cap    = capital
    daily_pnl: Dict[int, float] = {}

    pos = 0
    entry_px = trail_stop = peak_close = 0.0
    bars_in_trade = 0

    for i in range(entry_bar, len(prices)):
        cc     = float(close.iloc[i])
        cc_prev= float(close.iloc[i - 1]) if i > 0 else cc
        atr_i  = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else cc * 0.02

        if pos > 0:
            bars_in_trade += 1
            daily_pnl[i]   = (cc - cc_prev) / cc_prev
            peak_close     = max(peak_close, cc)

            if cc > entry_px + atr_i:
                trail_stop = max(trail_stop, peak_close - ATR_TRAIL_MULT * atr_i)

            # Tighten after 2.5x ATR gain
            if cc > entry_px + ATR_TIGHTEN_MULT * atr_i * 2.5:
                trail_stop = max(trail_stop, peak_close - ATR_TIGHTEN_MULT * atr_i)

            hit_sl   = cc < trail_stop or float(low.iloc[i]) < trail_stop
            hit_time = bars_in_trade >= SMART_TIME_STOP and cc < entry_px + 0.5 * atr_i
            exit_now = hit_sl or hit_time or (i == len(prices) - 1)

            if exit_now:
                if hit_sl:
                    exit_px = min(trail_stop, float(opens.iloc[i]))
                else:
                    exit_px = cc * (1 - SLIPPAGE_PCT)

                cost    = BROKERAGE_FLAT + exit_px * pos * (STT_PCT + SLIPPAGE_PCT)
                net     = (exit_px - entry_px) * pos - cost
                trade_cap += net
                wins      += int(net > 0)
                trades    += 1
                pos = 0
        else:
            daily_pnl[i] = 0.0

    pnl_series = pd.Series(daily_pnl)
    return trade_cap, {
        "return_pct": (trade_cap / capital - 1) * 100.0,
        "win_rate":   wins / trades if trades else None,
        "trades":     trades,
        "daily_pnl":  pnl_series,
    }


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    universe: List[str] = UNIVERSE,
    start_date: str = START_DATE,
    oos_start: str = OOS_START,
    end_date: str = END_DATE,
    top_n: int = TOP_N,
    output_path: str = "reports/momentum_portfolio_results.json",
) -> Dict:

    logger.info(f"Loading NIFTY50 index data {start_date}→{end_date}...")
    nifty_raw = fetch_nifty50_index(start_date, end_date)

    logger.info("Loading FII flow data...")
    fii_series = _load_fii_series(start_date, end_date)

    logger.info(f"Loading OHLCV for {len(universe)} stocks...")
    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in universe:
        df = fetch_equity_ohlcv(ticker, start_date, end_date)
        if df is not None and len(df) >= OOS_WARMUP_BARS:
            all_data[ticker] = df
        else:
            logger.debug(f"[{ticker}] skipped — insufficient data")

    if len(all_data) < top_n:
        logger.error(f"Only {len(all_data)} tickers loaded — need {top_n}")
        return {"status": "failed", "reason": "insufficient data"}

    logger.info(f"Universe loaded: {len(all_data)} stocks")

    # Build sector peer close-series lookup (pre-sliced to latest bar for efficiency)
    def _peers_at(ticker_base: str, until: pd.Timestamp) -> List[pd.Series]:
        peer_names = SECTOR_PEERS.get(ticker_base, [])
        peers = []
        for p in peer_names:
            p_ns = f"{p}.NS"
            if p_ns in all_data:
                peer_close = all_data[p_ns]["close"].loc[:until]
                if len(peer_close) >= 20:
                    peers.append(peer_close)
        return peers

    regime_detector = _build_regime_detector()
    ms = MomentumSignal()

    oos_start_ts = pd.Timestamp(oos_start)

    # Get reference index for OOS trading dates
    ref_df   = next(iter(all_data.values()))
    oos_idx  = ref_df.index[ref_df.index >= oos_start_ts]

    if len(oos_idx) < REBAL_DAYS:
        logger.error("Too few OOS bars to run even one rebalance")
        return {"status": "failed", "reason": "insufficient OOS data"}

    # Rebalance dates: first bar of each quarter-equivalent (~63 days)
    rebal_dates = [oos_idx[i] for i in range(0, len(oos_idx), REBAL_DAYS)]
    logger.info(f"OOS: {oos_start}→{end_date} | {len(rebal_dates)} rebalances")

    equity_curve  = [INITIAL_CAPITAL]
    current_equity = INITIAL_CAPITAL
    all_trades: List[Dict] = []
    fold_results: List[Dict] = []

    for rebal_date in rebal_dates:
        # Build NIFTY slice for regime detector
        nifty_slice_df: Optional[pd.DataFrame] = None
        if nifty_raw is not None:
            nifty_before = nifty_raw[nifty_raw.index <= rebal_date]
            if len(nifty_before) >= 5:
                # Regime detector needs open/high/low/close/volume columns
                if isinstance(nifty_before, pd.Series):
                    nifty_slice_df = pd.DataFrame({
                        "close": nifty_before,
                        "open":  nifty_before,
                        "high":  nifty_before,
                        "low":   nifty_before,
                        "volume": 1.0,
                        "returns": nifty_before.pct_change(),
                        "volatility_20": nifty_before.pct_change().rolling(20).std(),
                        "sma_20": nifty_before.rolling(20).mean(),
                        "sma_50": nifty_before.rolling(50).mean(),
                    })
                else:
                    nifty_slice_df = nifty_before.copy()

        # ── Gate 1: Regime ───────────────────────────────────────────────
        if _regime_blocks_entry(regime_detector, nifty_slice_df):
            logger.info(f"  [{rebal_date.date()}] ❌ REGIME GATE: confirmed bearish — skipping rebalance")
            fold_results.append({
                "date": str(rebal_date.date()),
                "gate_blocked": "regime",
                "selected": [],
                "portfolio_ret": 0.0,
            })
            continue

        # ── Gate 2: FII ──────────────────────────────────────────────────
        if _fii_blocks_entry(fii_series, rebal_date):
            logger.info(f"  [{rebal_date.date()}] ❌ FII GATE: heavy selling — skipping rebalance")
            fold_results.append({
                "date": str(rebal_date.date()),
                "gate_blocked": "fii",
                "selected": [],
                "portfolio_ret": 0.0,
            })
            continue

        # ── Score stocks ─────────────────────────────────────────────────
        fii_5d_val = 0.0
        if not fii_series.empty:
            fii_before = fii_series.loc[fii_series.index <= rebal_date]
            if len(fii_before) >= 1:
                fii_5d_val = float(fii_before.tail(5).sum())

        nifty_before_series = (
            nifty_raw[nifty_raw.index <= rebal_date]
            if nifty_raw is not None else None
        )

        scores: Dict[str, float] = {}
        for ticker, full_df in all_data.items():
            df_slice = full_df.loc[full_df.index <= rebal_date]
            if len(df_slice) < OOS_WARMUP_BARS:
                continue

            # Gate 3: daily momentum score
            ticker_base = ticker.replace(".NS", "")
            peers = _peers_at(ticker_base, rebal_date)
            daily_score = ms.score(
                df_slice,
                nifty_before_series,
                fii_5d_val,
                peers,
            )
            if daily_score < ENTRY_THRESHOLD:
                continue

            # Gate 4: weekly alignment
            if not ms.weekly_aligned(df_slice):
                continue

            scores[ticker] = daily_score

        if not scores:
            logger.info(f"  [{rebal_date.date()}] No stocks passed all gates")
            fold_results.append({
                "date": str(rebal_date.date()),
                "gate_blocked": "signal",
                "selected": [],
                "portfolio_ret": 0.0,
            })
            continue

        # Select top-N
        ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ranked[:top_n]]
        score_str = " | ".join(f"{t}:{scores[t]:.3f}" for t in selected)
        logger.info(f"  [{rebal_date.date()}] Selected: {score_str}")

        # Determine test window: until next rebalance date or end
        rebal_pos = rebal_dates.index(rebal_date)
        next_rebal = rebal_dates[rebal_pos + 1] if rebal_pos + 1 < len(rebal_dates) else pd.Timestamp(end_date)

        # ── Simulate each selected stock ─────────────────────────────────
        per_cap = current_equity * ALLOC_PER_STOCK
        period_results: Dict[str, Dict] = {}

        for ticker in selected:
            full_df = all_data[ticker]
            test_df = full_df.loc[
                (full_df.index >= rebal_date) & (full_df.index < next_rebal)
            ].copy()
            if len(test_df) < 3:
                continue

            # Find entry bar (bar 0 of test window = rebalance day, enter at open bar 1)
            entry_bar = 1 if len(test_df) > 1 else 0

            final_cap, metrics = simulate_position(test_df, entry_bar, per_cap)
            period_results[ticker] = {**metrics, "capital": final_cap}

            # Record trade
            all_trades.append({
                "ticker":    ticker,
                "entry_date": str(rebal_date.date()),
                "exit_date":  str(next_rebal.date()),
                "return_pct": metrics["return_pct"],
                "win_rate":   metrics["win_rate"],
                "trades":     metrics["trades"],
            })

            wr_str = f"{metrics['win_rate']:.0%}" if metrics["win_rate"] is not None else "N/A"
            logger.info(f"    [{ticker}] {metrics['return_pct']:+.2f}% | trades={metrics['trades']} | WR={wr_str}")

        if not period_results:
            continue

        # Aggregate fold P&L
        total_pnl = sum(r["capital"] - per_cap for r in period_results.values())
        port_ret  = (total_pnl / current_equity) * 100.0
        current_equity = max(current_equity + total_pnl, 1.0)
        equity_curve.append(current_equity)

        # NIFTY reference for same period
        nifty_ret = 0.0
        if nifty_raw is not None:
            nf = nifty_raw.loc[
                (nifty_raw.index >= rebal_date) & (nifty_raw.index < next_rebal)
            ]
            if isinstance(nf, pd.DataFrame):
                nf = nf["close"] if "close" in nf.columns else nf.iloc[:, 0]
            if len(nf) >= 2:
                nifty_ret = (float(nf.iloc[-1]) / float(nf.iloc[0]) - 1) * 100.0

        fold_results.append({
            "date":          str(rebal_date.date()),
            "gate_blocked":  None,
            "selected":      selected,
            "portfolio_ret": round(port_ret, 4),
            "nifty_ret":     round(nifty_ret, 4),
            "alpha":         round(port_ret - nifty_ret, 4),
        })

    # ── Summary statistics ────────────────────────────────────────────────────
    total_return_pct = (current_equity / INITIAL_CAPITAL - 1) * 100.0
    n_years = (pd.Timestamp(end_date) - pd.Timestamp(oos_start)).days / 365.25
    cagr = ((current_equity / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100.0 if n_years > 0 else 0.0

    # MaxDrawdown
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    max_dd_pct = float(drawdown.min()) * 100.0

    # Win rate across all trades
    trade_wins = [t for t in all_trades if t.get("return_pct", 0) > 0]
    win_rate = len(trade_wins) / len(all_trades) if all_trades else 0.0

    results = {
        "summary": {
            "initial_capital": INITIAL_CAPITAL,
            "final_equity":    round(current_equity, 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct":        round(cagr, 2),
            "max_dd_pct":      round(max_dd_pct, 2),
            "win_rate":        round(win_rate, 4),
            "total_trades":    len(all_trades),
            "oos_start":       oos_start,
            "end_date":        end_date,
            "universe_size":   len(all_data),
        },
        "equity_curve": [round(v, 2) for v in equity_curve],
        "fold_results":  fold_results,
        "trades":        all_trades,
    }

    logger.info("\n" + "═" * 60)
    logger.info(f"  CAGR:      {cagr:+.1f}%")
    logger.info(f"  MaxDD:     {max_dd_pct:.1f}%")
    logger.info(f"  Win Rate:  {win_rate:.0%}")
    logger.info(f"  Trades:    {len(all_trades)}")
    logger.info("═" * 60)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written → {output_path}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARK5 Momentum Portfolio Backtest")
    parser.add_argument("--start",   default=START_DATE)
    parser.add_argument("--oos",     default=OOS_START)
    parser.add_argument("--end",     default=END_DATE)
    parser.add_argument("--top-n",   type=int, default=TOP_N)
    parser.add_argument("--output",  default="reports/momentum_portfolio_results.json")
    args = parser.parse_args()

    run_backtest(
        universe=UNIVERSE,
        start_date=args.start,
        oos_start=args.oos,
        end_date=args.end,
        top_n=args.top_n,
        output_path=args.output,
    )
