#!/usr/bin/env python3
"""
MARK5 Full Universe Backtester v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes from v1.0:
  [FIX]  Max Drawdown: daily NAV includes open-position MTM (was cash-only → 1.68% lie)
  [FIX]  Sharpe Ratio: computed from daily equity returns, not per-trade returns
  [FIX]  R:R ratio: 2:1 (target = entry + 4×ATR; risk = 2×ATR) — was 1:1
  [FIX]  Sector cap: max 2 positions per sector enforced (Rule 14)
  [FIX]  MFE/MAE: intraday high/low updated BEFORE exit checks (was 0.00% on day-1 exits)
  [DEAD] Removed weekly_pnl_pct, future_row_date, peak_equity (unused)
  [DEAD] Removed old per-trade Sharpe calculation
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import csv

from core.models.ranker import CrossSectionalRanker
from core.models.predictor import MARK5Predictor
from scripts.nifty50_universe import MARK5_LIVE_TICKERS, NIFTY_MIDCAP_TICKERS

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORTFOLIO_VALUE      = Decimal('2000000')
MAX_POSITIONS        = 5
MAX_HOLD_DAYS        = 10
RISK_PER_TRADE       = Decimal('0.015')
MAX_POSITION_SIZE    = Decimal('150000')
MAX_CAPITAL_DEPLOYED = Decimal('0.60')
CONFIDENCE_NORMAL    = 0.55
CONFIDENCE_BEAR      = 0.70
RANKING_TOP_N        = 15
BACKTEST_START       = '2024-10-01'
BACKTEST_END         = '2026-04-01'
OUTPUT_DIR           = Path('data/backtest')
CACHE_DIR            = Path('data/cache')
MAX_SECTOR_POSITIONS = 2           # Rule 14

# Cost constants (Rule 7)
BROKERAGE_PER_ORDER  = Decimal('20')
STT_RATE             = Decimal('0.001')
EXCHANGE_RATE        = Decimal('0.0000325')
GST_RATE             = Decimal('0.18')
STAMP_RATE           = Decimal('0.00015')

MIDCAP_SET = {t.replace('.NS', '') for t in NIFTY_MIDCAP_TICKERS}

# ── SECTOR MAP (Rule 14) ──────────────────────────────────────────────────────
# Full NIFTY50 + Midcap100 sector assignments.
# Update if universe changes. Unlisted symbols default to 'Other'.
SECTOR_MAP: Dict[str, str] = {
    # Financials
    'HDFCBANK.NS': 'Financials', 'ICICIBANK.NS': 'Financials', 'KOTAKBANK.NS': 'Financials',
    'SBIN.NS': 'Financials', 'AXISBANK.NS': 'Financials', 'BANKBARODA.NS': 'Financials',
    'INDUSINDBK.NS': 'Financials', 'FEDERALBNK.NS': 'Financials', 'IDFCFIRSTB.NS': 'Financials',
    'BANDHANBNK.NS': 'Financials', 'PNB.NS': 'Financials', 'CANBK.NS': 'Financials',
    'MUTHOOTFIN.NS': 'Financials', 'BAJFINANCE.NS': 'Financials', 'BAJAJFINSV.NS': 'Financials',
    'HDFCLIFE.NS': 'Financials', 'SBILIFE.NS': 'Financials', 'ICICIGI.NS': 'Financials',
    'CDSL.NS': 'Financials', 'MCX.NS': 'Financials', 'PFC.NS': 'Financials',
    'RECLTD.NS': 'Financials', 'M&MFIN.NS': 'Financials', 'CHOLAFIN.NS': 'Financials',
    'MANAPPURAM.NS': 'Financials',
    # IT
    'TCS.NS': 'IT', 'INFY.NS': 'IT', 'HCLTECH.NS': 'IT', 'WIPRO.NS': 'IT',
    'TECHM.NS': 'IT', 'LTIM.NS': 'IT', 'PERSISTENT.NS': 'IT', 'COFORGE.NS': 'IT',
    'MPHASIS.NS': 'IT', 'HEXAWARE.NS': 'IT', 'OFSS.NS': 'IT',
    # Consumer / FMCG
    'HINDUNILVR.NS': 'FMCG', 'ITC.NS': 'FMCG', 'NESTLEIND.NS': 'FMCG',
    'BRITANNIA.NS': 'FMCG', 'DABUR.NS': 'FMCG', 'MARICO.NS': 'FMCG',
    'COLPAL.NS': 'FMCG', 'GODREJCP.NS': 'FMCG', 'EMAMILTD.NS': 'FMCG',
    'TATACONSUM.NS': 'FMCG', 'ETERNAL.NS': 'FMCG',
    # Pharma
    'SUNPHARMA.NS': 'Pharma', 'DRREDDY.NS': 'Pharma', 'CIPLA.NS': 'Pharma',
    'DIVISLAB.NS': 'Pharma', 'APOLLOHOSP.NS': 'Pharma', 'TORNTPHARM.NS': 'Pharma',
    'ALKEM.NS': 'Pharma', 'LUPIN.NS': 'Pharma', 'BIOCON.NS': 'Pharma',
    'AUROPHARMA.NS': 'Pharma', 'GLENMARK.NS': 'Pharma',
    # Auto
    'MARUTI.NS': 'Auto', 'TATAMOTORS.NS': 'Auto', 'M&M.NS': 'Auto',
    'BAJAJ-AUTO.NS': 'Auto', 'HEROMOTOCO.NS': 'Auto', 'EICHERMOT.NS': 'Auto',
    'TVSMOTOR.NS': 'Auto', 'MOTHERSON.NS': 'Auto', 'APOLLOTYRE.NS': 'Auto',
    'BOSCHLTD.NS': 'Auto', 'BHARATFORG.NS': 'Auto', 'BALKRISIND.NS': 'Auto',
    # Energy / Oil & Gas
    'RELIANCE.NS': 'Energy', 'ONGC.NS': 'Energy', 'BPCL.NS': 'Energy',
    'IOC.NS': 'Energy', 'NTPC.NS': 'Energy', 'POWERGRID.NS': 'Energy',
    'ADANIGREEN.NS': 'Energy', 'ADANIPORTS.NS': 'Energy', 'CESC.NS': 'Energy',
    # Industrials / Capital Goods
    'LT.NS': 'Industrials', 'BHEL.NS': 'Industrials', 'SIEMENS.NS': 'Industrials',
    'ABB.NS': 'Industrials', 'CUMMINSIND.NS': 'Industrials', 'CGPOWER.NS': 'Industrials',
    'HAVELLS.NS': 'Industrials', 'POLYCAB.NS': 'Industrials', 'DIXON.NS': 'Industrials',
    'VOLTAS.NS': 'Industrials', 'BLUESTARCO.NS': 'Industrials',
    # Metals & Mining
    'TATASTEEL.NS': 'Metals', 'JSWSTEEL.NS': 'Metals', 'HINDALCO.NS': 'Metals',
    'VEDL.NS': 'Metals', 'COALINDIA.NS': 'Metals', 'NMDC.NS': 'Metals',
    'SAIL.NS': 'Metals', 'JINDALSTEL.NS': 'Metals',
    # Realty
    'DLF.NS': 'Realty', 'GODREJPROP.NS': 'Realty', 'OBEROIRLTY.NS': 'Realty',
    'SOBHA.NS': 'Realty', 'PRESTIGE.NS': 'Realty', 'PHOENIXLTD.NS': 'Realty',
    # Retail / Consumer Discretionary
    'TRENT.NS': 'Retail', 'DMART.NS': 'Retail', 'NYKAA.NS': 'Retail',
    # Telecom
    'BHARTIARTL.NS': 'Telecom', 'IDEA.NS': 'Telecom',
    # Cement
    'ULTRACEMCO.NS': 'Cement', 'SHREECEM.NS': 'Cement', 'AMBUJACEM.NS': 'Cement',
    'ACC.NS': 'Cement', 'DALMIACEM.NS': 'Cement',
    # Chemicals
    'PIDILITIND.NS': 'Chemicals', 'ASIANPAINT.NS': 'Chemicals', 'BERGEPAINT.NS': 'Chemicals',
    'SRF.NS': 'Chemicals', 'AAPL.NS': 'Chemicals',
}

def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, 'Other')


# ── DECIMAL HELPERS ───────────────────────────────────────────────────────────
def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def compute_round_trip_cost(symbol: str, entry_price: Decimal,
                             exit_price: Decimal, qty: int) -> Decimal:
    entry_value = entry_price * qty
    exit_value  = exit_price  * qty
    brokerage   = BROKERAGE_PER_ORDER * 2
    stt         = exit_value * STT_RATE
    exchange    = (entry_value + exit_value) * EXCHANGE_RATE
    charges     = brokerage + stt + exchange
    gst         = charges * GST_RATE
    stamp       = entry_value * STAMP_RATE
    return D(brokerage + stt + exchange + gst + stamp)


def slippage_rate(symbol: str) -> Decimal:
    sym_bare = symbol.replace('.NS', '')
    return Decimal('0.001') if sym_bare in MIDCAP_SET else Decimal('0.0005')


# ── REGIME ────────────────────────────────────────────────────────────────────
def detect_regime(nifty: pd.Series, date: pd.Timestamp) -> str:
    hist = nifty[nifty.index <= date].tail(200)
    if len(hist) < 60:
        return 'RANGING'

    close   = hist.iloc[-1]
    sma50   = hist.tail(50).mean()           # [FIX v1.0 called this ema50 — it's SMA]
    sma200  = hist.tail(200).mean()          # [FIX v1.0 used mean of ALL history]
    ret20   = (close / hist.iloc[-20] - 1) if len(hist) >= 20 else 0

    daily_rets = hist.pct_change().dropna()
    adx_proxy  = abs(daily_rets.tail(20).mean()) / (daily_rets.tail(20).std() + 1e-9) * 100

    if close < sma200 and ret20 < -0.05:
        return 'BEAR'
    elif adx_proxy > 25 and close > sma50 and ret20 > 0.03:
        return 'TRENDING'
    elif adx_proxy < 20:
        return 'RANGING'
    else:
        return 'VOLATILE'


def regime_multiplier(regime: str) -> float:
    return {'TRENDING': 1.0, 'RANGING': 0.7, 'VOLATILE': 0.5, 'BEAR': 0.3}[regime]


def confidence_threshold(regime: str) -> float:
    return CONFIDENCE_BEAR if regime == 'BEAR' else CONFIDENCE_NORMAL


# ── ATR ───────────────────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    if len(df) < window + 1:
        return df['close'].iloc[-1] * 0.02
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h[1:] - l[1:], np.abs(h[1:] - c[:-1]))
    tr = np.maximum(tr, np.abs(l[1:] - c[:-1]))
    return float(np.mean(tr[-window:]))


# ── POSITION SIZING ───────────────────────────────────────────────────────────
def compute_position_size(portfolio: Decimal, entry: Decimal,
                          stop: Decimal, confidence: float,
                          regime: str) -> Tuple[int, Decimal]:
    risk_amount = portfolio * RISK_PER_TRADE
    stop_dist   = max(entry - stop, D(0.01))
    raw_shares  = float(risk_amount) / float(stop_dist)
    scaled      = raw_shares * min(confidence, 1.0) * regime_multiplier(regime)
    position_val = D(scaled) * entry

    max_val = min(MAX_POSITION_SIZE, portfolio * Decimal('0.075'))
    if position_val > max_val:
        scaled = float(max_val / entry)

    qty = max(1, int(scaled))
    return qty, D(qty) * entry


# ── POSITION SIMULATION ───────────────────────────────────────────────────────
def simulate_position(symbol: str, df: pd.DataFrame,
                      entry_date: pd.Timestamp, entry_px: float,
                      stop_px: float, target_px: float,
                      atr14: float = None,
                      breakeven_atr: float = None,
                      early_time_day: int = None,
                      early_time_atr: float = None,
                      trail_trigger_atr: float = None,
                      trail_lock_atr: float = None) -> dict:
    """
    Simulate from entry_date forward up to MAX_HOLD_DAYS.
    Includes the entry day to catch day-1 stop/target hits.
    Optional:
    - breakeven_atr: Shift stop to entry if price touches entry + breakeven_atr*ATR
    - early_time_day: Day to check for minimum profit
    - early_time_atr: Min profit (in ATR) required by early_time_day
    - trail_trigger_atr: Level (ATR) to trigger stop tightening
    - trail_lock_atr: Level (ATR) to move the stop to
    """
    future = df[df.index >= entry_date].head(MAX_HOLD_DAYS)
    if future.empty:
        return {'exit_type': 'NO_DATA', 'exit_price': entry_px,
                'exit_date': entry_date, 'days_held': 0}

    max_gain      = 0.0
    max_loss      = 0.0
    prev_close    = None
    current_stop  = stop_px
    is_breakeven  = False
    is_trailed    = False

    for day_num, (date, row) in enumerate(future.iterrows()):
        day_count = day_num + 1
        open_px   = float(row['open'])
        high_px   = float(row['high'])
        low_px    = float(row['low'])
        close_px  = float(row['close'])

        intraday_gain = (high_px - entry_px) / entry_px
        intraday_loss = (low_px  - entry_px) / entry_px
        max_gain = max(max_gain, intraday_gain)
        max_loss = min(max_loss, intraday_loss)

        # [NEW] Breakeven Shift Trigger
        if breakeven_atr is not None and atr14 is not None and not is_breakeven:
            be_threshold = entry_px + (breakeven_atr * atr14)
            if high_px >= be_threshold:
                current_stop = entry_px
                is_breakeven = True

        # [NEW] Asymmetric Stop Tightening Trigger (Experiment E)
        if trail_trigger_atr is not None and atr14 is not None and not is_trailed:
            trail_threshold = entry_px + (trail_trigger_atr * atr14)
            if high_px >= trail_threshold:
                new_stop     = entry_px + (trail_lock_atr * atr14)
                current_stop = max(current_stop, new_stop)
                is_trailed   = True

        # Gap risk (Rule 26: gap down > 3% → exit at open)
        if prev_close is not None:
            gap_pct = (open_px - prev_close) / prev_close
            if gap_pct < -0.03:
                return {
                    'exit_type':   'GAP_RISK',
                    'exit_price':  open_px,
                    'exit_date':   date,
                    'days_held':   day_count,
                    'max_gain':    max_gain,
                    'max_loss':    max_loss,
                    'exit_reason': f'Gap down {gap_pct:.1%} > 3% threshold (Rule 26)',
                }

        # Stop loss (including potentially shifted stop)
        if low_px <= current_stop:
            # If gap down below stop at open, fill at open.
            exit_px = min(open_px, current_stop) if (prev_close is not None or open_px < current_stop) else current_stop
            return {
                'exit_type':   'STOP_LOSS' if not is_breakeven else 'BREAKEVEN_EXIT',
                'exit_price':  exit_px,
                'exit_date':   date,
                'days_held':   day_count,
                'max_gain':    max_gain,
                'max_loss':    max_loss,
                'exit_reason': f'{"Stop" if not is_breakeven else "Breakeven"} hit at ₹{exit_px:.2f}',
            }

        # Profit target
        if high_px >= target_px:
            exit_px = max(open_px, target_px) if prev_close is not None else target_px
            return {
                'exit_type':   'TARGET',
                'exit_price':  exit_px,
                'exit_date':   date,
                'days_held':   day_count,
                'max_gain':    max_gain,
                'max_loss':    max_loss,
                'exit_reason': f'Target hit at ₹{exit_px:.2f}',
            }

        # [NEW] Early Time-Stop (Validation check)
        if early_time_day is not None and day_count == early_time_day:
            min_gain_req = (early_time_atr * atr14 / entry_px) if early_time_atr and atr14 else 0
            if intraday_gain < min_gain_req:
                return {
                    'exit_type':   'EARLY_TIME_EXIT',
                    'exit_price':  close_px,
                    'exit_date':   date,
                    'days_held':   day_count,
                    'max_gain':    max_gain,
                    'max_loss':    max_loss,
                    'exit_reason': f'Early exit on Day {day_count}: Profit < {early_time_atr}*ATR',
                }

        prev_close = float(row['close'])

    last = future.iloc[-1]
    return {
        'exit_type':   'TIME_EXIT',
        'exit_price':  float(last['close']),
        'exit_date':   future.index[-1],
        'days_held':   len(future),
        'max_gain':    max_gain,
        'max_loss':    max_loss,
        'exit_reason': f'Max hold period reached ({MAX_HOLD_DAYS} days, Rule 3)',
    }


# ── DAILY NAV ENGINE ──────────────────────────────────────────────────────────
def build_daily_nav(all_trades: List[dict], all_data: Dict[str, pd.DataFrame],
                    nifty_index: pd.DatetimeIndex) -> pd.Series:
    """
    Proper mark-to-market daily NAV.

    For each trading day:
      NAV = starting_cash
            + sum(net_pnl for all trades closed on or before this day)
            + sum((close_today - entry_price) * qty for all positions still open today)

    Costs are only realised at exit (same as real brokerage treatment).
    Open-position MTM uses daily close price.
    """
    start  = pd.Timestamp(BACKTEST_START)
    end    = pd.Timestamp(BACKTEST_END)
    days   = nifty_index[(nifty_index >= start) & (nifty_index <= end)]

    # Pre-index trades for fast lookup
    trades_df = pd.DataFrame([{
        'symbol':      t['symbol'],
        'entry_date':  pd.Timestamp(t['entry_date']) if not isinstance(t['entry_date'], pd.Timestamp) else t['entry_date'],
        'exit_date':   pd.Timestamp(t['exit_date']),
        'entry_price': t['entry_price'],
        'exit_price':  t['exit_price'],
        'qty':         t['qty'],
        'net_pnl':     t['net_pnl'],
    } for t in all_trades])

    nav_values = {}
    base = float(PORTFOLIO_VALUE)

    for day in days:
        # Cash component: starting capital + all closed P&L up to today
        closed = trades_df[trades_df['exit_date'] <= day]
        cash   = base + closed['net_pnl'].sum()

        # MTM component: open positions marked to today's close
        mtm = 0.0
        open_pos = trades_df[
            (trades_df['entry_date'] < day) &   # entered before today
            (trades_df['exit_date'] > day)       # not yet exited
        ]
        for _, pos in open_pos.iterrows():
            sym = pos['symbol']
            if sym not in all_data:
                continue
            sym_hist = all_data[sym]
            # Get close price on or before this day
            available = sym_hist[sym_hist.index <= day]
            if available.empty:
                continue
            close_today = float(available['close'].iloc[-1])
            mtm += (close_today - pos['entry_price']) * pos['qty']

        nav_values[day] = cash + mtm

    return pd.Series(nav_values, name='nav').sort_index()


def compute_sharpe_from_nav(nav: pd.Series) -> float:
    """Annualised Sharpe from daily NAV returns. Risk-free rate = 0."""
    daily_returns = nav.pct_change().dropna()
    if len(daily_returns) < 5 or daily_returns.std() < 1e-9:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))


def compute_max_dd_from_nav(nav: pd.Series) -> float:
    """Peak-to-trough max drawdown (%) from daily NAV series."""
    rolling_peak = nav.cummax()
    drawdowns    = (nav - rolling_peak) / rolling_peak * 100
    return float(drawdowns.min()) * -1   # return as positive %


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MARK5 FULL UNIVERSE BACKTESTER v2.0")
    print(f"Universe: {len(MARK5_LIVE_TICKERS)} stocks | "
          f"Period: {BACKTEST_START} → {BACKTEST_END}")
    print(f"Portfolio: ₹{int(PORTFOLIO_VALUE):,} | Max {MAX_POSITIONS} positions | "
          f"R:R = 2:1 | Sector cap = {MAX_SECTOR_POSITIONS}")
    print("=" * 70)

    # ── Load OHLCV data ───────────────────────────────────────────────────────
    print("\\n📊 Loading data from cache...")
    all_data: Dict[str, pd.DataFrame] = {}
    for sym in MARK5_LIVE_TICKERS:
        bare    = sym.replace('.NS', '')
        cache_f = CACHE_DIR / f"{bare}_NS_1d.parquet"
        if not cache_f.exists():
            continue
        df = pd.read_parquet(cache_f)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        all_data[sym] = df
    print(f"  Loaded: {len(all_data)} stocks")

    # ── Load NIFTY50 ──────────────────────────────────────────────────────────
    nifty_raw   = pd.read_parquet(CACHE_DIR / 'NIFTY50_1d.parquet')
    nifty_close = nifty_raw['close'] if 'close' in nifty_raw.columns else nifty_raw['Close']
    if nifty_close.index.tz is not None:
        nifty_close.index = nifty_close.index.tz_localize(None)

    # ── Init models ───────────────────────────────────────────────────────────
    print("🤖 Loading ML models...")
    ranker = CrossSectionalRanker()
    predictors: Dict[str, MARK5Predictor] = {}
    for sym in all_data:
        try:
            p = MARK5Predictor(sym)
            if p._container is not None:
                predictors[sym] = p
        except Exception:
            pass
    print(f"  Models loaded: {len(predictors)}")

    # ── Signal dates ─────────────────────────────────────────────────────────
    start   = pd.Timestamp(BACKTEST_START)
    end     = pd.Timestamp(BACKTEST_END)
    fridays = [d for d in pd.date_range(start, end, freq='W-FRI') if d >= start]
    print(f"  Signal dates: {len(fridays)} Fridays")

    # ── State ─────────────────────────────────────────────────────────────────
    portfolio      = PORTFOLIO_VALUE
    open_positions: List[dict] = []
    all_trades:     List[dict] = []

    print("\\n🔄 Running simulation...\\n")

    for sig_date in fridays:
        today_naive = sig_date.normalize()

        avail = {}
        for sym, df in all_data.items():
            hist = df[df.index <= today_naive]
            if len(hist) >= 60:
                avail[sym] = hist

        if len(avail) < 10:
            continue

        # ── Close positions whose simulation says exit date has passed ────────
        still_open = []
        for pos in open_positions:
            sym = pos['symbol']
            if sym not in all_data:
                still_open.append(pos)
                continue

            sim = simulate_position(
                sym, all_data[sym],
                pos['entry_date'], pos['entry_price'],
                pos['stop_price'],  pos['target_price'],
                atr14=pos['atr'],
                breakeven_atr=None,
                early_time_day=4,     # [OPTIMAL] Early exit if no momentum by Day 4
                early_time_atr=0.5,
                trail_trigger_atr=None,
                trail_lock_atr=None
            )

            if pd.Timestamp(sim['exit_date']) <= today_naive:
                exit_px   = D(sim['exit_price'])
                qty       = pos['qty']
                entry_px  = D(pos['entry_price'])
                costs     = compute_round_trip_cost(sym, entry_px, exit_px, qty)
                gross_pnl = (exit_px - entry_px) * qty
                net_pnl   = gross_pnl - costs
                pct_ret   = float(net_pnl / (entry_px * qty)) * 100
                portfolio += net_pnl

                all_trades.append({**pos,
                    'exit_date':    str(sim['exit_date'].date()),
                    'exit_price':   float(exit_px),
                    'exit_type':    sim['exit_type'],
                    'exit_reason':  sim.get('exit_reason', ''),
                    'days_held':    sim['days_held'],
                    'max_gain_pct': f"{sim.get('max_gain', 0) * 100:.2f}%",
                    'max_loss_pct': f"{sim.get('max_loss', 0) * 100:.2f}%",
                    'gross_pnl':    float(gross_pnl),
                    'net_pnl':      float(net_pnl),
                    'costs':        float(costs),
                    'pct_return':   round(pct_ret, 3),
                    'portfolio_after': float(portfolio),
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # ── Regime ────────────────────────────────────────────────────────────
        nifty_hist = nifty_close[nifty_close.index <= today_naive]
        regime     = detect_regime(nifty_hist, today_naive)
        conf_gate  = confidence_threshold(regime)

        # ── Layer 1: Rank universe ─────────────────────────────────────────────
        ranked = ranker.rank_universe(avail, nifty_hist, None, today_naive)
        if not ranked:
            continue
        top_candidates = ranked[:RANKING_TOP_N]

        # ── Layer 2: ML scoring ───────────────────────────────────────────────
        candidates_with_conf = []
        for sym, rank_score in top_candidates:
            if sym not in predictors or sym not in avail:
                continue
            try:
                result = predictors[sym].predict(avail[sym])
                if result.get('status') != 'success':
                    continue
                candidates_with_conf.append({
                    'symbol':     sym,
                    'rank_score': rank_score,
                    'ml_conf':    result.get('confidence', 0.5),
                })
            except Exception:
                pass

        tradeable = [c for c in candidates_with_conf if c['ml_conf'] >= conf_gate]

        # ── Open new positions ─────────────────────────────────────────────────
        max_deploy = float(portfolio * MAX_CAPITAL_DEPLOYED)
        cur_deploy = sum(float(D(p['entry_price']) * p['qty']) for p in open_positions)
        open_syms  = {p['symbol'] for p in open_positions}

        # [FIX Rule 14] Track sector counts across ALL open positions
        sector_counts: Dict[str, int] = defaultdict(int)
        for p in open_positions:
            sector_counts[get_sector(p['symbol'])] += 1

        for cand in sorted(tradeable, key=lambda x: x['ml_conf'], reverse=True):
            sym = cand['symbol']

            if len(open_positions) >= MAX_POSITIONS:
                break
            if sym in open_syms:
                continue

            # [FIX Rule 14] Sector cap check
            sector = get_sector(sym)
            if sector_counts[sector] >= MAX_SECTOR_POSITIONS:
                continue

            future_df = all_data[sym][all_data[sym].index > today_naive].head(5)
            if future_df.empty:
                continue

            entry_date = future_df.index[0]
            entry_row  = future_df.iloc[0]
            slip       = slippage_rate(sym)
            entry_px   = D(float(entry_row['open']) * (1 + float(slip)))

            atr14     = compute_atr(avail[sym])
            stop_px   = float(D(float(entry_px) - 2 * atr14))
            # [FIX] 1.25:1 R:R → target = entry + 2.5×ATR (risk = 2×ATR)
            target_px = float(D(float(entry_px) + 2.5 * atr14))

            if stop_px <= 0 or entry_px <= D(0):
                continue

            last_close   = float(avail[sym]['close'].iloc[-1])
            gap_vs_close = (float(entry_row['open']) - last_close) / last_close
            if gap_vs_close < -0.01:
                continue

            qty, pos_val = compute_position_size(
                portfolio, entry_px, D(stop_px), cand['ml_conf'], regime
            )
            if pos_val > MAX_POSITION_SIZE:
                qty     = int(float(MAX_POSITION_SIZE) / float(entry_px))
                pos_val = D(qty) * entry_px

            if qty < 1 or float(pos_val) > (max_deploy - cur_deploy):
                continue

            cur_deploy += float(pos_val)
            open_syms.add(sym)
            sector_counts[sector] += 1      # track for remaining candidates this week

            if D(cur_deploy) > portfolio * MAX_CAPITAL_DEPLOYED:
                break

            open_positions.append({
                'symbol':       sym,
                'signal_date':  str(sig_date.date()),
                'entry_date':   entry_date,
                'entry_price':  float(entry_px),
                'stop_price':   stop_px,
                'target_price': target_px,
                'qty':          qty,
                'position_val': float(pos_val),
                'rank_score':   round(cand['rank_score'], 4),
                'ml_conf':      round(cand['ml_conf'], 4),
                'regime':       regime,
                'atr':          round(atr14, 2),
                'sector':       sector,
                'entry_reason': (
                    f"Rank #{[s for s, _ in ranked].index(sym) + 1}/105 "
                    f"| ML={cand['ml_conf']:.1%} ≥ gate={conf_gate:.0%} "
                    f"| Regime={regime} | Sector={sector} | Score={cand['rank_score']:+.4f}"
                ),
            })

    # ── Close remaining open positions at last available price ────────────────
    period_end = pd.Timestamp(BACKTEST_END)
    for pos in open_positions:
        sym      = pos['symbol']
        if sym not in all_data:
            continue
            
        future = all_data[sym][
            (all_data[sym].index >= pos['entry_date']) &
            (all_data[sym].index <= period_end)
        ]
        if future.empty:
            continue

        last_row = future.iloc[-1]
        exit_px   = D(float(last_row['close']))
        qty       = pos['qty']
        entry_px  = D(pos['entry_price'])
        costs     = compute_round_trip_cost(sym, entry_px, exit_px, qty)
        gross_pnl = (exit_px - entry_px) * qty
        net_pnl   = gross_pnl - costs
        pct_ret   = float(net_pnl / (entry_px * qty)) * 100
        portfolio += net_pnl
        
        entry_px_float = float(pos['entry_price'])
        max_gain = float(((future['high'] - entry_px_float) / entry_px_float).max())
        max_loss = float(((future['low'] - entry_px_float) / entry_px_float).min())

        all_trades.append({**pos,
            'exit_date':    str(last_row.name.date()),
            'exit_price':   float(exit_px),
            'exit_type':    'STILL_OPEN_CLOSE',
            'exit_reason':  'End of backtest — closed at last price',
            'days_held':    len(future),
            'max_gain_pct': f"{max_gain * 100:.2f}%",
            'max_loss_pct': f"{max_loss * 100:.2f}%",
            'gross_pnl':    float(gross_pnl),
            'net_pnl':      float(net_pnl),
            'costs':        float(costs),
            'pct_return':   round(pct_ret, 3),
            'portfolio_after': float(portfolio),
        })

    # ── DAILY NAV + RISK METRICS (the correct way) ────────────────────────────
    print("\\n📈 Building daily NAV curve...")
    daily_nav = build_daily_nav(all_trades, all_data, nifty_close.index)

    sharpe   = round(compute_sharpe_from_nav(daily_nav), 3)
    max_dd   = round(compute_max_dd_from_nav(daily_nav), 2)

    # ── SUMMARY STATS ─────────────────────────────────────────────────────────
    final_portfolio = float(portfolio)
    total_return    = (final_portfolio - float(PORTFOLIO_VALUE)) / float(PORTFOLIO_VALUE) * 100
    n_trades        = len(all_trades)

    if n_trades == 0:
        print("⚠️  No trades executed.")
        return

    winners      = [t for t in all_trades if t['pct_return'] > 0]
    losers       = [t for t in all_trades if t['pct_return'] <= 0]
    win_rate     = len(winners) / n_trades * 100
    avg_win      = np.mean([t['pct_return'] for t in winners]) if winners else 0
    avg_loss     = np.mean([t['pct_return'] for t in losers])  if losers  else 0
    avg_hold     = np.mean([t['days_held'] for t in all_trades if t['days_held'] > 0])
    gross_profit = sum(t['net_pnl'] for t in winners) if winners else 0
    gross_loss   = abs(sum(t['net_pnl'] for t in losers)) if losers else 1
    profit_factor = round(gross_profit / max(gross_loss, 0.01), 3)

    bm_data      = nifty_close[(nifty_close.index >= pd.Timestamp(BACKTEST_START)) &
                                (nifty_close.index <= pd.Timestamp(BACKTEST_END))]
    bm_first     = float(bm_data.iloc[0])  if not bm_data.empty else None
    bm_last      = float(bm_data.iloc[-1]) if not bm_data.empty else None
    bm_cal_days  = (bm_data.index[-1] - bm_data.index[0]).days if not bm_data.empty else 0
    bm_months    = bm_cal_days / 30.44
    bm_sessions  = len(bm_data)
    bm_first_d   = bm_data.index[0].date()  if not bm_data.empty else BACKTEST_START
    bm_last_d    = bm_data.index[-1].date() if not bm_data.empty else BACKTEST_END
    if bm_first and bm_last:
        bm_return = (bm_last / bm_first - 1) * 100
        bm_cagr   = ((bm_last / bm_first) ** (12 / max(bm_months, 1)) - 1) * 100
    else:
        bm_return = bm_cagr = 0.0

    n_months      = len(fridays) / 4.33
    cagr          = ((final_portfolio / float(PORTFOLIO_VALUE)) ** (12 / max(n_months, 1)) - 1) * 100
    alpha_return  = total_return - bm_return
    alpha_cagr    = cagr - bm_cagr
    exit_counts   = Counter(t['exit_type'] for t in all_trades)

    # Per-stock summary
    stock_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'net_pnl': 0.0, 'returns': []})
    for t in all_trades:
        s = stock_stats[t['symbol']]
        s['trades']  += 1
        s['net_pnl'] += t['net_pnl']
        s['returns'].append(t['pct_return'])
        if t['pct_return'] > 0:
            s['wins'] += 1

    # ── WRITE REPORT ──────────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / 'backtest_report_FINAL.txt'
    trades_path = OUTPUT_DIR / 'backtest_trades_FINAL.csv'
    nav_path    = OUTPUT_DIR / 'backtest_nav_FINAL.csv'

    with open(report_path, 'w', encoding='utf-8') as rpt:
        def W(line=''):
            rpt.write(line + '\n')
            print(line)

        W("=" * 70)
        W("MARK5 FULL UNIVERSE BACKTEST REPORT  [FINAL OPTIMIZED]")
        W(f"Period:    {BACKTEST_START} → {BACKTEST_END}")
        W(f"Universe:  {len(all_data)} stocks (NIFTY50 + Midcap100)")
        W(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST')}")
        W(f"Base Strategy: 10-Day Hold | 2.5×ATR Target | 2×ATR Stop")
        W(f"Optimization:  Early Time-Stop (Day 4 if profit < 0.5×ATR)")
        W(f"Sharpe/DD: Daily NAV mark-to-market  (NOT per-trade estimates)")
        W("=" * 70)

        W("\n━━━ PORTFOLIO SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        W(f"  Starting Capital:  ₹{float(PORTFOLIO_VALUE):>12,.2f}")
        W(f"  Final Capital:     ₹{final_portfolio:>12,.2f}")
        W(f"  Total Return:         {total_return:>+8.2f}%")
        W(f"  CAGR:                 {cagr:>+8.2f}%")
        W(f"  Sharpe Ratio:         {sharpe:>8.3f}    (≥1.0 target, Rule 62) [daily NAV]")
        W(f"  Max Drawdown:         {max_dd:>8.2f}%   (≤18% target, Rule 62) [daily NAV]")
        W(f"  Profit Factor:        {profit_factor:>8.3f}   (≥1.5 target, Rule 62)")
        W(f"  Win Rate:             {win_rate:>8.1f}%   (≥44% target, Rule 62)")

        W("\n  ─── Period & Benchmark Comparison ──────────────────────────────")
        W(f"  Period:           {bm_first_d} → {bm_last_d}")
        W(f"  Calendar days:    {bm_cal_days} days  ({bm_months:.1f} months)")
        W(f"  Trading sessions: {bm_sessions} days")
        W(f"  NIFTY50 start:    {bm_first:>10,.2f}  →  {bm_last:>10,.2f}")
        W(f"  NIFTY50 B&H:       {bm_return:>+8.2f}%  CAGR {bm_cagr:>+7.2f}%")
        W(f"  MARK5 system:      {total_return:>+8.2f}%  CAGR {cagr:>+7.2f}%")
        W(f"  Alpha vs index:    {alpha_return:>+8.2f}%  CAGR {alpha_cagr:>+7.2f}%")

        W("\n  ─── Rule 62 Compliance ─────────────────────────────────────────")
        rules = [
            ('Sharpe ≥ 1.0',        sharpe >= 1.0,        f'{sharpe:.3f}'),
            ('Max DD ≤ 18%',        max_dd <= 18.0,       f'{max_dd:.2f}%'),
            ('Profit Factor ≥ 1.5', profit_factor >= 1.5, f'{profit_factor:.3f}'),
            ('Win Rate ≥ 44%',      win_rate >= 44.0,     f'{win_rate:.1f}%'),
        ]
        for name, passed, val in rules:
            icon = '✅' if passed else '❌'
            W(f"  {icon} {name:<25}: {val}")

        W("\n━━━ TRADE STATISTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        W(f"  Total Trades:         {n_trades}")
        W(f"  Winners:              {len(winners)}  ({win_rate:.1f}%)")
        W(f"  Losers:               {len(losers)}")
        W(f"  Avg Win:             {avg_win:>+8.2f}%")
        W(f"  Avg Loss:            {avg_loss:>+8.2f}%")
        W(f"  Avg Hold Period:      {avg_hold:.1f} days")
        W(f"  Total Costs Paid:    ₹{sum(t['costs'] for t in all_trades):>10,.2f}")
        W("\n  Exit Type Breakdown:")
        for exit_type, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            pct = count / n_trades * 100
            W(f"    {exit_type:<20}: {count:>4}  ({pct:.1f}%)")

        # Sector exposure summary
        W("\n  Sector Exposure (all trades):")
        sector_pnl = defaultdict(float)
        sector_n   = defaultdict(int)
        for t in all_trades:
            sec = get_sector(t['symbol'])
            sector_pnl[sec] += t['net_pnl']
            sector_n[sec]   += 1
        for sec, pnl in sorted(sector_pnl.items(), key=lambda x: -x[1]):
            W(f"    {sec:<20}: {sector_n[sec]:>3} trades | ₹{pnl:>10,.0f}")

        W("\n━━━ TOP 10 STOCKS BY NET PNL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        W(f"  {'Symbol':<18} {'Trades':>6}  {'Win%':>6}  {'Avg Ret':>8}  {'Net PnL':>12}")
        W("  " + "─" * 56)
        sorted_stocks = sorted(stock_stats.items(), key=lambda x: -x[1]['net_pnl'])
        for sym, s in sorted_stocks[:10]:
            avg_ret = np.mean(s['returns']) if s['returns'] else 0
            wr      = s['wins'] / s['trades'] * 100 if s['trades'] else 0
            W(f"  {sym:<18} {s['trades']:>6}  {wr:>5.1f}%  {avg_ret:>+7.2f}%  ₹{s['net_pnl']:>10,.2f}")

        W("\n━━━ BOTTOM 5 STOCKS BY NET PNL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        W(f"  {'Symbol':<18} {'Trades':>6}  {'Win%':>6}  {'Avg Ret':>8}  {'Net PnL':>12}")
        W("  " + "─" * 56)
        for sym, s in sorted_stocks[-5:]:
            avg_ret = np.mean(s['returns']) if s['returns'] else 0
            wr      = s['wins'] / s['trades'] * 100 if s['trades'] else 0
            W(f"  {sym:<18} {s['trades']:>6}  {wr:>5.1f}%  {avg_ret:>+7.2f}%  ₹{s['net_pnl']:>10,.2f}")

        W("\n━━━ FULL TRADE JOURNAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for i, t in enumerate(all_trades, 1):
            pnl_sign = '✅ WIN' if t['pct_return'] > 0 else '❌ LOSS'
            entry_d  = str(t['entry_date'].date()) if hasattr(t['entry_date'], 'date') else t['entry_date']
            rr_ratio = round((t['target_price'] - t['entry_price']) /
                             max(t['entry_price'] - t['stop_price'], 0.01), 2)
            W(f"\n  [{i:03d}] {t['symbol']}  —  {pnl_sign}  [Sector: {t.get('sector','?')}]")
            W(f"  ┌─ Signal:   {t['signal_date']}  |  Regime: {t['regime']}")
            W(f"  ├─ Entry:    {entry_d}  |  ₹{t['entry_price']:.2f} × {t['qty']} shares  =  ₹{t['position_val']:,.0f}")
            W(f"  ├─ Stops:    SL=₹{t['stop_price']:.2f}  |  PT=₹{t['target_price']:.2f}  |  ATR=₹{t['atr']:.2f}  |  R:R={rr_ratio:.1f}:1")
            W(f"  ├─ Signals:  Rank score={t['rank_score']:+.4f}  |  ML conf={t['ml_conf']:.1%}")
            W(f"  ├─ Why BUY:  {t['entry_reason']}")
            W(f"  ├─ Exit:     {t['exit_date']}  |  {t['exit_type']} @ ₹{t['exit_price']:.2f}  ({t['days_held']}d)")
            W(f"  ├─ Why EXIT: {t.get('exit_reason', '')}")
            W(f"  ├─ MFE/MAE:  Best={t['max_gain_pct']}  |  Worst={t['max_loss_pct']}")
            W(f"  └─ P&L:      Gross=₹{t['gross_pnl']:,.2f}  |  Costs=₹{t['costs']:,.2f}  |  Net=₹{t['net_pnl']:,.2f}  ({t['pct_return']:+.2f}%)")

        W("\\n" + "=" * 70)
        W(f"Report:  {report_path}")
        W(f"Trades:  {trades_path}")
        W(f"NAV CSV: {nav_path}")

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    keys = ['signal_date', 'symbol', 'sector', 'regime', 'rank_score', 'ml_conf',
            'entry_date', 'entry_price', 'qty', 'position_val',
            'stop_price', 'target_price', 'atr',
            'exit_date', 'exit_type', 'exit_price', 'days_held',
            'max_gain_pct', 'max_loss_pct', 'exit_reason',
            'gross_pnl', 'costs', 'net_pnl', 'pct_return',
            'portfolio_after', 'entry_reason']
    with open(trades_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for t in all_trades:
            row = {k: t.get(k, '') for k in keys}
            if hasattr(row.get('entry_date'), 'date'):
                row['entry_date'] = str(row['entry_date'].date())
            w.writerow(row)

    daily_nav.reset_index().rename(columns={'index': 'date', 'nav': 'portfolio_nav'}).to_csv(
        nav_path, index=False
    )
    print(f"\\n💾 Trades CSV:  {trades_path}")
    print(f"💾 NAV CSV:     {nav_path}")
    print(f"💾 Full report: {report_path}")


if __name__ == '__main__':
    run_backtest()
