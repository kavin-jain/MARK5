#!/usr/bin/env python3
"""
MARK5 Full Universe Backtester v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Simulates the complete live pipeline (Layer 1 Ranker → Layer 2 ML → Risk Gates)
on historical data across the full 105-stock MARK5 universe.

OUTPUT (three files written to data/backtest/):
  1. backtest_report.txt   — full trade journal + portfolio stats
  2. backtest_trades.csv   — machine-readable trade log
  3. backtest_equity.csv   — daily equity curve

METHODOLOGY (zero lookahead):
  - Signal date:  every Friday (market close)
  - Entry:        following Monday open + slippage
  - Exit:         stop loss / profit target / 10-day time limit
  - Regime:       detected from NIFTY data available AT signal date
  - Costs:        Rule 7 (full round-trip: brokerage + STT + exchange + GST + stamp)
  - Slippage:     Rule 8 (0.05% large-cap, 0.10% mid-cap)
  - Position size: Rule 22 (confidence-scaled) + Rule 13 (₹1.5L cap)
  - Max positions: Rule 11 (5 simultaneous)
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
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import csv, json

from core.models.ranker import CrossSectionalRanker
from core.models.predictor import MARK5Predictor
from scripts.nifty50_universe import MARK5_LIVE_TICKERS, NIFTY_MIDCAP_TICKERS

# ── BACKTEST CONFIG ──────────────────────────────────────────────────────────
PORTFOLIO_VALUE       = Decimal('2000000')   # ₹20 lakh starting capital
MAX_POSITIONS         = 5
MAX_HOLD_DAYS         = 10
RISK_PER_TRADE        = Decimal('0.015')     # 1.5% of portfolio (Rule 10)
MAX_POSITION_SIZE     = Decimal('150000')    # ₹1.5L (Rule 13)
MAX_CAPITAL_DEPLOYED  = Decimal('0.60')      # 60% max (Rule 12)
CONFIDENCE_NORMAL     = 0.55
CONFIDENCE_BEAR       = 0.70
RANKING_TOP_N         = 15
BACKTEST_START        = '2024-10-01'         # genuine OOS: after training window ends
BACKTEST_END          = '2026-04-01'
OUTPUT_DIR            = Path('data/backtest')
CACHE_DIR             = Path('data/cache')

# Cost constants (Rule 7)
BROKERAGE_PER_ORDER   = Decimal('20')        # flat ₹20 per order
STT_RATE              = Decimal('0.001')     # 0.1% on sell side
EXCHANGE_RATE         = Decimal('0.0000325') # 0.00325%
GST_RATE              = Decimal('0.18')      # 18% on charges
STAMP_RATE            = Decimal('0.00015')   # 0.015% on buy side

# Midcap stocks get higher slippage (Rule 8)
MIDCAP_SET = {t.replace('.NS','') for t in NIFTY_MIDCAP_TICKERS}

# ── DECIMAL HELPERS ──────────────────────────────────────────────────────────
def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

def compute_round_trip_cost(symbol: str, entry_price: Decimal,
                             exit_price: Decimal, qty: int) -> Decimal:
    """Full round-trip cost per Rule 7."""
    sym_bare = symbol.replace('.NS', '')
    entry_value = entry_price * qty
    exit_value  = exit_price  * qty

    brokerage   = BROKERAGE_PER_ORDER * 2                      # both legs
    stt         = exit_value * STT_RATE                        # sell-side only
    exchange    = (entry_value + exit_value) * EXCHANGE_RATE
    charges     = brokerage + stt + exchange
    gst         = charges * GST_RATE
    stamp       = entry_value * STAMP_RATE

    return D(brokerage + stt + exchange + gst + stamp)

def slippage_rate(symbol: str) -> Decimal:
    """Rule 8: 0.05% for NIFTY50, 0.10% for midcap."""
    sym_bare = symbol.replace('.NS', '')
    return Decimal('0.001') if sym_bare in MIDCAP_SET else Decimal('0.0005')

# ── REGIME DETECTION ─────────────────────────────────────────────────────────
def detect_regime(nifty: pd.Series, date: pd.Timestamp) -> str:
    """Rule 23: detect TRENDING/RANGING/VOLATILE/BEAR."""
    hist = nifty[nifty.index <= date].tail(200)
    if len(hist) < 60:
        return 'RANGING'

    close   = hist.iloc[-1]
    ema50   = hist.tail(50).mean()
    ema200  = hist.mean() if len(hist) >= 200 else hist.mean()
    ret20   = (close / hist.iloc[-20] - 1) if len(hist) >= 20 else 0

    # ADX proxy: trend strength via 20-day relative volatility
    daily_rets = hist.pct_change().dropna()
    adx_proxy  = abs(daily_rets.tail(20).mean()) / (daily_rets.tail(20).std() + 1e-9) * 100

    if close < ema200 and ret20 < -0.05:
        return 'BEAR'
    elif adx_proxy > 25 and close > ema50 and ret20 > 0.03:
        return 'TRENDING'
    elif adx_proxy < 20:
        return 'RANGING'
    else:
        return 'VOLATILE'

def regime_multiplier(regime: str) -> float:
    return {'TRENDING': 1.0, 'RANGING': 0.7, 'VOLATILE': 0.5, 'BEAR': 0.3}[regime]

def confidence_threshold(regime: str) -> float:
    return CONFIDENCE_BEAR if regime == 'BEAR' else CONFIDENCE_NORMAL

# ── ATR COMPUTATION ──────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    if len(df) < window + 1:
        return df['close'].iloc[-1] * 0.02
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h[1:]-l[1:], np.abs(h[1:]-c[:-1]))
    tr = np.maximum(tr, np.abs(l[1:]-c[:-1]))
    return float(np.mean(tr[-window:]))

# ── POSITION SIZING ─────────────────────────────────────────────────────────
def compute_position_size(portfolio: Decimal, entry: Decimal,
                          stop: Decimal, confidence: float,
                          regime: str) -> Tuple[int, Decimal]:
    """Rule 22: confidence + regime scaled position sizing."""
    risk_amount  = portfolio * RISK_PER_TRADE
    stop_dist    = max(entry - stop, D(0.01))
    raw_shares   = float(risk_amount) / float(stop_dist)
    scaled       = raw_shares * min(confidence, 1.0) * regime_multiplier(regime)
    position_val = D(scaled) * entry

    # Rule 13: cap at ₹1.5L or 7.5% of portfolio
    max_val = min(MAX_POSITION_SIZE, portfolio * Decimal('0.075'))
    if position_val > max_val:
        scaled = float(max_val / entry)

    qty = max(1, int(scaled))
    return qty, D(qty) * entry

# ── POSITION SIMULATION ──────────────────────────────────────────────────────
def simulate_position(symbol: str, df: pd.DataFrame,
                      entry_date: pd.Timestamp, entry_px: float,
                      stop_px: float, target_px: float) -> dict:
    """
    Simulate a position from entry_date forward through the OHLCV data.
    Returns exit info with full day-by-day tracking.
    """
    future = df[df.index > entry_date].head(MAX_HOLD_DAYS + 2)
    if future.empty:
        return {'exit_type': 'NO_DATA', 'exit_price': entry_px,
                'exit_date': entry_date, 'days_held': 0, 'daily_log': []}

    daily_log = []
    max_gain  = 0.0
    max_loss  = 0.0
    prev_close = entry_px

    for day_num, (date, row) in enumerate(future.iterrows()):
        open_px  = float(row['open'])
        high_px  = float(row['high'])
        low_px   = float(row['low'])
        close_px = float(row['close'])

        # Gap risk check (Rule 26: gap > 3% against → exit at open)
        gap_pct = (open_px - prev_close) / prev_close
        if gap_pct < -0.03:
            return {
                'exit_type':  'GAP_RISK',
                'exit_price': open_px,
                'exit_date':  date,
                'days_held':  day_num + 1,
                'max_gain':   max_gain,
                'max_loss':   max_loss,
                'daily_log':  daily_log,
                'exit_reason': f'Gap down {gap_pct:.1%} > 3% threshold (Rule 26)',
            }

        # Stop loss (Rule 15)
        if low_px <= stop_px:
            exit_px = min(open_px, stop_px)  # realistic fill
            return {
                'exit_type':  'STOP_LOSS',
                'exit_price': exit_px,
                'exit_date':  date,
                'days_held':  day_num + 1,
                'max_gain':   max_gain,
                'max_loss':   max_loss,
                'daily_log':  daily_log,
                'exit_reason': f'Stop hit at ₹{exit_px:.2f} (stop was ₹{stop_px:.2f})',
            }

        # Profit target
        if high_px >= target_px:
            return {
                'exit_type':  'TARGET',
                'exit_price': target_px,
                'exit_date':  date,
                'days_held':  day_num + 1,
                'max_gain':   max_gain,
                'max_loss':   max_loss,
                'daily_log':  daily_log,
                'exit_reason': f'Target hit at ₹{target_px:.2f}',
            }

        intraday_gain = (high_px - entry_px) / entry_px
        intraday_loss = (low_px  - entry_px) / entry_px
        max_gain = max(max_gain, intraday_gain)
        max_loss = min(max_loss, intraday_loss)
        prev_close = close_px

        daily_log.append({
            'date': str(date.date()),
            'open': open_px, 'high': high_px,
            'low': low_px, 'close': close_px,
            'gap_pct': f'{gap_pct:.2%}',
            'vs_entry': f'{(close_px - entry_px) / entry_px:.2%}',
        })

    # Time exit (Rule 3: max 10 days)
    last = future.iloc[-1]
    return {
        'exit_type':  'TIME_EXIT',
        'exit_price': float(last['close']),
        'exit_date':  future.index[-1],
        'days_held':  len(future),
        'max_gain':   max_gain,
        'max_loss':   max_loss,
        'daily_log':  daily_log,
        'exit_reason': f'Max hold period reached ({MAX_HOLD_DAYS} days, Rule 3)',
    }


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MARK5 FULL UNIVERSE BACKTESTER v1.0")
    print(f"Universe: {len(MARK5_LIVE_TICKERS)} stocks | "
          f"Period: {BACKTEST_START} → {BACKTEST_END}")
    print(f"Portfolio: ₹{int(PORTFOLIO_VALUE):,} | Max {MAX_POSITIONS} positions")
    print("=" * 70)

    # ── Load all OHLCV data from cache ───────────────────────────────────────
    print("\n📊 Loading data from cache...")
    all_data: Dict[str, pd.DataFrame] = {}
    for sym in MARK5_LIVE_TICKERS:
        bare = sym.replace('.NS', '')
        cache_f = CACHE_DIR / f"{bare}_NS_1d.parquet"
        if not cache_f.exists():
            continue
        df = pd.read_parquet(cache_f)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        all_data[sym] = df
    print(f"  Loaded: {len(all_data)} stocks")

    # ── Load NIFTY50 index ───────────────────────────────────────────────────
    nifty_raw = pd.read_parquet(CACHE_DIR / 'NIFTY50_1d.parquet')
    nifty_close = nifty_raw['close'] if 'close' in nifty_raw.columns else nifty_raw['Close']
    if nifty_close.index.tz is not None:
        nifty_close.index = nifty_close.index.tz_localize(None)

    # ── Initialise models ────────────────────────────────────────────────────
    print("🤖 Loading ML models...")
    ranker    = CrossSectionalRanker()
    predictors: Dict[str, MARK5Predictor] = {}
    for sym in all_data:
        try:
            p = MARK5Predictor(sym)
            if p._container is not None:
                predictors[sym] = p
        except Exception:
            pass
    print(f"  Models loaded: {len(predictors)}")

    # ── Build list of signal dates (every Friday in period) ──────────────────
    start   = pd.Timestamp(BACKTEST_START)
    end     = pd.Timestamp(BACKTEST_END)
    fridays = pd.date_range(start, end, freq='W-FRI')
    fridays = [d for d in fridays if d >= start]
    print(f"  Signal dates: {len(fridays)} Fridays")

    # ── State ────────────────────────────────────────────────────────────────
    portfolio     = PORTFOLIO_VALUE
    peak_equity   = portfolio
    open_positions: List[dict] = []   # active trades
    all_trades:    List[dict] = []    # completed trades
    equity_curve:  List[dict] = []    # daily equity snapshots
    weekly_pnl_pct = []

    print("\n🔄 Running simulation...\n")

    for sig_date in fridays:
        today_naive = sig_date.normalize()

        # Slice data available up to signal date (Rule 61: no lookahead)
        avail = {}
        for sym, df in all_data.items():
            hist = df[df.index <= today_naive]
            if len(hist) >= 60:
                avail[sym] = hist

        if len(avail) < 10:
            continue

        # First close out any positions that should exit today
        still_open = []
        for pos in open_positions:
            sym   = pos['symbol']
            if sym not in all_data:
                still_open.append(pos)
                continue

            future_df = all_data[sym][all_data[sym].index > pos['entry_date']]
            if future_df.empty:
                still_open.append(pos)
                continue

            sim = simulate_position(
                sym, all_data[sym],
                pos['entry_date'], pos['entry_price'],
                pos['stop_price'], pos['target_price'],
            )

            # If position has exited (exit_date <= today)
            if pd.Timestamp(sim['exit_date']) <= today_naive:
                exit_px  = D(sim['exit_price'])
                qty      = pos['qty']
                entry_px = D(pos['entry_price'])
                costs    = compute_round_trip_cost(sym, entry_px, exit_px, qty)

                gross_pnl = (exit_px - entry_px) * qty
                net_pnl   = gross_pnl - costs
                pct_ret   = float(net_pnl / (entry_px * qty)) * 100

                portfolio += net_pnl

                trade = {**pos,
                    'exit_date':   str(sim['exit_date'].date()),
                    'exit_price':  float(exit_px),
                    'exit_type':   sim['exit_type'],
                    'exit_reason': sim.get('exit_reason',''),
                    'days_held':   sim['days_held'],
                    'max_gain_pct':f"{sim.get('max_gain',0)*100:.2f}%",
                    'max_loss_pct':f"{sim.get('max_loss',0)*100:.2f}%",
                    'gross_pnl':   float(gross_pnl),
                    'net_pnl':     float(net_pnl),
                    'costs':       float(costs),
                    'pct_return':  round(pct_ret, 3),
                    'portfolio_after': float(portfolio),
                }
                all_trades.append(trade)
                peak_equity = max(peak_equity, portfolio)
            else:
                still_open.append(pos)

        open_positions = still_open

        # Equity snapshot
        deployed = sum(D(p['entry_price']) * p['qty'] for p in open_positions)
        equity_curve.append({
            'date':       str(sig_date.date()),
            'portfolio':  float(portfolio),
            'deployed':   float(deployed),
            'open_pos':   len(open_positions),
        })

        # Regime detection
        nifty_hist = nifty_close[nifty_close.index <= today_naive]
        regime     = detect_regime(nifty_hist, today_naive)
        conf_gate  = confidence_threshold(regime)
        reg_mult   = regime_multiplier(regime)
        fii_net    = None  # FII data not historically available in cache

        # Layer 1: Rank
        ranked = ranker.rank_universe(avail, nifty_hist, fii_net, today_naive)
        if not ranked:
            continue
        top_candidates = ranked[:RANKING_TOP_N]

        # Layer 2: ML scoring
        candidates_with_conf = []
        for sym, rank_score in top_candidates:
            if sym not in predictors or sym not in avail:
                continue
            try:
                result    = predictors[sym].predict(avail[sym])
                if result.get('status') != 'success':
                    continue
                ml_conf   = result.get('confidence', 0.5)
                features  = result.get('features', {})
                candidates_with_conf.append({
                    'symbol':     sym,
                    'rank_score': rank_score,
                    'ml_conf':    ml_conf,
                    'features':   features,
                })
            except Exception:
                pass

        # Gate: only tradeable signals
        tradeable = [c for c in candidates_with_conf if c['ml_conf'] >= conf_gate]

        # Open new positions (respect slot + capital limits)
        max_deploy  = float(portfolio * MAX_CAPITAL_DEPLOYED)
        cur_deploy  = sum(float(D(p['entry_price']) * p['qty']) for p in open_positions)
        open_syms   = {p['symbol'] for p in open_positions}
        # Sector check (Rule 14: max 2 per sector) — simplified via symbol check

        for cand in sorted(tradeable, key=lambda x: x['ml_conf'], reverse=True):
            sym = cand['symbol']
            if len(open_positions) >= MAX_POSITIONS:
                break
            if sym in open_syms:
                continue

            # Find next Monday open price (entry day)
            future_df = all_data[sym][all_data[sym].index > today_naive].head(5)
            if future_df.empty:
                continue
            entry_row = future_df.iloc[0]
            entry_date = future_row_date = future_df.index[0]

            # Slippage (Rule 8)
            slip = slippage_rate(sym)
            raw_open = D(float(entry_row['open']))
            entry_px = D(float(raw_open * (1 + slip)))

            # Stop loss: 2×ATR14 below entry (Rule 15)
            atr14    = compute_atr(avail[sym])
            stop_px  = float(D(float(entry_px) - 2 * atr14))
            target_px= float(D(float(entry_px) + 2 * atr14))   # symmetric 2×ATR target

            if stop_px <= 0 or entry_px <= D(0):
                continue

            # Skip if open gap > 1% adverse vs last close (Rule 40)
            last_close = float(avail[sym]['close'].iloc[-1])
            gap_vs_close = (float(raw_open) - last_close) / last_close
            if gap_vs_close < -0.01:
                continue

            # Position sizing (Rule 22)
            qty, pos_val = compute_position_size(
                portfolio, entry_px, D(stop_px),
                cand['ml_conf'], regime
            )

            if pos_val > MAX_POSITION_SIZE:
                qty     = int(float(MAX_POSITION_SIZE) / float(entry_px))
                pos_val = D(qty) * entry_px

            if qty < 1 or float(pos_val) > (max_deploy - cur_deploy):
                continue

            cur_deploy += float(pos_val)
            open_syms.add(sym)

            # Capital check Rule 12
            if D(cur_deploy) > portfolio * MAX_CAPITAL_DEPLOYED:
                break

            pos_record = {
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
                'features':     {k: round(float(v), 4) for k, v in
                                 (cand['features'] or {}).items()},
                'entry_reason': (
                    f"Rank #{[s for s,_ in ranked].index(sym)+1}/105 "
                    f"| ML={cand['ml_conf']:.1%} ≥ gate={conf_gate:.0%} "
                    f"| Regime={regime} | Score={cand['rank_score']:+.4f}"
                ),
            }
            open_positions.append(pos_record)

    # ── Close any remaining open positions at last available price ───────────
    for pos in open_positions:
        sym  = pos['symbol']
        if sym not in all_data:
            continue
        last_row  = all_data[sym][all_data[sym].index > pos['entry_date']].tail(1)
        if last_row.empty:
            continue
        exit_px   = D(float(last_row['close'].iloc[-1]))
        qty       = pos['qty']
        entry_px  = D(pos['entry_price'])
        costs     = compute_round_trip_cost(sym, entry_px, exit_px, qty)
        gross_pnl = (exit_px - entry_px) * qty
        net_pnl   = gross_pnl - costs
        pct_ret   = float(net_pnl / (entry_px * qty)) * 100
        portfolio += net_pnl

        all_trades.append({**pos,
            'exit_date':   str(last_row.index[-1].date()),
            'exit_price':  float(exit_px),
            'exit_type':   'STILL_OPEN_CLOSE',
            'exit_reason': 'End of backtest — closed at last price',
            'days_held':   0,
            'max_gain_pct': 'N/A',
            'max_loss_pct': 'N/A',
            'gross_pnl':   float(gross_pnl),
            'net_pnl':     float(net_pnl),
            'costs':       float(costs),
            'pct_return':  round(pct_ret, 3),
            'portfolio_after': float(portfolio),
        })

    # ── COMPUTE STATS ────────────────────────────────────────────────────────
    final_portfolio = float(portfolio)
    total_return    = (final_portfolio - float(PORTFOLIO_VALUE)) / float(PORTFOLIO_VALUE) * 100
    n_trades        = len(all_trades)

    if n_trades == 0:
        print("⚠️  No trades executed in the backtest period.")
        return

    returns      = [t['pct_return'] / 100 for t in all_trades]
    winners      = [t for t in all_trades if t['pct_return'] > 0]
    losers       = [t for t in all_trades if t['pct_return'] <= 0]
    win_rate     = len(winners) / n_trades * 100
    avg_win      = np.mean([t['pct_return'] for t in winners]) if winners else 0
    avg_loss     = np.mean([t['pct_return'] for t in losers])  if losers  else 0
    avg_hold     = np.mean([t['days_held'] for t in all_trades if t['days_held'] > 0])

    # Profit factor
    gross_profit = sum(t['net_pnl'] for t in winners) if winners else 0
    gross_loss   = abs(sum(t['net_pnl'] for t in losers)) if losers else 1
    profit_factor = round(gross_profit / max(gross_loss, 0.01), 3)

    # Sharpe (daily, annualised)
    daily_rets = pd.Series(returns)
    sharpe     = round(float(daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252/avg_hold if avg_hold > 0 else 252)), 3)

    # Max drawdown from equity curve
    eq_vals  = [e['portfolio'] for e in equity_curve]
    peak     = eq_vals[0]
    max_dd   = 0.0
    for v in eq_vals:
        peak   = max(peak, v)
        dd     = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    # Period length
    n_months = len(fridays) / 4.33
    cagr     = ((final_portfolio / float(PORTFOLIO_VALUE)) ** (12 / max(n_months, 1)) - 1) * 100

    # Benchmark: NIFTY50 buy & hold over same period
    bm_start = pd.Timestamp(BACKTEST_START)
    bm_end   = pd.Timestamp(BACKTEST_END)
    bm_data  = nifty_close[(nifty_close.index >= bm_start) & (nifty_close.index <= bm_end)]
    bm_first_val = float(bm_data.iloc[0])  if not bm_data.empty else None
    bm_last_val  = float(bm_data.iloc[-1]) if not bm_data.empty else None
    bm_first_date= bm_data.index[0].date()  if not bm_data.empty else BACKTEST_START
    bm_last_date = bm_data.index[-1].date() if not bm_data.empty else BACKTEST_END
    bm_cal_days  = (bm_data.index[-1] - bm_data.index[0]).days if not bm_data.empty else 0
    bm_months    = bm_cal_days / 30.44
    bm_sessions  = len(bm_data)
    if bm_first_val and bm_last_val:
        bm_return = (bm_last_val / bm_first_val - 1) * 100
        bm_cagr   = ((bm_last_val / bm_first_val) ** (12 / max(bm_months, 1)) - 1) * 100
    else:
        bm_return = bm_cagr = 0.0
    alpha_return = total_return - bm_return
    alpha_cagr   = cagr - bm_cagr

    # By-exit-type breakdown
    from collections import Counter
    exit_counts = Counter(t['exit_type'] for t in all_trades)

    # ── Per-stock summary ────────────────────────────────────────────────────
    from collections import defaultdict
    stock_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'net_pnl': 0.0, 'returns': []})
    for t in all_trades:
        s = stock_stats[t['symbol']]
        s['trades'] += 1
        s['net_pnl'] += t['net_pnl']
        s['returns'].append(t['pct_return'])
        if t['pct_return'] > 0:
            s['wins'] += 1

    # ── WRITE REPORT ─────────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / 'backtest_report.txt'
    trades_path = OUTPUT_DIR / 'backtest_trades.csv'
    equity_path = OUTPUT_DIR / 'backtest_equity.csv'

    with open(report_path, 'w', encoding='utf-8') as rpt:
        def W(line=''):
            rpt.write(line + '\n')
            print(line)

        W("=" * 70)
        W("MARK5 FULL UNIVERSE BACKTEST REPORT")
        W(f"Period:    {BACKTEST_START} → {BACKTEST_END}")
        W(f"Universe:  {len(all_data)} stocks (NIFTY50 + Midcap100)")
        W(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST')}")
        W("=" * 70)

        W("\n━━━ PORTFOLIO SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        W(f"  Starting Capital:  ₹{float(PORTFOLIO_VALUE):>12,.2f}")
        W(f"  Final Capital:     ₹{final_portfolio:>12,.2f}")
        W(f"  Total Return:         {total_return:>+8.2f}%")
        W(f"  CAGR:                 {cagr:>+8.2f}%")
        W(f"  Sharpe Ratio:         {sharpe:>8.3f}    (≥1.0 target, Rule 62)")
        W(f"  Max Drawdown:         {max_dd:>8.2f}%   (≤18% target, Rule 62)")
        W(f"  Profit Factor:        {profit_factor:>8.3f}   (≥1.5 target, Rule 62)")
        W(f"  Win Rate:             {win_rate:>8.1f}%   (≥44% target, Rule 62)")

        W("\n  ─── Period & Benchmark Comparison ──────────────────────────────")
        W(f"  Period:          {bm_first_date} → {bm_last_date}")
        W(f"  Calendar days:   {bm_cal_days} days  ({bm_months:.1f} months)")
        W(f"  Trading sessions: {bm_sessions} days")
        W(f"  NIFTY50 start:   {bm_first_val:>10,.2f}  →  {bm_last_val:>10,.2f}")
        W(f"  NIFTY50 B&H:      {bm_return:>+8.2f}%  CAGR {bm_cagr:>+7.2f}%")
        W(f"  MARK5 system:     {total_return:>+8.2f}%  CAGR {cagr:>+7.2f}%")
        W(f"  Alpha vs index:   {alpha_return:>+8.2f}%  CAGR {alpha_cagr:>+7.2f}%  ← edge")

        # Rule 62 compliance
        W("\n  ─── Rule 62 Compliance ─────────────────────────────────────────")
        rules = [
            ('Sharpe ≥ 1.0',    sharpe >= 1.0,    f'{sharpe:.3f}'),
            ('Max DD ≤ 18%',    max_dd <= 18.0,   f'{max_dd:.2f}%'),
            ('Profit Factor ≥ 1.5', profit_factor >= 1.5, f'{profit_factor:.3f}'),
            ('Win Rate ≥ 44%',  win_rate >= 44.0, f'{win_rate:.1f}%'),
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
        W(f"\n  Exit Type Breakdown:")
        for exit_type, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            pct = count / n_trades * 100
            W(f"    {exit_type:<20}: {count:>4}  ({pct:.1f}%)")

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
            W(f"\n  [{i:03d}] {t['symbol']}  —  {pnl_sign}")
            W(f"  ┌─ Signal:   {t['signal_date']}  |  Regime: {t['regime']}")
            W(f"  ├─ Entry:    {str(t['entry_date'].date()) if hasattr(t['entry_date'],'date') else t['entry_date']}  |  ₹{t['entry_price']:.2f} × {t['qty']} shares  =  ₹{t['position_val']:,.0f}")
            W(f"  ├─ Stops:    SL=₹{t['stop_price']:.2f}  |  PT=₹{t['target_price']:.2f}  |  ATR=₹{t['atr']:.2f}")
            W(f"  ├─ Signals:  Rank score={t['rank_score']:+.4f}  |  ML conf={t['ml_conf']:.1%}")
            W(f"  ├─ Why BUY:  {t['entry_reason']}")
            if t['features']:
                feat_str = '  '.join(f"{k}={v:+.3f}" for k, v in list(t['features'].items())[:5])
                W(f"  ├─ Features: {feat_str}")
            W(f"  ├─ Exit:     {t['exit_date']}  |  {t['exit_type']} @ ₹{t['exit_price']:.2f}  ({t['days_held']}d)")
            W(f"  ├─ Why EXIT: {t.get('exit_reason','')}")
            W(f"  ├─ MFE/MAE:  Best={t['max_gain_pct']}  |  Worst={t['max_loss_pct']}")
            W(f"  └─ P&L:      Gross=₹{t['gross_pnl']:,.2f}  |  Costs=₹{t['costs']:,.2f}  |  Net=₹{t['net_pnl']:,.2f}  ({t['pct_return']:+.2f}%)")

        W("\n" + "=" * 70)
        W(f"Report saved: {report_path}")

    # ── Write CSVs ───────────────────────────────────────────────────────────
    if all_trades:
        keys = ['signal_date', 'symbol', 'regime', 'rank_score', 'ml_conf',
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
        print(f"\n💾 Trades CSV: {trades_path}")

    with open(equity_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['date','portfolio','deployed','open_pos'])
        w.writeheader()
        w.writerows(equity_curve)
    print(f"💾 Equity CSV: {equity_path}")
    print(f"💾 Full report: {report_path}")


if __name__ == '__main__':
    run_backtest()
