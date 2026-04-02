#!/usr/bin/env python3
"""
MARK5 "Deep History" Layer-1 Backtester
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Simulates the core momentum system (Layer 1 Base) + Strict Risk Management
over 15+ years of data (2007 - 2026) covering multiple major bear markets
(2008 GFC, 2011, 2015, 2020 COVID).

Data Source: Yahoo Finance (`yfinance`). Adjusted for splits/dividends.
Universe:    NIFTY_MIDCAP_TICKERS (Excluding NIFTY 50, per user constraint).
ML:          DISABLED to ensure ZERO in-sample lookahead bias.
Entry Rule:  Top 3 cross-sectional momentum rankers per session, minimum score > +0.5.
"""

import os, sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import numpy as np
import pandas as pd
import yfinance as yf
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

from core.models.ranker import CrossSectionalRanker
from scripts.nifty50_universe import NIFTY_MIDCAP_TICKERS

# ── BACKTEST CONFIG ──────────────────────────────────────────────────────────
PORTFOLIO_VALUE       = Decimal('2000000')   # ₹20 lakh starting capital
MAX_POSITIONS         = 5
MAX_HOLD_DAYS         = 10
RISK_PER_TRADE        = Decimal('0.015')     # 1.5% of portfolio (Rule 10)
MAX_POSITION_SIZE     = Decimal('150000')    # ₹1.5L (Rule 13)
MAX_CAPITAL_DEPLOYED  = Decimal('0.60')      # 60% max (Rule 12)
RANKING_TOP_N         = 15
BACKTEST_START        = '2007-01-01'         # Start of 2007, gives build-up to 2008 crash
BACKTEST_END          = '2026-04-01'
OUTPUT_DIR            = Path('data/backtest')
CACHE_YF_DIR          = Path('data/cache_yf')

# Costs (Rule 7) + Slippage (Rule 8: midcap = 0.10%)
BROKERAGE_PER_ORDER   = Decimal('20')
STT_RATE              = Decimal('0.001')
EXCHANGE_RATE         = Decimal('0.0000325')
GST_RATE              = Decimal('0.18')
STAMP_RATE            = Decimal('0.00015')
SLIPPAGE_RATE         = Decimal('0.001')     # Nifty 51-150 slippage

# Layer 1 specific thresholds
L1_SCORE_GATE         = 0.50                 # Minimum z-score to consider a trade

# ── DECIMAL HELPERS ──────────────────────────────────────────────────────────
def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

def compute_round_trip_cost(entry_price: Decimal, exit_price: Decimal, qty: int) -> Decimal:
    entry_value = entry_price * qty
    exit_value  = exit_price  * qty
    brokerage   = BROKERAGE_PER_ORDER * 2                      # both legs
    stt         = exit_value * STT_RATE                        # sell-side only
    exchange    = (entry_value + exit_value) * EXCHANGE_RATE
    charges     = brokerage + stt + exchange
    gst         = charges * GST_RATE
    stamp       = entry_value * STAMP_RATE
    return D(brokerage + stt + exchange + gst + stamp)

# ── REGIME DETECTION ─────────────────────────────────────────────────────────
def detect_regime(nifty: pd.Series, date: pd.Timestamp) -> str:
    hist = nifty[nifty.index <= date].tail(200)
    if len(hist) < 60: return 'RANGING'
    close   = hist.iloc[-1]
    ema50   = hist.tail(50).mean()
    ema200  = hist.mean() if len(hist) >= 200 else hist.mean()
    ret20   = (close / hist.iloc[-20] - 1) if len(hist) >= 20 else 0
    daily_rets = hist.pct_change().dropna()
    adx_proxy  = abs(daily_rets.tail(20).mean()) / (daily_rets.tail(20).std() + 1e-9) * 100

    if close < ema200 and ret20 < -0.05: return 'BEAR'
    elif adx_proxy > 25 and close > ema50 and ret20 > 0.03: return 'TRENDING'
    elif adx_proxy < 20: return 'RANGING'
    else: return 'VOLATILE'

def regime_multiplier(regime: str) -> float:
    return {'TRENDING': 1.0, 'RANGING': 0.7, 'VOLATILE': 0.5, 'BEAR': 0.3}[regime]

# ── ATR COMPUTATION ──────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    if len(df) < window + 1: return df['close'].iloc[-1] * 0.02
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h[1:]-l[1:], np.abs(h[1:]-c[:-1]))
    tr = np.maximum(tr, np.abs(l[1:]-c[:-1]))
    return float(np.mean(tr[-window:]))

def compute_position_size(portfolio: Decimal, entry: Decimal,
                          stop: Decimal, confidence: float,
                          regime: str) -> Tuple[int, Decimal]:
    risk_amount  = portfolio * RISK_PER_TRADE
    stop_dist    = max(entry - stop, D(0.01))
    raw_shares   = float(risk_amount) / float(stop_dist)
    scaled       = raw_shares * min(confidence, 1.0) * regime_multiplier(regime)
    position_val = D(scaled) * entry
    max_val = min(MAX_POSITION_SIZE, portfolio * Decimal('0.075'))
    if position_val > max_val: scaled = float(max_val / entry)
    qty = max(1, int(scaled))
    return qty, D(qty) * entry

# ── DATA DOWNLOADING & CACHING ───────────────────────────────────────────────
def get_yf_data(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    CACHE_YF_DIR.mkdir(parents=True, exist_ok=True)
    all_data = {}
    print(f"📥 Checking historical data for {len(tickers)} midcap stocks + NIFTY from 2007...")
    
    # NIFTY index
    nf_f = CACHE_YF_DIR / "NSEI_1d.parquet"
    if not nf_f.exists():
        print("  Downloading ^NSEI (NIFTY 50)...")
        nf = yf.download('^NSEI', start=BACKTEST_START, end=BACKTEST_END, progress=False)
        nf.columns = nf.columns.get_level_values(0)
        nf = nf.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        nf.index = pd.to_datetime(nf.index)
        if nf.index.tz is not None: nf.index = nf.index.tz_localize(None)
        nf.to_parquet(nf_f)
    else:
        nf = pd.read_parquet(nf_f)
    nifty_close = nf['close']
    
    # Midcap series
    for idx, sym in enumerate(tickers, 1):
        f = CACHE_YF_DIR / f"{sym.replace('.NS', '')}_1d.parquet"
        if not f.exists():
            print(f"  Downloading {sym} ({idx}/{len(tickers)})...")
            try:
                # yfinance returns adjusted close as 'Close' when auto_adjust=True (default now)
                df = yf.download(sym, start=BACKTEST_START, end=BACKTEST_END, progress=False)
                df.columns = df.columns.get_level_values(0)
                if df.empty or 'Close' not in df.columns:
                    print(f"    Missing data for {sym}")
                    continue
                df = df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                df.to_parquet(f)
                all_data[sym] = df
            except Exception as e:
                print(f"    Error fetching {sym}: {e}")
        else:
            all_data[sym] = pd.read_parquet(f)
            
    print(f"✅ Loaded {len(all_data)} stocks caching 15+ years of daily data.")
    return all_data, nifty_close

# ── POSITION SIMULATION ──────────────────────────────────────────────────────
def simulate_position(df: pd.DataFrame, entry_date: pd.Timestamp, entry_px: float,
                      stop_px: float, target_px: float) -> dict:
    future = df[df.index > entry_date].head(MAX_HOLD_DAYS + 2)
    if future.empty:
        return {'exit_type': 'NO_DATA', 'exit_price': entry_px, 'exit_date': entry_date, 'days_held': 0, 'daily_log': []}

    daily_log, max_gain, max_loss = [], 0.0, 0.0
    prev_close = entry_px

    for day_num, (date, row) in enumerate(future.iterrows()):
        open_px, high_px, low_px, close_px = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        
        # Rule 26: Gap risk check
        gap_pct = (open_px - prev_close) / prev_close
        if gap_pct < -0.03:
            return {'exit_type': 'GAP_RISK', 'exit_price': open_px, 'exit_date': date, 'days_held': day_num + 1, 'max_gain': max_gain, 'max_loss': max_loss, 'exit_reason': f'Gap down {gap_pct:.1%} > 3% threshold'}

        # Rule 15: Stop loss
        if low_px <= stop_px:
            exit_px = min(open_px, stop_px)
            return {'exit_type': 'STOP_LOSS', 'exit_price': exit_px, 'exit_date': date, 'days_held': day_num + 1, 'max_gain': max_gain, 'max_loss': max_loss, 'exit_reason': f'Stop hit at ₹{exit_px:.2f}'}

        # Rule 15: Profit target
        if high_px >= target_px:
            return {'exit_type': 'TARGET', 'exit_price': target_px, 'exit_date': date, 'days_held': day_num + 1, 'max_gain': max_gain, 'max_loss': max_loss, 'exit_reason': f'Target hit at ₹{target_px:.2f}'}

        max_gain = max(max_gain, (high_px - entry_px) / entry_px)
        max_loss = min(max_loss, (low_px - entry_px) / entry_px)
        prev_close = close_px

    last = future.iloc[-1]
    return {'exit_type': 'TIME_EXIT', 'exit_price': float(last['close']), 'exit_date': future.index[-1], 'days_held': len(future), 'max_gain': max_gain, 'max_loss': max_loss, 'exit_reason': f'Max hold period ({MAX_HOLD_DAYS} days)'}

# ── MAIN ENGINE ─────────────────────────────────────────────────────────────
def run_deep_backtest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MARK5 DEEP HISTORY BACKTESTER v1.0 (2007 - 2026)")
    print("Operating on Layer 1 Math Only (No ML Bias) | Universe: NIFTY 51-150")
    print("=" * 70)

    # 1. Fetch 19 years of data
    midcap_tickers = list(NIFTY_MIDCAP_TICKERS)
    all_data, nifty_close = get_yf_data(midcap_tickers)
    
    ranker = CrossSectionalRanker()
    
    # 2. Build dates
    start, end = pd.Timestamp(BACKTEST_START), pd.Timestamp(BACKTEST_END)
    # Align fridays strictly to NIFTY trading days
    all_dates = nifty_close[(nifty_close.index >= start) & (nifty_close.index <= end)].index
    # Group by week (ISO year-week) and take the last trading day of the week as the "Friday" equivalent
    sig_dates = pd.Series(all_dates).groupby(all_dates.strftime('%G-%V')).last().values
    sig_dates = [pd.Timestamp(d) for d in sig_dates]
    print(f"📅 Trading weeks available: {len(sig_dates)}")

    portfolio     = PORTFOLIO_VALUE
    peak_equity   = portfolio
    open_positions: List[dict] = []
    all_trades:    List[dict] = []
    equity_curve:  List[dict] = []
    
    print("\n🔄 Running 19-year simulation (progress displayed per year)...")
    last_year = 0

    for sig_date in sig_dates:
        today_naive = sig_date.normalize()
        if today_naive.year != last_year:
            print(f"  [>] Processing Year {today_naive.year}...")
            last_year = today_naive.year

        # Historical visibility cutoff
        avail = {}
        for sym, df in all_data.items():
            hist = df[df.index <= today_naive]
            # yfinance returns NA for old dates on some stocks. We must drop NAs.
            hist = hist.dropna(subset=['close'])
            if len(hist) >= 60:
                avail[sym] = hist

        if len(avail) < 10: continue

        # Resolve open positions
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

            sim = simulate_position(future_df, pos['entry_date'], pos['entry_price'], pos['stop_price'], pos['target_price'])

            if pd.Timestamp(sim['exit_date']) <= today_naive:
                exit_px  = D(sim['exit_price'])
                qty      = pos['qty']
                entry_px = D(pos['entry_price'])
                costs    = compute_round_trip_cost(entry_px, exit_px, qty)
                gross_pnl = (exit_px - entry_px) * qty
                net_pnl   = gross_pnl - costs
                portfolio += net_pnl

                all_trades.append({**pos,
                    'exit_date':   str(sim['exit_date'].date()),
                    'exit_price':  float(exit_px),
                    'exit_type':   sim['exit_type'],
                    'exit_reason': sim.get('exit_reason',''),
                    'days_held':   sim['days_held'],
                    'gross_pnl':   float(gross_pnl),
                    'net_pnl':     float(net_pnl),
                    'costs':       float(costs),
                    'pct_return':  float(net_pnl / (entry_px * qty)) * 100,
                    'portfolio_after': float(portfolio),
                })
                peak_equity = max(peak_equity, portfolio)
            else:
                still_open.append(pos)

        open_positions = still_open
        deployed = sum(D(p['entry_price']) * p['qty'] for p in open_positions)
        equity_curve.append({'date': str(sig_date.date()), 'portfolio': float(portfolio), 'deployed': float(deployed)})

        nifty_hist = nifty_close[nifty_close.index <= today_naive]
        regime     = detect_regime(nifty_hist, today_naive)
        
        # Rule 23 (Strict Layer 1 implementation): Suspend long entries in BEAR markes
        if regime == 'BEAR':
            continue

        # Layer 1 Ranking
        ranked = ranker.rank_universe(avail, nifty_hist, None, today_naive)
        if not ranked: continue
        
        # Filter: At least +0.50 cross-sectional momo required to trade without ML
        top_candidates = [cand for cand in ranked[:3] if cand[1] >= L1_SCORE_GATE]
        
        max_deploy  = float(portfolio * MAX_CAPITAL_DEPLOYED)
        cur_deploy  = sum(float(D(p['entry_price']) * p['qty']) for p in open_positions)
        open_syms   = {p['symbol'] for p in open_positions}

        for sym, rank_score in top_candidates:
            if len(open_positions) >= MAX_POSITIONS: break
            if sym in open_syms: continue

            future_df = all_data[sym][all_data[sym].index > today_naive].head(5)
            if future_df.empty: continue
            
            entry_row = future_df.iloc[0]
            entry_date = future_df.index[0]
            raw_open = D(float(entry_row['open']))
            if raw_open <= D(0.01) or pd.isna(raw_open): continue
            
            entry_px = D(float(raw_open * (1 + SLIPPAGE_RATE)))
            atr14    = compute_atr(avail[sym])
            
            # 2x ATR standard 
            stop_px  = float(D(float(entry_px) - 2 * atr14))
            target_px= float(D(float(entry_px) + 2 * atr14))

            if stop_px <= 0: continue

            # Rule 40 Gap check vs last close
            last_close = float(avail[sym]['close'].iloc[-1].dropna() if hasattr(avail[sym]['close'].iloc[-1], 'dropna') else avail[sym]['close'].iloc[-1])
            if pd.isna(last_close) or last_close <= 0: continue
            
            gap_vs_close = (float(raw_open) - last_close) / last_close
            if gap_vs_close < -0.01: continue

            # In Layer 1-only, surrogate 'confidence' as 0.70 to trigger standardized position scaling
            surrogate_conf = 0.70
            qty, pos_val = compute_position_size(portfolio, entry_px, D(stop_px), surrogate_conf, regime)

            if pos_val > MAX_POSITION_SIZE:
                 qty = int(float(MAX_POSITION_SIZE) / float(entry_px))
                 pos_val = D(qty) * entry_px

            if qty < 1 or float(pos_val) > (max_deploy - cur_deploy): continue

            cur_deploy += float(pos_val)
            open_syms.add(sym)

            open_positions.append({
                'symbol':       sym,
                'signal_date':  str(sig_date.date()),
                'entry_date':   entry_date,
                'entry_price':  float(entry_px),
                'stop_price':   stop_px,
                'target_price': target_px,
                'qty':          qty,
                'position_val': float(pos_val),
                'rank_score':   round(rank_score, 4),
                'regime':       regime,
                'entry_reason': f"Layer-1 Only | Rank #{[s for s,_ in ranked].index(sym)+1} | Score={rank_score:+.3f} > Gate"
            })

    # Close left overs
    for pos in open_positions:
        sym = pos['symbol']
        last_row = all_data[sym].tail(1)
        if last_row.empty: continue
        exit_px = D(float(last_row['close'].iloc[-1]))
        qty, entry_px = pos['qty'], D(pos['entry_price'])
        costs     = compute_round_trip_cost(entry_px, exit_px, qty)
        gross_pnl = (exit_px - entry_px) * qty
        net_pnl   = gross_pnl - costs
        portfolio += net_pnl
        all_trades.append({**pos, 'exit_date': str(last_row.index[-1].date()), 'exit_price': float(exit_px), 'exit_type': 'END_OF_BACKTEST', 'net_pnl': float(net_pnl), 'pct_return': float(net_pnl / (entry_px * qty)) * 100, 'costs': float(costs)})

    # Calculate Benchmark
    bm_start, bm_end = pd.Timestamp(BACKTEST_START), sig_dates[-1] if len(sig_dates) > 0 else pd.Timestamp(BACKTEST_END)
    bm_data  = nifty_close[(nifty_close.index >= bm_start) & (nifty_close.index <= bm_end)]
    if not bm_data.empty:
        bm_ret = (bm_data.iloc[-1] / bm_data.iloc[0] - 1) * 100
        bm_yrs = (bm_data.index[-1] - bm_data.index[0]).days / 365.25
        bm_cagr= ((bm_data.iloc[-1] / bm_data.iloc[0]) ** (1 / max(bm_yrs, 1)) - 1) * 100
    else:
        bm_ret, bm_cagr = 0.0, 0.0

    # Write Stats
    total_ret = (float(portfolio) / float(PORTFOLIO_VALUE) - 1) * 100
    yrs       = len(sig_dates) / 52.1
    cagr      = ((float(portfolio) / float(PORTFOLIO_VALUE)) ** (1 / max(yrs, 1)) - 1) * 100
    wins      = [t for t in all_trades if t['pct_return'] > 0]
    wr        = len(wins) / max(len(all_trades), 1) * 100

    eq_vals   = [e['portfolio'] for e in equity_curve]
    peak, mdd = eq_vals[0] if eq_vals else portfolio, 0.0
    for v in eq_vals:
        peak = max(peak, v)
        mdd  = max(mdd, (peak - v) / peak * 100)

    report_f = OUTPUT_DIR / 'deep_report.txt'
    with open(report_f, 'w') as f:
        f.write(f"MARK5 DEEP HISTORY {BACKTEST_START} to {BACKTEST_END}\n")
        f.write("UNIVERSE: NIFTY 51-150 (Midcap 100 exactly)\n")
        f.write(f"Final Value: ₹{portfolio:,.2f}  (+{total_ret:.2f}% | CAGR: {cagr:.2f}%)\n")
        f.write(f"NIFTY Bench: Buy&Hold   (+{bm_ret:.2f}% | CAGR: {bm_cagr:.2f}%)\n")
        f.write(f"Alpha CAGR:  {cagr - bm_cagr:+.2f}%\n")
        f.write(f"Max DD:      -{mdd:.2f}%\n")
        f.write(f"Trades:      {len(all_trades)}  (Win Rate: {wr:.1f}%)\n")
    
    # Save curves
    pd.DataFrame(equity_curve).to_csv(OUTPUT_DIR / 'deep_equity.csv', index=False)
    pd.DataFrame(all_trades).to_csv(OUTPUT_DIR / 'deep_trades.csv', index=False)
    
    print("\n✅ Deep Historical Backtest Complete!")
    print(f"  Final Capital: ₹{portfolio:,.2f} (CAGR: {cagr:.2f}%)")
    print(f"  Nifty Benchmark CAGR: {bm_cagr:.2f}%")
    print(f"  Alpha: {cagr - bm_cagr:+.2f}%/yr")
    print(f"  Max Drawdown: -{mdd:.2f}%")
    print(f"  Report saved to: {report_f}")


if __name__ == '__main__':
    run_deep_backtest()
