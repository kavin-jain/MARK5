#!/usr/bin/env python3
"""
MARK5 Universe IC Comparison — NIFTY50 vs NIFTY Midcap 100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Downloads daily data for Midcap 100 stocks via Kite and computes:
  1. Cross-sectional IC of 10-day forward return vs momentum signals
  2. ATR comparison (more ATR = more profit potential per trade)
  3. Volume liquidity check (can we fill ₹1.5L at open?)
  4. Signal persistence (how many days does the momentum signal last?)
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from dotenv import load_dotenv; load_dotenv('.env')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from core.data.adapters.kite_adapter import KiteFeedAdapter

# ── Confirmed liquid NIFTY Midcap 100 stocks (ranks ~51-150) ─────────────────
MIDCAP_100 = [
    # Financials / NBFC
    'CHOLAFIN', 'MUTHOOTFIN', 'SUNDARMFIN', 'IDFCFIRSTB', 'AUBANK',
    'BANKBARODA', 'PNB', 'JIOFIN',
    # Power / Infra / PSU
    'PFC', 'RECLTD', 'IRFC', 'GAIL', 'HAL', 'BHEL', 'CONCOR',
    # Real Estate
    'DLF', 'LODHA', 'PRESTIGE',
    # Industrials / Capital Goods
    'SIEMENS', 'ABB', 'HAVELLS', 'CGPOWER', 'POLYCAB', 'CUMMINSIND', 'DIXON',
    # IT / Technology
    'LTIM', 'TATATECH', 'PERSISTENT',
    # Auto / Auto Ancillaries
    'TVSMOTOR', 'MOTHERSON', 'APOLLOTYRE', 'MRF',
    # Pharma / Healthcare
    'TORNTPHARM', 'ZYDUSLIFE', 'MAXHEALTH', 'ALKEM',
    # FMCG / Consumer
    'MARICO', 'GODREJCP', 'BERGEPAINT', 'DABUR', 'COLPAL',
    # Travel / Hospitality
    'INDIGO', 'INDHOTEL',
    # Markets / Finance
    'MCX', 'BSE', 'CDSL',
    # Energy / Chemicals
    'IGL', 'PETRONET', 'PIIND', 'SRF', 'DEEPAKNTR',
    # Others
    'JUBLFOOD', 'TRENT', 'VOLTAS',
]

# ── NIFTY500 range — ranks ~150-400 (SmallCap / Next 150) ────────────────────
# These are liquid but less-covered stocks. Testing if IC is even higher.
SMALLCAP_NEXT = [
    # PSU Power/Infra (high momentum, less covered)
    'NHPC', 'SJVN', 'RVNL', 'RAILTEL', 'IRCON', 'NBCC', 'HUDCO', 'GMRINFRA',
    # PSU Banks (high beta, volatile)
    'IOB', 'UCOBANK', 'CENTRALBK', 'MAHABANK', 'BANDHANBNK',
    # Commodities / Metals
    'SAIL', 'NMDC', 'NATIONALUM', 'HINDCOPPER', 'MOIL',
    # Energy
    'IEX', 'CASTROLIND', 'GSPL',
    # NBFC / Finance (mid-small)
    'MANAPPURAM', 'IIFL', 'CREDITACC',
    # Pharma (smaller names)
    'AUROPHARMA', 'GLENMARK', 'IPCALAB', 'LUPIN',
    # Chemicals
    'GNFC', 'GSFC', 'CHAMBLFERT', 'AARTIIND', 'NAVINFLUOR', 'TATACHEM',
    # Consumer
    'BATAINDIA', 'PAGEIND', 'RELAXO', 'GODREJIND',
    # Infra / Engineering
    'SUZLON', 'KPIL', 'KEC', 'GRINDWELL',
    # IT mid-small
    'MPHASIS', 'COFORGE', 'MASTEK',
    # Real estate small
    'SOBHA', 'BRIGADE', 'PHOENIXLTD',
]

CACHE_DIR = Path('data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_stock(kite, symbol: str) -> pd.DataFrame | None:
    """Download via Kite if not cached, or load from cache."""
    cache_file = CACHE_DIR / f"{symbol}_NS_1d.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    try:
        df = kite.fetch_ohlcv(f"{symbol}.NS", period='3y', interval='day')
        if df is None or len(df) < 200:
            return None
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.to_parquet(cache_file)
        print(f"  ✅ {symbol}: {len(df)} bars")
        return df
    except Exception as e:
        print(f"  ⚠️  {symbol}: {str(e)[:60]}")
        return None

def compute_momentum_ic(stocks_dict: dict, nifty_close: pd.Series, label: str) -> dict:
    """
    Compute cross-sectional IC of a simple 20-day relative momentum signal
    vs 10-day forward return. This is the SAME logic as our ranker's core signal.
    """
    test_dates = pd.date_range('2023-01-01', '2024-12-31', freq='ME')
    ic_vals = []

    for dt in test_dates:
        scores = {}
        fwd_rets = {}

        for sym, df in stocks_dict.items():
            try:
                hist = df[df.index <= dt]
                if len(hist) < 40:
                    continue

                # Signal: 20-day relative strength vs NIFTY
                nifty_slice = nifty_close.reindex(hist.index, method='ffill')
                stock_ret = hist['close'].pct_change(20).iloc[-1]
                nifty_ret = nifty_slice.pct_change(20).iloc[-1]
                scores[sym] = stock_ret - nifty_ret  # cross-sectional: relative to index

                # Forward return: 10 days after date
                future = df[df.index > dt]['close'].iloc[:10]
                if len(future) == 10:
                    fwd_rets[sym] = future.iloc[-1] / hist['close'].iloc[-1] - 1
            except Exception:
                continue

        common = [s for s in scores if s in fwd_rets]
        if len(common) < 6:
            continue

        ic, _ = spearmanr([scores[s] for s in common], [fwd_rets[s] for s in common])
        if not np.isnan(ic):
            ic_vals.append(ic)

    if not ic_vals:
        return {'mean_ic': 0, 'icir': 0, 'n_months': 0, 'pct_positive': 0}

    return {
        'mean_ic':     round(float(np.mean(ic_vals)), 4),
        'icir':        round(float(np.mean(ic_vals) / (np.std(ic_vals) + 1e-9)), 3),
        'n_months':    len(ic_vals),
        'pct_positive': round(sum(1 for x in ic_vals if x > 0) / len(ic_vals) * 100, 1),
    }

def compute_liquidity_stats(df: pd.DataFrame, position_size: float = 150000) -> dict:
    """Compute ATR, liquidity, and position-fillability."""
    if len(df) < 30:
        return {}
    recent = df.tail(60)
    atr14 = (recent['high'] - recent['low']).rolling(14).mean().iloc[-1]
    avg_price = recent['close'].mean()
    atr_pct = atr14 / avg_price * 100

    # Daily traded value in ₹
    avg_volume = recent['volume'].mean()
    daily_value_cr = avg_price * avg_volume / 1e7

    # Can we fill ₹1.5L in 15 min (≈5% of daily volume)?
    fillable_at_open = (daily_value_cr * 1e7 * 0.05) >= position_size

    return {
        'atr_pct': round(atr_pct, 2),
        'daily_value_cr': round(daily_value_cr, 1),
        'fillable_150k': fillable_at_open,
        'avg_price': round(avg_price, 1),
    }


def main():
    print("=" * 70)
    print("MARK5 UNIVERSE IC: LargeCap(50) vs MidCap(100) vs SmallCap(Next150)")
    print("=" * 70)

    kite = KiteFeedAdapter({
        'api_key':      os.getenv('KITE_API_KEY'),
        'access_token': os.getenv('KITE_ACCESS_TOKEN'),
        'api_secret':   os.getenv('KITE_API_SECRET'),
    })
    kite.connect()

    # ── NIFTY index ───────────────────────────────────────────────────────────
    nifty_df = pd.read_parquet(CACHE_DIR / 'NIFTY50_1d.parquet')
    nifty_close = nifty_df['close'] if 'close' in nifty_df.columns else nifty_df['Close']

    # ── Cohort 1: NIFTY50 (from cache) ───────────────────────────────────────
    print("\n📊 [1/3] NIFTY50 — loading from cache...")
    largecap = {}
    for f in sorted(CACHE_DIR.glob('*_NS_1d.parquet')):
        sym = f.stem.replace('_NS_1d', '')
        df = pd.read_parquet(f)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        largecap[sym] = df
    print(f"  Loaded: {len(largecap)} stocks")

    # ── Cohort 2: Midcap 100 ─────────────────────────────────────────────────
    print(f"\n📥 [2/3] Midcap 100 — downloading {len(MIDCAP_100)} symbols...")
    midcap = {}
    mid_failed = []
    for sym in MIDCAP_100:
        if sym in largecap:   # already in cache from previous run
            midcap[sym] = largecap[sym]
            continue
        df = download_stock(kite, sym)
        if df is not None:
            midcap[sym] = df
        else:
            mid_failed.append(sym)
    print(f"  Loaded: {len(midcap)} | Failed/Skip: {len(mid_failed)}")
    if mid_failed:
        print(f"  Failed: {mid_failed}")

    # ── Cohort 3: SmallCap / Next 150 ────────────────────────────────────────
    print(f"\n📥 [3/3] SmallCap/Next150 — downloading {len(SMALLCAP_NEXT)} symbols...")
    smallcap = {}
    small_failed = []
    for sym in SMALLCAP_NEXT:
        if sym in largecap or sym in midcap:
            smallcap[sym] = largecap.get(sym) if largecap.get(sym) is not None else midcap.get(sym)
            continue
        df = download_stock(kite, sym)
        if df is not None:
            smallcap[sym] = df
        else:
            small_failed.append(sym)
    print(f"  Loaded: {len(smallcap)} | Failed/Skip: {len(small_failed)}")
    if small_failed:
        print(f"  Failed: {small_failed}")

    # ── Compute IC for each cohort ────────────────────────────────────────────
    print("\n🔬 Computing IC for each cohort (20-day relative momentum → 10-day fwd return)")
    print("   [Same signal as MARK5 ranker — higher IC = stronger predictive power]\n")

    large_ic = compute_momentum_ic(largecap, nifty_close, "LargeCap")
    mid_ic   = compute_momentum_ic(midcap,   nifty_close, "MidCap")
    small_ic = compute_momentum_ic(smallcap, nifty_close, "SmallCap")

    # ── IC Comparison Table ───────────────────────────────────────────────────
    print("┌──────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│  Metric              │  LargeCap50  │  MidCap100   │  SmallCap150 │")
    print("├──────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│  Stocks              │  {len(largecap):<12} │  {len(midcap):<12} │  {len(smallcap):<12} │")
    print(f"│  Mean IC             │  {large_ic['mean_ic']:<12.4f} │  {mid_ic['mean_ic']:<12.4f} │  {small_ic['mean_ic']:<12.4f} │")
    print(f"│  ICIR                │  {large_ic['icir']:<12.3f} │  {mid_ic['icir']:<12.3f} │  {small_ic['icir']:<12.3f} │")
    print(f"│  % Months IC > 0     │  {large_ic['pct_positive']:<12.1f} │  {mid_ic['pct_positive']:<12.1f} │  {small_ic['pct_positive']:<12.1f} │")
    print(f"│  Months tested       │  {large_ic['n_months']:<12} │  {mid_ic['n_months']:<12} │  {small_ic['n_months']:<12} │")
    print("└──────────────────────┴──────────────┴──────────────┴──────────────┘")

    # ── ATR Distribution ─────────────────────────────────────────────────────
    print("\n📐 ATR Distribution (higher ATR = more profit potential per 10-day swing):")
    print(f"  {'Universe':<14} {'Avg ATR%':>9}  {'Median ATR%':>12}  {'Liquid (✅)':>11}")
    print("  " + "─" * 52)

    for label, cohort in [("LargeCap50", largecap), ("MidCap100", midcap), ("SmallCap150", smallcap)]:
        atrs = []
        liquid_count = 0
        for df in cohort.values():
            s = compute_liquidity_stats(df)
            if s:
                atrs.append(s['atr_pct'])
                if s['fillable_150k']:
                    liquid_count += 1
        if atrs:
            print(f"  {label:<14} {np.mean(atrs):>9.2f}%  {np.median(atrs):>12.2f}%  {liquid_count:>11}/{len(cohort)}")

    # ── Illiquid smallcap filter ──────────────────────────────────────────────
    illiquid_small = []
    for sym, df in smallcap.items():
        s = compute_liquidity_stats(df)
        if s and not s['fillable_150k']:
            illiquid_small.append(sym)

    # ── Final Recommendation ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📋  VERDICT — OPTIMAL MARK5 TRADING UNIVERSE")
    print("=" * 70)

    ics = {'LargeCap50': large_ic, 'MidCap100': mid_ic, 'SmallCap150': small_ic}
    best = max(ics, key=lambda k: ics[k]['icir'])
    print(f"\n  Best ICIR: {best}  (ICIR={ics[best]['icir']:.3f})")
    print(f"\n  LargeCap IC:  {large_ic['mean_ic']:+.4f}  ICIR={large_ic['icir']:.3f}")
    print(f"  MidCap IC:    {mid_ic['mean_ic']:+.4f}  ICIR={mid_ic['icir']:.3f}  (+{mid_ic['mean_ic']-large_ic['mean_ic']:.4f} vs largecap)")
    print(f"  SmallCap IC:  {small_ic['mean_ic']:+.4f}  ICIR={small_ic['icir']:.3f}  (+{small_ic['mean_ic']-large_ic['mean_ic']:.4f} vs largecap)")

    print("\n  Recommendation:")
    if small_ic['icir'] > mid_ic['icir'] * 1.2:
        print("  🏆 SMALLCAP has strongest IC — expand to SmallCap/Next150")
        print("     → BUT apply strict liquidity filter (> ₹20cr/day volume)")
        rec_universe = "combined_mid_small"
    elif mid_ic['icir'] > large_ic['icir']:
        print("  🏆 MIDCAP universe is optimal for MARK5")
        print("     → Expand live universe to NIFTY50 + Midcap100")
        rec_universe = "combined_large_mid"
    else:
        print("  ⚠️  No clear winner — stay on current NIFTY50 universe")
        rec_universe = "largecap"

    # ── Final Ticker Sync and Recommendation ──────────────────────────────────
    liquid_mid   = [s for s in midcap if s not in largecap]
    liquid_small = [s for s in smallcap if s not in largecap and s not in midcap and s not in illiquid_small]
    
    # Calculate unique totals
    all_unique = set(largecap.keys()) | set(midcap.keys()) | set(smallcap.keys())
    # Exclude known illiquids from smallcap
    all_filtered = all_unique - set(illiquid_small)
    
    print(f"\n  Potential combined universe: {len(largecap)} large + {len(liquid_mid)} mid + {len(liquid_small)} small = {len(all_filtered)} stocks")
    print(f"  (Rule 34 min=30: {'✅ SATISFIED' if total >= 30 else '❌ FAIL'})")
    print()


if __name__ == '__main__':
    main()
