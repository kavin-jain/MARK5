"""
Data hygiene: re-fetch EVERY cached single-name + key indices to a UNIFORM end date,
so the recent-window tail isn't partly frozen (audit pass-2 finding: only ~half the
cache reached the intended END). Split/div-adjusted (auto_adjust=True). Overwrites a
file only on a successful fetch (never destroys good data on a network blip).

  python3 scripts/refetch_all.py
"""
import os, sys, time
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio.universe import discover_tickers

CACHE = os.path.join(_ROOT, "data", "cache")
START = "2015-01-01"
END = "2026-06-10"            # uniform end


def normalize(df):
    if df is None or len(df) < 200:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].dropna(how="all")
    df.index.name = "date"
    return df if len(df) >= 200 else None


def main():
    tickers = discover_tickers()
    print(f"Re-fetching {len(tickers)} names to uniform END={END} ...", flush=True)
    ok = fail = 0
    failed = []
    for i, t in enumerate(tickers):
        try:
            df = yf.download(f"{t}.NS", start=START, end=END,
                             auto_adjust=True, progress=False, threads=False)
            nd = normalize(df)
            if nd is None:
                fail += 1; failed.append(t); continue
            nd.reset_index().to_parquet(os.path.join(CACHE, f"{t}_daily.parquet"))
            ok += 1
        except Exception:
            fail += 1; failed.append(t)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tickers)}  ok={ok} fail={fail}", flush=True)
        time.sleep(0.15)
    # refresh the multi-asset sleeves (excluded from discover_tickers as ETFs)
    for etf in ("GOLDBEES", "MON100", "LIQUIDBEES"):
        try:
            nd = normalize(yf.download(f"{etf}.NS", start=START, end=END,
                                       auto_adjust=True, progress=False, threads=False))
            if nd is not None:
                nd.reset_index().to_parquet(os.path.join(CACHE, f"{etf}_daily.parquet"))
                print(f"  refreshed {etf} (multi-asset sleeve)")
        except Exception:
            print(f"  WARN: {etf} refresh failed")
    # refresh the Nifty proxy used as benchmark
    try:
        nd = normalize(yf.download("^NSEI", start=START, end=END, auto_adjust=True,
                                   progress=False, threads=False))
        if nd is not None:
            nd.reset_index().to_parquet(os.path.join(CACHE, "sector_NSEI.parquet"))
            print("  refreshed sector_NSEI (Nifty50 benchmark)")
    except Exception:
        print("  WARN: Nifty benchmark refresh failed")
    print(f"\nDONE: ok={ok} fail={fail} of {len(tickers)}")
    if failed:
        print(f"  failed: {failed}")


if __name__ == "__main__":
    main()
