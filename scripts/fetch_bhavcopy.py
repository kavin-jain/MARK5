"""
Fetch NSE daily bhavcopy — the raw material for a TRUE point-in-time universe.
=============================================================================
Why this exists: the yfinance cache in data/cache/ contains only names that are
STILL LISTED today. Measured against bhavcopy, 385 of the 1,467 liquid NSE names
alive in Jan-2016 are gone by 2026 (26.2%); among the top 500 by turnover, 99 died
(19.8%). Backtests on survivors alone overstate returns by ~2-5pp/yr. Bhavcopy is
the free fix: it is a daily snapshot of what actually traded THAT DAY, so a name
that delists simply stops appearing — no separate delisted list needed.

Two endpoint generations (NSE switched to UDiFF on 2024-07-08); both handled.
Resumable: already-fetched days are skipped, known holidays are remembered.

  python3 scripts/fetch_bhavcopy.py --start 2016-01-01 --end 2026-06-09

Output: data/bhavcopy/raw/YYYY-MM-DD.parquet (symbol, isin, ohlc, volume, turnover)
NOTE: these prices are UNADJUSTED. Run scripts/build_pit_cache.py next — it joins
corporate actions and writes an adjusted, engine-readable cache.
"""
import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "data", "bhavcopy", "raw")
HOLIDAYS = os.path.join(_ROOT, "data", "bhavcopy", "_nontrading.json")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
UDIFF_FROM = pd.Timestamp("2024-07-08")
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# canonical output columns, mapped from each generation's own schema
LEGACY = {"SYMBOL": "symbol", "ISIN": "isin", "OPEN": "open", "HIGH": "high",
          "LOW": "low", "CLOSE": "close", "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover"}
UDIFF = {"TckrSymb": "symbol", "ISIN": "isin", "OpnPric": "open", "HghPric": "high",
         "LwPric": "low", "ClsPric": "close", "TtlTradgVol": "volume", "TtlTrfVal": "turnover"}


def url_for(d: pd.Timestamp) -> str:
    if d >= UDIFF_FROM:
        return ("https://nsearchives.nseindia.com/content/cm/"
                f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
    return ("https://nsearchives.nseindia.com/content/historical/EQUITIES/"
            f"{d:%Y}/{MONTHS[d.month - 1]}/cm{d:%d}{MONTHS[d.month - 1]}{d:%Y}bhav.csv.zip")


def parse(blob: bytes, d: pd.Timestamp) -> pd.DataFrame | None:
    z = zipfile.ZipFile(io.BytesIO(blob))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if d >= UDIFF_FROM:
        df = df[df.get("FinInstrmTp", "STK").astype(str).str.upper().isin(["STK", "EQ"])]
        series_col, mapping = "SctySrs", UDIFF
    else:
        df.columns = [c.upper() for c in df.columns]
        series_col, mapping = "SERIES", LEGACY
    if series_col not in df.columns:
        return None
    df = df[df[series_col].astype(str).str.strip().isin(["EQ", "BE"])]
    have = {k: v for k, v in mapping.items() if k in df.columns}
    df = df[list(have)].rename(columns=have)
    if "symbol" not in df.columns or "close" not in df.columns or df.empty:
        return None
    df["symbol"] = df["symbol"].astype(str).str.strip()
    for c in ("open", "high", "low", "close", "volume", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def fetch_day(d: pd.Timestamp) -> str:
    """-> 'ok' | 'skip' | 'holiday' | 'fail'. Writes one parquet per trading day."""
    path = os.path.join(OUT, f"{d:%Y-%m-%d}.parquet")
    if os.path.exists(path):
        return "skip"
    try:
        req = urllib.request.Request(url_for(d), headers=UA)
        with urllib.request.urlopen(req, timeout=45) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        return "holiday" if e.code in (403, 404) else "fail"
    except Exception:
        return "fail"
    try:
        df = parse(blob, d)
    except Exception:
        return "fail"
    if df is None or df.empty:
        return "holiday"
    df.to_parquet(path, index=False)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-06-09")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    known = set(json.load(open(HOLIDAYS))) if os.path.exists(HOLIDAYS) else set()
    days = [d for d in pd.bdate_range(args.start, args.end) if f"{d:%Y-%m-%d}" not in known]
    print(f"Fetching {len(days)} candidate trading days with {args.workers} workers "
          f"({len(known)} known non-trading days skipped)...", flush=True)

    tally = {"ok": 0, "skip": 0, "holiday": 0, "fail": 0}
    new_holidays, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (d, res) in enumerate(zip(days, ex.map(fetch_day, days)), 1):
            tally[res] += 1
            if res == "holiday":
                new_holidays.append(f"{d:%Y-%m-%d}")
            elif res == "fail":
                failures.append(f"{d:%Y-%m-%d}")
            if i % 250 == 0:
                print(f"  {i}/{len(days)}  {tally}", flush=True)

    json.dump(sorted(known | set(new_holidays)), open(HOLIDAYS, "w"))
    print(f"\nDONE: {tally}")
    if failures:
        print(f"  {len(failures)} transient failures (re-run to retry): {failures[:8]}")
    files = len([f for f in os.listdir(OUT) if f.endswith(".parquet")])
    print(f"  {files} trading days cached in data/bhavcopy/raw/")
    print("  NEXT: python3 scripts/build_pit_cache.py   (joins corporate actions, adjusts prices)")


if __name__ == "__main__":
    main()
