"""
Fetch NSE security-wise delivery data — the one free signal that is NOT price.
=============================================================================
Every factor this project has ever tested (momentum, trend, low-vol, stability,
candlesticks, foundation models) is a FUNCTION OF THE SAME OHLCV SERIES, so they
all live in the same span. That is the structural reason K1-K9 all failed, and
why the factor regression finds the equity sleeve's alpha insignificant once
market/size/momentum/low-vol are controlled for. New alpha needs new information.

NSE publishes `sec_bhavdata_full_DDMMYYYY.csv` daily, free, containing:
  DELIV_QTY / DELIV_PER  what fraction of the day's traded volume was actually
                         DELIVERED to a demat account rather than squared off
                         intraday. Price cannot express this. High delivery =
                         someone took real ownership; low delivery = churn.
  NO_OF_TRADES           with turnover, gives average trade size — a crude but
                         real institutional-participation proxy.

Point-in-time safe: published same day, never restated, no disclosure lag. This
is the property the quarterly shareholding data (K7) also had and the reason it
was worth testing even though it failed.

HONEST LIMIT, stated before any result: the archive starts ~2019-10, so this
gives ~6.8 years against the backtest's 10.6. It misses the 2018 NBFC crisis and
begins months before COVID. Any conclusion drawn from it is bounded by that.

Resumable and polite: already-fetched days are skipped, non-trading days are
remembered so re-runs never retry them, modest concurrency.

  python3 scripts/fetch_delivery.py                 # from 2019-10-01 to today
  python3 scripts/fetch_delivery.py --start 2024-01-01
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "data", "delivery", "raw")
NONTRADING = os.path.join(_ROOT, "data", "delivery", "_nontrading.json")
URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
       "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
KEEP = ["symbol", "deliv_qty", "deliv_per", "no_of_trades", "ttl_trd_qnty", "turnover_lacs"]


def fetch_day(day: pd.Timestamp, retries: int = 3):
    """-> ('ok', df) | ('none', None) for a non-trading day | ('err', msg)."""
    url = URL.format(d=day.strftime("%d%m%Y"))
    for a in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=40)
            txt = r.read().decode("utf-8", "replace")
            df = pd.read_csv(io.StringIO(txt))
            df.columns = [c.strip().lower() for c in df.columns]
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).str.strip()
            if "series" not in df.columns:
                return "err", f"unexpected columns {list(df.columns)[:6]}"
            df = df[df["series"] == "EQ"]              # equities only, not GS/bonds
            if df.empty:
                return "none", None
            for c in ("deliv_qty", "deliv_per", "no_of_trades",
                      "ttl_trd_qnty", "turnover_lacs"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return "ok", df[[c for c in KEEP if c in df.columns]].reset_index(drop=True)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "none", None                    # holiday / weekend
            time.sleep(1.5 * (a + 1))
        except Exception:
            time.sleep(1.5 * (a + 1))
    return "err", "retries exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-10-01")
    ap.add_argument("--end", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    skip = set(json.load(open(NONTRADING))) if os.path.exists(NONTRADING) else set()
    days = [d for d in pd.bdate_range(args.start, args.end)
            if not os.path.exists(os.path.join(OUT, f"{d.date()}.parquet"))
            and str(d.date()) not in skip]
    print(f"  {len(days)} days to fetch  ({args.start} -> {args.end}, "
          f"{len(skip)} known non-trading days skipped)", flush=True)
    if not days:
        print("  nothing to do")
        return

    ok = err = none = 0
    errs = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_day, d): d for d in days}
        for i, f in enumerate(cf.as_completed(futs), 1):
            d = futs[f]
            status, payload = f.result()
            if status == "ok":
                payload.to_parquet(os.path.join(OUT, f"{d.date()}.parquet"), index=False)
                ok += 1
            elif status == "none":
                skip.add(str(d.date()))
                none += 1
            else:
                err += 1
                errs.append((str(d.date()), payload))
            if i % 100 == 0 or i == len(days):
                print(f"    {i}/{len(days)}  ok={ok} holiday={none} err={err}", flush=True)
    json.dump(sorted(skip), open(NONTRADING, "w"))
    print(f"\n  done: {ok} days written, {none} non-trading, {err} errors")
    if errs:
        # failures are reported, never swallowed — a silently short archive
        # would quietly bias every statistic computed from it
        print(f"  FAILED DAYS (re-run to retry): {[e[0] for e in errs[:10]]}"
              f"{' ...' if len(errs) > 10 else ''}")
    files = sorted(os.listdir(OUT))
    if files:
        print(f"  archive: {len(files)} days, {files[0][:10]} -> {files[-1][:10]}")


if __name__ == "__main__":
    main()
