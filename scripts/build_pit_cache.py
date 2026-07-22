"""
Build a TRUE point-in-time, survivorship-free price cache from raw bhavcopy.
===========================================================================
Takes the unadjusted daily snapshots from scripts/fetch_bhavcopy.py, joins NSE
corporate actions, applies standard backward total-return adjustment, and writes
per-symbol parquet in exactly the layout core/portfolio/universe.py already reads.

Point the engine at it with:
    MARK5_CACHE=data/pit_cache python3 scripts/run_mark6.py

WHY THE ADJUSTMENT IS NOT OPTIONAL: bhavcopy prices are raw. The widely-repeated
claim that PREVCLOSE is split-adjusted on the ex-date is FALSE — verified on the
IRCTC 1:5 split, where the 28-OCT-2021 row reads CLOSE=913.5 against PREVCLOSE=
4130.15. Chaining those raw closes injects a fake -78% day straight into a 12-1
momentum factor. Splits and bonuses MUST be joined; dividends are also applied so
the series is total-return and therefore comparable to the yfinance auto_adjust
cache the rest of the project uses.

  python3 scripts/build_pit_cache.py --min-turnover-cr 5
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_ROOT, "data", "bhavcopy", "raw")
CA_CACHE = os.path.join(_ROOT, "data", "bhavcopy", "corporate_actions.json")
OUT = os.path.join(_ROOT, "data", "pit_cache")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
      "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}

RE_SPLIT = re.compile(r"from\s+rs?\.?\s*([\d.]+)\s*/?-?\s*per\s+share\s+to\s+"
                      r"(?:rs?|re)\.?\s*([\d.]+)", re.I)
RE_BONUS = re.compile(r"bonus\s+(\d+)\s*:\s*(\d+)", re.I)
RE_DIV = re.compile(r"dividend.*?rs\.?\s*([\d.]+)", re.I | re.S)


def fetch_corporate_actions(start="2016-01-01", end="2026-06-09") -> list:
    """NSE CA API, monthly chunks. Cached — the API needs warm cookies and is slow."""
    if os.path.exists(CA_CACHE):
        return json.load(open(CA_CACHE))
    import time

    def warm():
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
        op.addheaders = list(UA.items())
        op.open("https://www.nseindia.com/companies-listing/corporate-filings-actions",
                timeout=30).read()
        return op

    op, out, failed = warm(), [], []
    months = pd.date_range(start, end, freq="MS")
    for i, m0 in enumerate(months, 1):
        m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        url = ("https://www.nseindia.com/api/corporates-corporateActions?index=equities"
               f"&from_date={m0:%d-%m-%Y}&to_date={m1:%d-%m-%Y}")
        # a transient DNS/cookie blip must not silently drop a month of corporate
        # actions — a missed split injects a fake >50% move into the momentum factor
        for attempt in range(4):
            try:
                out += json.loads(op.open(url, timeout=45).read())
                break
            except Exception as e:
                if attempt == 3:
                    failed.append(f"{m0:%Y-%m}")
                    print(f"  WARN {m0:%Y-%m} unrecoverable: {type(e).__name__}", flush=True)
                else:
                    time.sleep(3 * (attempt + 1))
                    try:
                        op = warm()
                    except Exception:
                        pass
        if i % 24 == 0:
            print(f"  corporate actions {i}/{len(months)} months, {len(out)} records", flush=True)
    if failed:
        sys.exit(f"ABORT: {len(failed)} months of corporate actions could not be fetched "
                 f"({failed[:6]}). Adjusting prices with missing splits would inject fake "
                 f"returns into the factors — re-run when the network is stable.")
    json.dump(out, open(CA_CACHE, "w"))
    return out


def parse_events(records: list) -> dict:
    """-> {symbol: [(ex_date, kind, value)]}. value semantics:
       split/bonus -> ratio the price divides by on ex-date; dividend -> rupees/share."""
    ev, unparsed = {}, 0
    for r in records:
        sym, subj, ex = r.get("symbol"), str(r.get("subject") or ""), r.get("exDate")
        if not sym or not ex:
            continue
        try:
            d = pd.Timestamp(pd.to_datetime(ex, dayfirst=True))
        except Exception:
            continue
        s = subj.lower()
        if "split" in s or "sub-division" in s or "consolidation" in s:
            m = RE_SPLIT.search(subj)
            if m:
                frm, to = float(m.group(1)), float(m.group(2))
                if to > 0 and frm != to:
                    ev.setdefault(sym, []).append((d, "split", frm / to))
                    continue
            unparsed += 1
        elif "bonus" in s:
            m = RE_BONUS.search(subj)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                if b > 0:
                    ev.setdefault(sym, []).append((d, "split", (a + b) / b))
                    continue
            unparsed += 1
        elif "dividend" in s and "interest" not in s:
            m = RE_DIV.search(subj)
            if m:
                ev.setdefault(sym, []).append((d, "div", float(m.group(1))))
    if unparsed:
        print(f"  note: {unparsed} split/bonus subjects did not parse (left unadjusted)")
    return ev


def adjust(close: pd.Series, events: list) -> pd.Series:
    """Standard backward total-return adjustment: every price BEFORE an ex-date is
    scaled so the ex-date gap disappears. factor<1 shrinks history, as it should."""
    factors = pd.Series(1.0, index=close.index)
    for d, kind, v in events:
        idx = close.index.searchsorted(d)
        if idx <= 0 or idx >= len(close):
            continue
        if kind == "split":
            factors.iloc[idx] *= 1.0 / v
        else:                                   # dividend
            prev = close.iloc[idx - 1]
            if prev > 0 and 0 < v < prev:
                factors.iloc[idx] *= (1.0 - v / prev)
    # cumulative product of all FUTURE factors, applied to each past price
    cum = factors[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close * cum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-turnover-cr", type=float, default=5.0,
                    help="keep symbols whose PEAK 126d median daily turnover ever "
                         "reached this many crore (keeps the cache tractable)")
    ap.add_argument("--min-days", type=int, default=300)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RAW, "*.parquet")))
    if len(files) < 500:
        sys.exit(f"Only {len(files)} raw bhavcopy days in {RAW} — run "
                 f"scripts/fetch_bhavcopy.py first.")
    print(f"Loading {len(files)} bhavcopy days...", flush=True)
    frames = []
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        df["date"] = pd.Timestamp(os.path.basename(f)[:10])
        frames.append(df)
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    eod = pd.concat(frames, ignore_index=True)
    print(f"  {len(eod):,} symbol-days, {eod.symbol.nunique():,} distinct symbols")

    print("Fetching corporate actions (cached after first run)...", flush=True)
    events = parse_events(fetch_corporate_actions())
    print(f"  parsed events for {len(events):,} symbols")

    close = eod.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    vol = eod.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last")
    turn = eod.pivot_table(index="date", columns="symbol", values="turnover", aggfunc="last")
    if turn.isna().all().all():
        turn = close * vol
    med = turn.rolling(126, min_periods=40).median().max() / 1e7      # peak, in crore
    keep = [s for s in close.columns
            if close[s].notna().sum() >= args.min_days and med.get(s, 0) >= args.min_turnover_cr]
    print(f"  {len(keep):,} symbols pass history + liquidity")

    # ── structural ETF/fund exclusion by ISIN prefix ──────────────────────────
    # Indian ISINs: INE = operating company equity, INF = mutual fund / ETF units.
    # Name heuristics (endswith BEES/ETF) miss SETFGOLD, LICMFGOLD, AXISGOLD,
    # GROWWGOLD... and an ETF in an equity book is not a cosmetic problem: the
    # research log records LIQUIDBEES (~cash, lowest vol) being inverse-vol
    # OVERWEIGHTED to the top of the book. Prefix is structural, so use it.
    isin_of = eod.dropna(subset=["isin"]).groupby("symbol")["isin"].agg(
        lambda s: s.astype(str).mode().iat[0] if len(s) else "")
    funds = [s for s in keep if str(isin_of.get(s, "")).startswith("INF")]
    keep = [s for s in keep if s not in set(funds)]
    print(f"  excluded {len(funds):,} ETF/fund units by ISIN prefix INF "
          f"(e.g. {sorted(funds)[:6]})")

    # ── residual-jump guard, tested on the ADJUSTED series ────────────────────
    # The question is not "was there a corporate action nearby" but "did the
    # adjustment actually work". Testing raw prices and excusing moves near a known
    # event lets PARTIALLY adjusted names through — BAJFINANCE had a bonus AND a
    # split on one ex-date, so catching either one still leaves a fake -49% day.
    # Demergers and capital reductions are absent from the CA feed entirely. A
    # fabricated -90% return is strictly worse than omitting the name.
    suspect = []
    for s in keep:
        c = close[s].dropna()
        if len(c) < 2:
            continue
        r = (adjust(c, events[s]) if s in events else c).pct_change()
        if bool(((r < -0.45) | (r > 1.5)).any()):
            suspect.append(s)
    keep = [s for s in keep if s not in set(suspect)]
    print(f"  excluded {len(suspect):,} symbols still showing >45% single-day moves AFTER "
          f"adjustment (demerger / partial-CA): {sorted(suspect)[:6]}")
    print(f"  keeping {len(keep):,} symbols")

    os.makedirs(OUT, exist_ok=True)
    ohlc = {c: eod.pivot_table(index="date", columns="symbol", values=c, aggfunc="last")
            for c in ("open", "high", "low") if c in eod.columns}
    written = adjusted = 0
    for s in keep:
        c = close[s].dropna()
        if len(c) < args.min_days:
            continue
        ev = events.get(s, [])
        ca = adjust(c, ev) if ev else c
        adjusted += bool(ev)
        ratio = (ca / c).reindex(c.index)
        out = pd.DataFrame({"date": c.index, "close": ca.values,
                            "volume": vol[s].reindex(c.index).values})
        for name, panel in ohlc.items():
            out[name] = (panel[s].reindex(c.index) * ratio).values
        out.to_parquet(os.path.join(OUT, f"{s}_daily.parquet"), index=False)
        written += 1

    alive = close.iloc[-20:].notna().any()
    dead = [s for s in keep if not alive.get(s, False)]
    print(f"\nDONE: {written:,} symbols written to data/pit_cache/ "
          f"({adjusted:,} had corporate actions applied)")
    print(f"  SURVIVORSHIP: {len(dead):,} of them ({100*len(dead)/max(1,written):.1f}%) stopped "
          f"trading before the end — these are exactly the names data/cache/ is blind to.")
    print(f"  e.g. {sorted(dead)[:10]}")
    print(f"\n  USE IT:  MARK5_CACHE=data/pit_cache .venv/bin/python scripts/run_mark6.py")


if __name__ == "__main__":
    main()
