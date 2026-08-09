"""
Data hygiene: re-fetch EVERY cached single-name + key indices to a UNIFORM end date,
so the recent-window tail isn't partly frozen (audit pass-2 finding: only ~half the
cache reached the intended END). Split/div-adjusted (auto_adjust=True). Overwrites a
file only on a successful fetch (never destroys good data on a network blip).

  python3 scripts/refetch_all.py
"""
import json
import os, sys, time
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio.universe import discover_tickers

CACHE = os.path.join(_ROOT, "data", "cache")
START = os.environ.get("MARK5_REFETCH_START", "2015-01-01")
# A HARDCODED end date is why the cache silently rotted to 61 days stale: the
# constant stopped moving while the calendar did, and every later run refetched
# to the same frozen day. The uniform-end property that this script exists to
# guarantee is preserved by resolving "today" ONCE, here, and using it for every
# ticker in the run — not by freezing it in the source.
END = os.environ.get("MARK5_REFETCH_END") or str(pd.Timestamp.today().date())


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


def held_names():
    """Whatever the live book owns. Always fetched, whatever else is going on.

    This exists because of a specific hole. `discover_tickers()` reads the CACHE,
    so this script only ever refreshed names already in it — a holding that fell
    out could never come back, because nothing in the system fetched it again. On
    2026-08-09 that was 14 of the 20 stocks in the book, none of them in the
    pinned list either.

    Not a cosmetic gap: a name the cache cannot see cannot be ranked, so the next
    rebalance drops it from the target book, so it is SOLD. You must always be
    able to price what you own.
    """
    try:
        with open(os.path.join(_ROOT, "data", "paper", "paper_book.json")) as f:
            return set(json.load(f).get("positions", {}))
    except (OSError, ValueError):
        return set()


def pinned_names():
    try:
        with open(os.path.join(_ROOT, "config", "universe_tickers.json")) as f:
            return set(json.load(f)["tickers"])
    except (OSError, ValueError, KeyError):
        return set()


# How many of the market's most-traded names to keep cached. TOP_N_LIQUID is 300
# at selection time, so the cache must hold MORE than 300 for that screen to have
# anything to screen — with 203 cached it has never once been binding, and "top
# 300 by liquidity" has quietly meant "all of them". The margin also lets names
# near the boundary move in and out without falling off the edge of the world.
MARKET_TOP_N = 450


def market_names(top_n=MARKET_TOP_N):
    """The most-traded names currently listed, from the latest NSE bhavcopy.

    Without this the universe can only ever SHRINK. `discover_tickers()` reads the
    cache, so the set of investable names was whatever happened to be cached once,
    minus anything that later fell out. New listings — and NSE lists dozens every
    six months — could never enter, however large or liquid they became. A book
    left alone for a year would be choosing from a slowly decaying survivor set
    while believing it was ranking the market.

    Bhavcopy is the honest source: a snapshot of what actually traded that day, so
    a new listing simply appears and a delisted one simply stops. Names are taken
    by turnover, because an illiquid microcap that we could never buy is noise in
    the cross-section, not opportunity.

    A brand-new listing still will not be SELECTABLE for about a year — 252 days
    of history is required to score it, and that is deliberate. What matters is
    that it is now in the cache when that day comes, instead of invisible forever.
    """
    import glob
    raw = sorted(glob.glob(os.path.join(_ROOT, "data", "bhavcopy", "raw", "*.parquet")))
    if not raw:
        print("  no bhavcopy — cannot look for new listings")
        return set()
    df = pd.read_parquet(raw[-1])
    df = df[pd.to_numeric(df["turnover"], errors="coerce").fillna(0) > 0]
    names = set(df.nlargest(top_n, "turnover")["symbol"].astype(str).str.upper())
    print(f"  bhavcopy {os.path.basename(raw[-1])[:10]}: top {top_n} of "
          f"{len(df)} traded names")
    return names


def main():
    # Union of four sources, so the universe can grow as well as persist:
    #   cached   — what we already track
    #   pinned   — what a cacheless CI runner is told to fetch
    #   held     — what we own, which must always be priceable
    #   market   — what is actually trading now, so new listings can ever enter
    # Nothing here ever removes a name; delisting removes it, by ceasing to print.
    from core.portfolio.universe import STRUCTURAL_EXCLUDE
    tickers = sorted((set(discover_tickers()) | pinned_names() | held_names()
                      | market_names()) - STRUCTURAL_EXCLUDE)
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
        # 0.15s drew a ~16% failure rate from Yahoo — "possibly delisted" for names
        # like AMBUJACEM and ATUL that are plainly listed. That is throttling, not
        # delisting, and it is indistinguishable from the real thing downstream: a
        # throttled name is simply absent, and an absent name gets SOLD at the next
        # rebalance. Twice a year, a slower fetch that finishes is worth far more
        # than a fast one that lies.
        time.sleep(0.4)

    # Second pass over the failures, slower. Most are throttling, and throttling
    # clears. Without this an unattended January run would hand a knowingly
    # incomplete cache to the one rebalance that matters.
    if failed:
        print(f"\n  retrying {len(failed)} failures more slowly ...", flush=True)
        again = []
        for t in failed:
            try:
                nd = normalize(yf.download(f"{t}.NS", start=START, end=END,
                                           auto_adjust=True, progress=False,
                                           threads=False))
                if nd is None:
                    again.append(t)
                else:
                    nd.reset_index().to_parquet(os.path.join(CACHE, f"{t}_daily.parquet"))
                    ok += 1; fail -= 1
            except Exception:                                # noqa: BLE001
                again.append(t)
            time.sleep(2.0)
        failed = again
        print(f"  after retry: ok={ok} fail={fail}", flush=True)
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
    # refresh the Nifty proxy used as benchmark.
    # GUARD: a partial yf response once overwrote the benchmark with 2007-2017-only
    # data, silently corrupting every vs-Nifty figure. Never save unless the
    # download is long AND reaches the requested END year.
    try:
        nd = normalize(yf.download("^NSEI", start=START, end=END, auto_adjust=True,
                                   progress=False, threads=False))
        if nd is not None and len(nd) > 4000 and str(nd.index.max())[:4] >= END[:4]:
            nd.reset_index().to_parquet(os.path.join(CACHE, "sector_NSEI.parquet"))
            print("  refreshed sector_NSEI (Nifty50 benchmark)")
        else:
            print("  WARN: Nifty download partial/stale — benchmark NOT overwritten")
    except Exception:
        print("  WARN: Nifty benchmark refresh failed")
    print(f"\nDONE: ok={ok} fail={fail} of {len(tickers)}")
    if failed:
        print(f"  failed: {failed}")

    # Re-pin, so the cacheless CI runner is told to fetch what actually exists.
    # The pinned list had drifted to missing 14 of the 20 stocks in the live book,
    # which is how the January rebalance would have arrived at a universe that
    # could not see most of the portfolio. A list that is never regenerated stops
    # describing reality the moment the cache moves.
    #
    # Only on a mostly-successful run: re-pinning after a rate-limited or offline
    # run would BAKE IN the truncation and make the next cacheless rebuild worse.
    if ok and ok / max(len(tickers), 1) >= 0.9:
        names = sorted(set(discover_tickers()) | held_names())
        p = os.path.join(_ROOT, "config", "universe_tickers.json")
        json.dump({"description": "Version-pinned universe for rebuilding data/cache "
                                  "from scratch (scripts/refetch_all.py). Regenerated "
                                  "automatically after a successful full fetch.",
                   "pinned": END, "count": len(names), "tickers": names},
                  open(p, "w"), indent=1)
        print(f"  re-pinned {p}: {len(names)} names")
    else:
        print(f"  NOT re-pinning: only {ok}/{len(tickers)} succeeded — a truncated "
              f"list would make the next cacheless rebuild worse, not better")


if __name__ == "__main__":
    main()
