"""Run the net-of-tax factor-tilt optimizer on the fetched NSE MIDCAP universe.
Reuses build/run/simulate from factor_tilt_optimize (same engine, same tax model).
Survivorship caveat: current midcap list on historical data inflates absolute
levels; the tilt-EXCESS over same-universe buy-and-hold is the robust quantity."""
import os, sys
import pandas as pd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from factor_tilt_optimize import run, CACHE
from fetch_midcaps import MIDCAPS

# nifty benchmark
nifty = None
for p in [os.path.join(CACHE, "nse", "NIFTY50_20150101_20260521.parquet"),
          os.path.join(CACHE, "sector_NSEI.parquet")]:
    if os.path.exists(p):
        nifty = pd.read_parquet(p); nifty.columns = [c.lower() for c in nifty.columns]
        if "date" in nifty.columns:
            nifty.index = pd.to_datetime(nifty["date"])
        nifty.index = pd.to_datetime(nifty.index); nifty = nifty.sort_index()
        break

# keep only midcaps that actually cached with usable history
uni = [t for t in MIDCAPS if os.path.exists(os.path.join(CACHE, f"{t}_daily.parquet"))]
print(f"Midcap universe available: {len(uni)} names")

run("2016-01-01", "2021-12-31", uni, nifty, "MIDCAP — HONEST OOS 2016-2021 (UNTOUCHED)")
run("2022-01-01", "2026-05-21", uni, nifty, "MIDCAP — TUNED 2022-2026 (cross-check)")
