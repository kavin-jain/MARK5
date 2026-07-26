"""
MARK6 — External (shareholding-derived) factors
===============================================
Causal, point-in-time factors built from the free NSE shareholding XBRL archive
(`data/cache/shareholding_nse/`, see scripts/fetch_shareholding_nse.py). Each value
is indexed by the REAL public-disclosure date (NSE broadcastDate), so an as-of
lookup at a rebalance date uses only what was public then — zero look-ahead.

Factors (sign-normalised so higher = more attractive), per RESEARCH_LOG frontiers:
  - promoter_chg   (F6): QoQ change in promoter holding %. Weak but the only
                          ownership signal with a consistent +IC (~+0.04). Skin-in-
                          the-game *increasing*.
  - promoter_level (F3): promoter holding % level — governance/quality proxy
                          (founder skin-in-the-game). Higher = better.
  - inst_chg            : QoQ change in institutional (FII+DII) holding. Included for
                          completeness; I1 showed IC≈0 (expected to add nothing).

These are OPTIONAL inputs to the Backtester; the baseline price-only MARK6 is
unchanged when they are not supplied.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHP = os.path.join(_ROOT, "data", "cache", "shareholding_nse")

EXTERNAL_FACTOR_NAMES = ("promoter_chg", "promoter_level", "inst_chg")


def load_external_factors(src: str = SHP) -> dict[str, pd.DataFrame]:
    """ticker -> causal DataFrame(index=disclosure date, cols=EXTERNAL_FACTOR_NAMES).

    Only quarters with valid institutional data are kept (the parser drops
    parse-failures as None). Returns {} if the cache is absent.
    """
    out: dict[str, pd.DataFrame] = {}
    if not os.path.isdir(src):
        return out
    for f in glob.glob(os.path.join(src, "*.json")):
        t = os.path.basename(f).replace(".json", "")
        d = json.load(open(f))
        qs = d.get("quarters", [])
        disc = d.get("disclosure")
        if len(qs) < 5 or not disc:
            continue
        idx = pd.to_datetime(disc)
        promo = pd.Series(d.get("Promoters", []), index=idx, dtype="float64")
        inst = pd.Series(d.get("Institutions", []), index=idx, dtype="float64")
        df = pd.DataFrame({
            "promoter_level": promo,
            "promoter_chg": promo.diff(),
            "inst_chg": inst.diff(),
        }).sort_index()
        df = df[~df.index.duplicated(keep="last")].dropna(how="all")
        if len(df) >= 4:
            out[t] = df
    return out


# ── delivery-derived factors (v7.7, PROVISIONAL — see RESEARCH_LOG 4l) ────────
DELIVERY_RAW = os.path.join(_ROOT, "data", "delivery", "raw")


def load_delivery_factors(src: str = DELIVERY_RAW, universe=None) -> dict:
    """ticker -> causal DataFrame(index=date, cols=['deliv_per_z','deliv_chg']).

    Built from the free NSE `sec_bhavdata_full` archive. DELIV_PER is the share of
    a day's traded volume that was actually DELIVERED to a demat account rather
    than squared off intraday — genuinely orthogonal to price, which is the whole
    reason it is here (every price-derived factor lies in the same span).

      deliv_per_z  delivery % vs its OWN 126d history. Own-history rather than
                   cross-sectional because the permanent level differs by name and
                   sector; what carries information is the CHANGE in conviction.
      deliv_chg    21d mean delivery % minus the 126d mean. Rising conviction.
                   This is the one with evidence — deliv_per_z tested WORSE than
                   baseline and is deliberately NOT in the deployed blend.

    Causality: delivery for date t is published after that day's close, so a value
    indexed at t is knowable at t's close; the engine then executes at t+1
    (exec_lag=1). No look-ahead.

    Returns {} if the archive is absent, so every caller degrades to the
    price-only book rather than crashing — the archive is 130MB and gitignored.
    """
    import glob
    if not os.path.isdir(src):
        return {}
    rows = {}
    keep = set(universe) if universe else None
    for f in sorted(glob.glob(os.path.join(src, "*.parquet"))):
        d = pd.Timestamp(os.path.basename(f)[:10])
        df = pd.read_parquet(f)
        df = df[df["symbol"].notna()].drop_duplicates("symbol").set_index("symbol")
        rows[d] = df["deliv_per"]
    if not rows:
        return {}
    dp = pd.DataFrame(rows).T.sort_index()
    if keep:
        dp = dp[[c for c in dp.columns if c in keep]]
    mean126 = dp.rolling(126, min_periods=63).mean()
    std126 = dp.rolling(126, min_periods=63).std()
    z = (dp - mean126) / std126.replace(0, pd.NA)
    chg = dp.rolling(21, min_periods=10).mean() - mean126
    out = {}
    for t in dp.columns:
        df = pd.DataFrame({"deliv_per_z": z[t], "deliv_chg": chg[t]}).dropna(how="all")
        if len(df) > 60:
            out[t] = df
    return out
