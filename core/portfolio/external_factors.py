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
