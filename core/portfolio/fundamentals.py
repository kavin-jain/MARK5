"""
MARK6 — Fundamental quality factors (F3), from indianapi.in historical financials.
==================================================================================
Builds causal, disclosure-lagged quality sub-factors per ticker (higher = better):
  roce          : ROCE %                       (capital efficiency / profitability)
  low_debt      : -Borrowings/(Equity+Reserves) (balance-sheet strength)
  fcf_margin    : Free Cash Flow / Sales        (cash quality — earnings are real)
  earn_stability: -stdev(OPM %, trailing<=5y)   (earnings consistency — the literature's
                                                 cash-flow/earnings-variability quality)

ZERO look-ahead: annual results for FY ending "Mar YYYY" are only used from
1 Oct YYYY (a conservative ~6-month disclosure lag — real results publish ~May-Jul).
An as-of lookup at a rebalance date therefore sees only already-published statements.

Banks/financials report differently (sparse ROCE/OPM) -> those names get NaN ->
neutral in the cross-sectional composite. Data: scripts/fetch_fundamentals.py.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FUND = os.path.join(_ROOT, "data", "cache", "fundamentals")
QUALITY_FACTORS = ("roce", "low_debt", "fcf_margin", "earn_stability")
_MON = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}


def _series(d, *path):
    """Dig nested dict path -> {label: value} or {}."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return {}
        cur = cur[p]
    return cur if isinstance(cur, dict) else {}


def _disclosure(label):
    """'Mar 2020' -> 2020-10-01 (FY-end + ~6mo conservative publish lag)."""
    try:
        mon, yr = label.split()
        y = int(yr)
        # FY ending in month m, results public ~6 months later
        return pd.Timestamp(year=y, month=10, day=1) if mon == "Mar" else \
            pd.Timestamp(year=y, month=_MON.get(mon, 3), day=1) + pd.DateOffset(months=6)
    except Exception:
        return None


def load_quality_factors(src: str = FUND) -> dict[str, pd.DataFrame]:
    """ticker -> causal DataFrame(index=disclosure date, cols=QUALITY_FACTORS)."""
    out: dict[str, pd.DataFrame] = {}
    for f in glob.glob(os.path.join(src, "*.json")):
        t = os.path.basename(f).replace(".json", "")
        d = json.load(open(f))
        if d.get("error"):
            continue
        roce = _series(d, "ratios", "ROCE %")
        opm = _series(d, "yoy_results", "OPM %")
        sales = _series(d, "yoy_results", "Sales")
        fcf = _series(d, "cashflow", "Free Cash Flow")
        borr = _series(d, "balancesheet", "Borrowings")
        eqcap = _series(d, "balancesheet", "Equity Capital")
        resv = _series(d, "balancesheet", "Reserves")
        labels = sorted(set(roce) | set(opm) | set(sales) | set(borr),
                        key=lambda L: (_disclosure(L) or pd.Timestamp.min))
        rows = {}
        opm_hist = []
        for L in labels:
            disc = _disclosure(L)
            if disc is None:
                continue
            r = {}
            if L in roce and roce[L] is not None:
                r["roce"] = float(roce[L])
            # leverage
            eq = (eqcap.get(L) or 0) + (resv.get(L) or 0)
            if L in borr and eq and eq > 0:
                r["low_debt"] = -float(borr[L]) / float(eq)
            # cash quality
            if L in fcf and L in sales and sales.get(L):
                try:
                    r["fcf_margin"] = float(fcf[L]) / float(sales[L])
                except (TypeError, ZeroDivisionError):
                    pass
            # earnings stability (trailing OPM dispersion)
            if L in opm and opm[L] is not None:
                opm_hist.append(float(opm[L]))
            if len(opm_hist) >= 3:
                r["earn_stability"] = -float(np.std(opm_hist[-5:]))
            if r:
                rows[disc] = r
        if rows:
            df = pd.DataFrame(rows).T.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            out[t] = df
    return out
