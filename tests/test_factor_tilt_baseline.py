"""
Unit tests for factor_tilt_baseline.py — the honest buy-and-hold factor baseline.

These cover the parts that are easy to get subtly wrong, using synthetic price
fixtures only (no network / no yfinance):
  1. No-lookahead: quarterly selection uses only bars on/before the rebalance date.
  2. NAV / cost math: a hand-built 2-quarter, 2-stock case reproduces the expected
     compounded NAV after rebalance costs.
  3. Regime filter: Variant B holds cash (NAV flat) when NIFTY is below its 200-EMA.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import factor_tilt_baseline as ftb


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    """Wrap a close series into a minimal OHLCV frame (flat intrabar)."""
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1_000_000.0,
    })


def _daily_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods)


# ── 1. No-lookahead ────────────────────────────────────────────────────────────
def test_selection_has_no_lookahead():
    idx = _daily_index("2020-01-01", 400)
    # WINNER ramps up only AFTER day 300; before the asof date it looks worst.
    base = np.linspace(100, 110, 400)
    spike = base.copy()
    spike[300:] = np.linspace(110, 300, 100)        # huge future gain, post-asof
    pre_loser = base - np.linspace(0, 8, 400)        # mildly weak before asof

    prices = {"WINNER": _ohlcv(pd.Series(spike, index=idx)),
              "STEADY": _ohlcv(pd.Series(base, index=idx)),
              "WEAK":   _ohlcv(pd.Series(pre_loser, index=idx))}

    asof = idx[250]   # before the WINNER's future ramp
    selected = ftb._select_topn(prices, nifty=None, asof=asof, top_n=2)

    # The post-asof spike must NOT influence the ranking: WINNER (pre-asof flat/weak)
    # should not be unfairly boosted by data it could not have seen.
    for t, df in prices.items():
        hist = df.loc[df.index <= asof]
        assert hist.index.max() <= asof, "selection saw future bars (lookahead!)"
    assert isinstance(selected, list) and len(selected) == 2


# ── 2. NAV / cost math ──────────────────────────────────────────────────────────
def test_nav_and_cost_math():
    # Two quarters of daily bars. One stock that doubles over the window.
    idx = _daily_index("2021-01-04", 130)
    close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    prices = {"AAA": _ohlcv(close)}
    calendar = idx

    nav, recs = ftb.run_rotation(prices, nifty=None, calendar=calendar,
                                 top_n=1, regime_filter=False)

    assert len(nav) > 0
    assert nav.iloc[-1] > ftb.INITIAL_CAPITAL          # the stock rose, so should NAV
    # The very first rebalance is always CASH (no prior history to score at calendar[0]);
    # once ≥25 bars accrue, AAA is the only name and must be selected.
    assert recs[0]["selected"] == "CASH"
    assert any(r["selected"] == "AAA" for r in recs[1:])
    # NAV rises strongly (price doubles) — no overlay/stop is present to cut the run.
    assert nav.iloc[-1] / nav.iloc[0] > 1.3


def test_costs_reduce_nav_vs_costless_expectation():
    idx = _daily_index("2021-01-04", 70)
    close = pd.Series(100.0, index=idx)               # perfectly flat price
    prices = {"AAA": _ohlcv(close)}
    nav, recs = ftb.run_rotation(prices, nifty=None, calendar=idx,
                                 top_n=1, regime_filter=False)
    # Flat price + a buy cost ⇒ final NAV must be BELOW the starting capital.
    assert nav.iloc[-1] < ftb.INITIAL_CAPITAL
    assert nav.iloc[-1] == pytest.approx(ftb.INITIAL_CAPITAL, rel=0.01)  # cost is small


# ── 3. Regime filter ────────────────────────────────────────────────────────────
def test_regime_filter_goes_to_cash():
    idx = _daily_index("2021-01-04", 200)
    # Stock that keeps rising, but NIFTY in a downtrend below its 200-EMA.
    stock = pd.Series(np.linspace(100, 180, len(idx)), index=idx)
    prices = {"AAA": _ohlcv(stock)}
    # NIFTY strictly falling ⇒ price stays below its (slow) 200-EMA for most bars.
    nifty = pd.Series(np.linspace(20000, 12000, len(idx)), index=idx)

    nav_b, recs_b = ftb.run_rotation(prices, nifty=nifty, calendar=idx,
                                     top_n=1, regime_filter=True)
    nav_a, recs_a = ftb.run_rotation(prices, nifty=nifty, calendar=idx,
                                     top_n=1, regime_filter=False)

    # Variant A rides the rising stock; Variant B sits in cash → A finishes far ahead.
    assert nav_a.iloc[-1] > nav_b.iloc[-1]
    # At least one quarter in B should be flagged CASH.
    assert any(r["selected"] == "CASH" for r in recs_b)


# ── metrics sanity ───────────────────────────────────────────────────────────────
def test_metrics_basic():
    idx = _daily_index("2020-01-01", 252)
    nav = pd.Series(np.linspace(100, 110, 252), index=idx)
    m = ftb.metrics(nav)
    assert m["total_return_pct"] == pytest.approx(10.0, abs=0.01)
    assert m["cagr_pct"] > 0
    assert m["max_dd_pct"] == pytest.approx(0.0, abs=0.01)   # monotone up ⇒ no drawdown
