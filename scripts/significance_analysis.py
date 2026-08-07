"""
How certain can we actually be?  (run me)
=========================================
Every other report in this repo answers "what did it return?". This one answers
"how much of that is knowable?" — the question that decides whether real money
should move.

Three things get computed, all standard buy-side due-diligence:

  1. CONFIDENCE INTERVAL ON THE TRUE RETURN.  A backtest CAGR is one draw from a
     distribution. The standard error of an annualised mean return is vol/sqrt(T),
     so a 21% CAGR at 22% vol over 10.5 years carries a +/-13pp 95% band. Anyone
     quoting the point estimate as a forecast is quoting noise as signal.

  2. INFORMATION RATIO AND TIME-TO-SIGNIFICANCE.  IR = active return / tracking
     error. To reject "zero skill" at 95% one-sided you need t = IR*sqrt(T) > 1.645,
     i.e. T = (1.645/IR)^2 years of data. This is the number that says how long a
     live track must run before it proves anything.

  3. WHAT THE LIVE BOOK CAN AND CANNOT SHOW.  Same math applied to the actual
     paper-track length, to bound what the live number is worth.

Benchmarks are the two honest ones: equal-weight of the SAME point-in-time
universe (isolates selection skill) and Nifty 50 total-return (isolates the whole
package). Beating a cap-weighted index with a concentrated midcap book is mostly
a size tilt, so the equal-weight comparison is the one that matters.

  python3 scripts/significance_analysis.py
"""
import os, sys, json

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_nifty, load_sector_map, load_delivery_factors)

REPORTS = os.path.join(_ROOT, "reports")
END = os.environ.get("MARK5_END", "2026-07-21")
START = os.environ.get("MARK5_START", "2016-01-01")
TRADING_DAYS = 252
Z95 = 1.6448536269514722          # one-sided 95%
Z975 = 1.959963984540054          # two-sided 95%


def annualised(daily: pd.Series) -> tuple[float, float]:
    """(annualised mean, annualised vol) from a daily return series."""
    return float(daily.mean() * TRADING_DAYS), float(daily.std() * np.sqrt(TRADING_DAYS))


def return_ci(daily: pd.Series) -> dict:
    """95% confidence interval on the TRUE expected annual return.

    SE of an annualised mean is vol/sqrt(years) — the estimator improves with
    calendar time, not with sampling frequency, which is why a decade of daily
    data still leaves a double-digit band on an equity-vol strategy.
    """
    mu, vol = annualised(daily)
    years = len(daily) / TRADING_DAYS
    se = vol / np.sqrt(years)
    return {"mean_annual_pct": mu * 100, "vol_pct": vol * 100, "years": years,
            "se_pp": se * 100, "lo95_pct": (mu - Z975 * se) * 100,
            "hi95_pct": (mu + Z975 * se) * 100,
            "t_stat": mu / se if se else 0.0,
            "p_value_gt0": float(1 - _norm_cdf(mu / se)) if se else 1.0}


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def active_stats(strat: pd.Series, bench: pd.Series, label: str) -> dict:
    """Information ratio and years-to-significance against one benchmark."""
    a, b = strat.align(bench, join="inner")
    act = (a - b).dropna()
    if len(act) < 60:
        return {}
    mu, te = annualised(act)
    years = len(act) / TRADING_DAYS
    ir = mu / te if te else 0.0
    t = ir * np.sqrt(years)
    # Years of live data needed to reject zero skill at 95% one-sided.
    yrs_needed = (Z95 / ir) ** 2 if ir > 0 else float("inf")
    return {"benchmark": label, "active_return_pp": mu * 100,
            "tracking_error_pp": te * 100, "information_ratio": ir,
            "years_observed": years, "t_stat": t,
            "p_value": float(1 - _norm_cdf(t)),
            "significant_95": bool(t > Z95),
            "years_to_95pct_significance": yrs_needed}


def block_bootstrap_cagr(daily: pd.Series, n_boot: int = 2000,
                         block: int = 21, seed: int = 7) -> dict:
    """Stationary-block bootstrap CI on CAGR.

    Resamples month-long blocks so serial correlation and volatility clustering
    survive; an iid bootstrap would understate the band on a momentum book.
    """
    rng = np.random.default_rng(seed)
    r = daily.to_numpy()
    n = len(r)
    nblocks = int(np.ceil(n / block))
    years = n / TRADING_DAYS
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=nblocks)
        path = np.concatenate([r[s:s + block] for s in starts])[:n]
        out[i] = (np.prod(1 + path)) ** (1 / years) - 1
    return {"median_pct": float(np.median(out) * 100),
            "lo95_pct": float(np.percentile(out, 2.5) * 100),
            "hi95_pct": float(np.percentile(out, 97.5) * 100),
            "p_negative": float((out < 0).mean()),
            "p_below_nifty_11pct": float((out < 0.11).mean())}


def main():
    os.makedirs(REPORTS, exist_ok=True)
    print("Loading panel...", flush=True)
    panel = DataPanel(discover_tickers(), END)

    factor_cfg = ConstructionConfig(mode="factor_tilt", n_hold=20,
                                    base_weighting="inverse_vol", tilt_strength=1.5,
                                    max_weight=0.08,
                                    factor_weights={"momentum": 0.45, "low_vol": 0.15,
                                                    "trend": 0.25, "stability": 0.15})
    dfac = load_delivery_factors(universe=panel.tickers)
    if dfac:
        factor_cfg.factor_weights = {**factor_cfg.factor_weights, "deliv_chg": 0.10}
    bt_cfg = BacktestConfig(rebal_bars=126,
                            top_n_liquid=int(os.environ.get("MARK5_TOP_N", "300")))

    print("Running deployed book...", flush=True)
    strat = Backtester(panel, PortfolioConstructor(factor_cfg, sector_map=load_sector_map()),
                       bt_cfg, extra_factors=dfac).run(START, END)
    print("Running equal-weight benchmark...", flush=True)
    ew = Backtester(panel, PortfolioConstructor(ConstructionConfig(mode="equal_weight",
                                                                  base_weighting="equal")),
                    bt_cfg).run(START, END)

    # Gross series: we are measuring the RETURN PROCESS, so the one-off terminal
    # liquidation tax must not enter the daily returns (see metrics_after_exit_tax).
    s_ret = strat["nav_gross"].pct_change(fill_method=None).dropna()
    e_ret = ew["nav_gross"].pct_change(fill_method=None).dropna()
    # COVERAGE GUARD. load_nifty silently returns whatever history it has, so
    # asking for 2007 against a series that starts in 2015 yields a benchmark
    # computed on a different window than the strategy — the strategy meets the
    # 2008 crash, the benchmark does not, and the active return is meaningless.
    # Same failure as BUG3 in the research log. Drop the benchmark rather than
    # publish a mismatched one.
    nifty = load_nifty(total_return=True)
    n_ret = pd.Series(dtype=float)
    if nifty is not None and len(nifty):
        lag_days = (nifty.index.min() - pd.Timestamp(START)).days
        if lag_days > 45:
            print(f"  WARNING: Nifty TRI starts {nifty.index.min():%Y-%m-%d}, "
                  f"{lag_days}d after the requested start {START}. Dropping the "
                  f"benchmark comparison rather than scoring against a shorter "
                  f"window.", flush=True)
            res_note = (f"Nifty TRI unavailable before {nifty.index.min():%Y-%m-%d}; "
                        f"benchmark comparison omitted for this window.")
        else:
            n_ret = nifty.loc[START:END].pct_change(fill_method=None).dropna()
            res_note = None

    res = {"generated": pd.Timestamp.utcnow().isoformat(), "window": [START, END],
           "config": "deployed (mom_heavy / n20 / tilt1.5 / r126 / deliv_chg)",
           "cache": os.environ.get("MARK5_CACHE", "data/cache")}
    if res_note:
        res["benchmark_note"] = res_note

    print("\n" + "=" * 78)
    print("  1. HOW WELL DO WE EVEN KNOW THE RETURN?")
    print("=" * 78)
    res["return_ci"] = {}
    for name, r in [("MARK6 equity book", s_ret), ("Equal-weight universe", e_ret),
                    ("Nifty 50 TRI", n_ret)]:
        if r.empty:
            continue
        ci = return_ci(r)
        res["return_ci"][name] = ci
        print(f"  {name:<24} {ci['mean_annual_pct']:+6.1f}%/yr   "
              f"95% CI [{ci['lo95_pct']:+6.1f}%, {ci['hi95_pct']:+6.1f}%]   "
              f"+/-{Z975 * ci['se_pp']:.1f}pp")
    print(f"\n  Read: {res['return_ci']['MARK6 equity book']['years']:.1f} years of daily data "
          f"still leaves a "
          f"+/-{Z975 * res['return_ci']['MARK6 equity book']['se_pp']:.0f}pp band on the true "
          f"mean.\n  The point estimate is not a forecast.")

    print("\n" + "=" * 78)
    print("  2. IS THERE SKILL, AND HOW LONG TO PROVE IT?")
    print("=" * 78)
    res["active"] = []
    for bench, label in [(e_ret, "Equal-weight same universe (SELECTION SKILL)"),
                         (n_ret, "Nifty 50 TRI (WHOLE PACKAGE)")]:
        if bench.empty:
            continue
        a = active_stats(s_ret, bench, label)
        if not a:
            continue
        res["active"].append(a)
        verdict = "SIGNIFICANT" if a["significant_95"] else "NOT significant"
        print(f"\n  vs {label}")
        print(f"    active return   {a['active_return_pp']:+6.2f}pp/yr")
        print(f"    tracking error  {a['tracking_error_pp']:6.2f}pp")
        print(f"    Information Ratio {a['information_ratio']:.3f}")
        print(f"    t-stat {a['t_stat']:.2f} over {a['years_observed']:.1f}y -> "
              f"p={a['p_value']:.4f}  [{verdict} at 95%]")
        print(f"    years of LIVE data needed to prove this from scratch: "
              f"{a['years_to_95pct_significance']:.1f}")

    print("\n" + "=" * 78)
    print("  3. BOOTSTRAP — WHAT RANGE OF OUTCOMES IS CONSISTENT WITH THE EVIDENCE?")
    print("=" * 78)
    bs = block_bootstrap_cagr(s_ret)
    res["bootstrap"] = bs
    print(f"  median CAGR {bs['median_pct']:+.1f}%   95% band "
          f"[{bs['lo95_pct']:+.1f}%, {bs['hi95_pct']:+.1f}%]")
    print(f"  P(true CAGR < 0)      {bs['p_negative'] * 100:.1f}%")
    print(f"  P(true CAGR < Nifty)  {bs['p_below_nifty_11pct'] * 100:.1f}%")

    print("\n" + "=" * 78)
    print("  4. WHAT IS THE LIVE TRACK WORTH TODAY?")
    print("=" * 78)
    live = _live_verdict(res)
    res["live"] = live
    if live:
        print(f"  observations: {live['observations']}  ({live['days']} calendar days)")
        print(f"  at the measured IR of {live['ir_used']:.2f}, proving skill needs "
              f"{live['years_needed']:.1f} years")
        print(f"  -> the live track carries {live['information_pct']:.1f}% of the data "
              f"needed for a 95% verdict")

    with open(os.path.join(REPORTS, "significance_analysis.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\n  Saved -> reports/significance_analysis.json\n")
    return res


def _live_verdict(res: dict) -> dict:
    nav = os.path.join(_ROOT, "data", "paper", "paper_nav.csv")
    if not os.path.exists(nav) or not res.get("active"):
        return {}
    df = pd.read_csv(nav)
    sel = res["active"][0]                       # vs equal-weight = selection skill
    need = sel["years_to_95pct_significance"]
    days = int(df["day"].iloc[-1]) if len(df) else 0
    return {"observations": len(df), "days": days,
            "ir_used": sel["information_ratio"], "years_needed": need,
            "information_pct": (days / 365.25) / need * 100 if need else 0.0}


def _selftest():
    """Sanity-check the estimators on constructed series."""
    idx = pd.bdate_range("2016-01-01", periods=TRADING_DAYS * 4)
    rng = np.random.default_rng(0)
    # Zero-skill: active return is pure noise. The defining property is that it
    # does NOT clear the 95% bar -- not that the IR lands near zero. At 16% vol
    # over 4 years the SE on the annualised mean is ~8pp, so sizeable |IR| draws
    # are ordinary noise; asserting a tight IR band would be asserting luck.
    a = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    st = active_stats(a, pd.Series(0.0, index=idx), "noise")
    assert not st["significant_95"], st
    assert abs(st["t_stat"]) < 2.0, st
    # Known drift: 10% a year on 10% vol -> IR ~ 1.0, ~2.7 years to significance.
    d = pd.Series(0.10 / TRADING_DAYS, index=idx) + pd.Series(rng.normal(0, 0.10 / np.sqrt(TRADING_DAYS), len(idx)), index=idx)
    st2 = active_stats(d, pd.Series(0.0, index=idx), "drift")
    assert 0.7 < st2["information_ratio"] < 1.4, st2
    assert 1.0 < st2["years_to_95pct_significance"] < 6.0, st2
    # CI must widen as vol rises.
    lo_vol = return_ci(pd.Series(rng.normal(0.0004, 0.005, len(idx)), index=idx))
    hi_vol = return_ci(pd.Series(rng.normal(0.0004, 0.020, len(idx)), index=idx))
    assert hi_vol["se_pp"] > lo_vol["se_pp"] * 3, (lo_vol, hi_vol)
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
