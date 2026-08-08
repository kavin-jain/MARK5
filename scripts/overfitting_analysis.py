"""
MARK6 — Overfitting / Statistical-Significance Analysis
=======================================================
Answers the question every serious backtest must: given the many strategy
variants we tried, is the deployed config's Sharpe genuine skill or just the
luckiest draw? Computes (Bailey & Lopez de Prado):
  - Probabilistic Sharpe Ratio (PSR vs 0)
  - Deflated Sharpe Ratio (DSR) — deflated for the N trials attempted
  - Probability of Backtest Overfitting (PBO) via CSCV

It re-runs a grid of the strategy variants we explored (the "trials"), collects
their daily returns, and runs the tests on the deployed config (n_hold=12,
tilt=1.5, blend). Writes reports/OVERFITTING_ANALYSIS.md.

  python3 scripts/overfitting_analysis.py
"""
import os, sys, itertools
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_sector_map)
from core.portfolio.stats import (deflated_sharpe_ratio, pbo_cscv,
                                   probabilistic_sharpe_ratio, _sharpe)
START = os.environ.get("MARK5_START", "2016-01-01")
END = os.environ.get("MARK5_END", "2026-07-21")
REPORTS = os.path.join(_ROOT, "reports")

WEIGHTS = {
    "blend":      {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20},
    "mom_heavy":  {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15},
    "lowvol_hvy": {"momentum": .15, "low_vol": .50, "trend": .15, "stability": .20},
    "trend_hvy":  {"momentum": .20, "low_vol": .20, "trend": .45, "stability": .15},
    "stab_hvy":   {"momentum": .20, "low_vol": .20, "trend": .15, "stability": .45},
}
DEPLOYED = ("mom_heavy", 20, 1.5, 126)  # v7.5 deployed: mom-heavy, n_hold=20, 6-month
# refresh. n_hold=12 (the old v7.0 deployment) was FALSIFIED on the honest PIT
# universe in v7.3 (K25: 1/8 walk-forward windows, -5.42pp) and is now only a trial.

# Research variants whose full daily series we don't regenerate here, but which
# WERE trials and must count toward the deflation. Hard-coded ones are from
# efficiency_research.py / exit_speed_research.py (2026-06-11); the rest are READ
# from the v7.3-v7.5 research artifacts so the trial count cannot silently drift
# below the number of things actually tried. Under-counting trials inflates the
# Deflated Sharpe, so this list erring long is the conservative direction.
_LEGACY_SHARPES_ANN = [
    1.03, 1.03, 0.96, 0.91, 0.95, 0.97,   # asymmetric exit variants (6)
    0.80, 0.81,                            # TLH -7% / -12%
    0.84,                                  # FIP 10%
    0.87, 0.91, 0.90,                      # sleeve-frequency variants (3)
]


def _session_trial_sharpes():
    """Annualised Sharpes of every config tested in the v7.3-v7.5 sweeps."""
    import glob, json
    out = []
    for f, path in (("edge_research_2026_07.json", ("full_sy", "sharpe_excess")),
                    ("drawdown_research.json", ("sy", "sharpe_excess"))):
        p = os.path.join(REPORTS, f)
        if not os.path.exists(p):
            continue
        for v in json.load(open(p)).values():
            m = v.get(path[0]) if isinstance(v, dict) else None
            if isinstance(m, dict) and m.get(path[1]) is not None:
                out.append(float(m[path[1]]))
    for f, key, fld in (("allocation_walkforward.json", "fixed_grid", "sharpe_excess"),
                        ("sleeve_rebalance_erc.json", "rows", "sharpe_excess")):
        p = os.path.join(REPORTS, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for r in (d.get(key) or []):
            if r.get(fld) is not None:
                out.append(float(r[fld]))
    return out


EXTRA_TRIAL_SHARPES_ANN = _LEGACY_SHARPES_ANN + _session_trial_sharpes()


_SLEEVES = None


def _passive_sleeves(panel):
    """Daily INR returns for the two PASSIVE sleeves, aligned to the panel calendar.

    Uses the underlying assets in INR (gold spot, Nasdaq-100, each x USDINR) rather
    than GOLDBEES/MON100, whose cached history starts in 2015 and cannot cover the
    research window. Ignores ETF expense (~0.5%/yr), tracking error and the premium
    Indian international ETFs trade at, so the passive legs are modelled slightly
    generously — stated here because it flatters the product-level numbers a little.
    """
    import yfinance as yf

    def px(sym):
        h = yf.download(sym, start="2006-11-01", end="2026-07-22",
                        auto_adjust=True, progress=False)["Close"]
        if hasattr(h, "columns"):
            h = h.iloc[:, 0]
        h = h.dropna()
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return h[~h.index.duplicated()]

    idx = panel.close.index
    raw_fx = px("USDINR=X")
    if len(raw_fx) < 1000:
        sys.exit(f"ABORT: USDINR returned {len(raw_fx)} rows. A short or empty fetch "
                 f"would ffill/bfill into a constant series and silently model a "
                 f"zero-return sleeve.")
    fx = raw_fx.reindex(idx).ffill().bfill()
    out = {}
    for key, sym in (("gold", "GC=F"), ("us", "^NDX")):
        raw = px(sym)
        # A failed download reindexes to all-NaN, survives ffill/bfill, and becomes
        # a 0.0% daily return after fillna — i.e. a sleeve that silently contributes
        # nothing while still taking 25% of the book. Every statistic downstream
        # would be wrong and nothing would look broken. Same family as BUG2/BUG3.
        if len(raw) < 1000 or raw.index.max() < pd.Timestamp("2026-01-01"):
            sys.exit(f"ABORT: {sym} returned {len(raw)} rows ending "
                     f"{raw.index.max() if len(raw) else 'never'}. Refusing to model "
                     f"a passive sleeve from an incomplete fetch.")
        s = (raw.reindex(idx).ffill() * fx).ffill().bfill()
        r = s.pct_change(fill_method=None).fillna(0.0)
        if float(r.std()) < 1e-9:
            sys.exit(f"ABORT: {sym} sleeve has zero variance after alignment.")
        out[key] = r
    return out


def main():
    panel = DataPanel(discover_tickers(), END)
    from core.portfolio import BacktestConfig
    # the grid of "trials" we actually explored across the project (annual rebal era)
    grid = [(w, n, t, 252) for w, n, t in
            itertools.product(WEIGHTS.keys(), [8, 12, 16, 20], [0.5, 1.5, 3.0])]
    # + the rebalance-frequency dimension explored 2026-06-11 (mom_heavy book)
    # rebalance-frequency dimension, for BOTH the old n_hold=12 deployment and the
    # current n_hold=20 one. The deployed config must appear in this grid or its own
    # return series is never captured and the DSR silently evaluates to NaN.
    grid += [("mom_heavy", n, 1.5, rb)
             for n in (12, 20) for rb in (21, 42, 63, 126, 189)]
    print(f"Running {len(grid)} strategy trials to assemble the returns matrix...", flush=True)

    rets, sharpes, labels, deployed_ret = {}, [], [], None
    sleeve_sharpes = []
    cal = None
    global _SLEEVES
    if os.environ.get("MARK5_GRADE_PRODUCT") == "1":
        _SLEEVES = _passive_sleeves(panel)
        print("  grading the PRODUCT (50/25/25 equity/gold/US), not the equity sleeve",
              flush=True)
    for wname, nh, tilt, rb in grid:
        cfg = ConstructionConfig(mode="factor_tilt", n_hold=nh, base_weighting="inverse_vol",
                                 tilt_strength=tilt, max_weight=max(0.08, 1.5 / nh),
                                 factor_weights=WEIGHTS[wname])
        nav = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                         BacktestConfig(rebal_bars=rb,
                                        top_n_liquid=int(os.environ.get("MARK5_TOP_N", "300")))
                         ).run(START, END)["nav_gross"]
        r = nav.pct_change(fill_method=None).fillna(0.0)
        if cal is None:
            cal = r.index
        r = r.reindex(cal).fillna(0.0)
        # P5.1. Optionally grade the PRODUCT rather than the equity sleeve. The
        # deployed book is 50/25/25 equity/gold/US; statistics computed on the
        # sleeve alone describe something nobody owns (Mandate §3). This is a
        # correction, not a flattering re-basing, and it cuts BOTH ways: with half
        # the book fixed and passive, a config search can only move half the risk,
        # so the luck ceiling SR0 genuinely falls — but so does the deployed
        # strategy's own excess over that ceiling.
        r_sleeve = r                      # equity-sleeve returns, before blending
        if _SLEEVES is not None:
            r = 0.50 * r + 0.25 * _SLEEVES["gold"] + 0.25 * _SLEEVES["us"]
        sleeve_sharpes.append(_sharpe(r_sleeve.values))
        lab = f"{wname}|n{nh}|t{tilt}|r{rb}"
        rets[lab] = r.values
        sharpes.append(_sharpe(r.values))
        labels.append(lab)
        if (wname, nh, tilt, rb) == DEPLOYED:
            deployed_ret = r.values
    # count the un-regenerated research variants toward the luck ceiling
    # The counted-only historical trials are EQUITY-SLEEVE Sharpes. When grading
    # the product they must be mapped to the same basis, or the trial dispersion
    # mixes two units and the luck ceiling is nonsense — the first attempt at this
    # left them unconverted, which inflated sr* from 0.32 to 1.22 and crushed DSR
    # to 85.8% for purely arithmetic reasons. The map is FITTED on the 70 configs
    # where both bases are observed, so it is measured rather than assumed.
    extra = [s / np.sqrt(252) for s in EXTRA_TRIAL_SHARPES_ANN]
    if _SLEEVES is not None and len(sleeve_sharpes) == len(sharpes) > 2:
        b, a = np.polyfit(np.array(sleeve_sharpes), np.array(sharpes), 1)
        rho = float(np.corrcoef(sleeve_sharpes, sharpes)[0, 1])
        extra = [a + b * e for e in extra]
        print(f"  mapped {len(extra)} counted-only trials to product basis "
              f"(product = {a:.4f} + {b:.3f} x sleeve, r={rho:.3f})", flush=True)
    sharpes += extra
    print(f"  done. {len(labels)} series + {len(EXTRA_TRIAL_SHARPES_ANN)} "
          f"counted-only trials = {len(sharpes)} total.\n", flush=True)

    if deployed_ret is None:
        sys.exit(f"ERROR: deployed config {DEPLOYED} is not in the trial grid, so its "
                 f"return series was never captured. Every statistic below would be NaN. "
                 f"Add it to `grid` rather than reporting an empty result.")
    M = np.column_stack([rets[l] for l in labels])     # T x N
    dsr = deflated_sharpe_ratio(deployed_ret, sharpes)
    pbo = pbo_cscv(M, n_splits=12)
    ann = lambda d: d * np.sqrt(252)                   # daily->annual SR

    L = ["# MARK6 — Overfitting & Statistical-Significance Analysis", "",
         "Bailey & López de Prado tests on the DEPLOYED v7.5 config (momentum-heavy / "
         "n_hold=20 / tilt=1.5 / 126-bar refresh, rank-transformed scores, sector cap "
         "enforced, FY-netting tax), using every strategy "
         "variant explored across the project as the trial set (factor-weight grid, "
         "rebalance frequencies, asymmetric exits, TLH, FIP, sleeve frequencies). "
         f"All on daily returns, {START[:4]}-{END[:4]}, universe "
         f"`{os.environ.get('MARK5_CACHE', 'data/cache')}`.", "",
         "## Deflated Sharpe Ratio (is the Sharpe real, given how many we tried?)", "",
         f"- Strategy variants tried (N): **{dsr['n_trials']}**",
         f"- Observed Sharpe: **{ann(dsr['observed_sharpe_daily']):.2f}** annualised "
         f"({dsr['observed_sharpe_daily']:.3f} daily)",
         f"- Probabilistic Sharpe Ratio vs 0 (P true SR>0): **{dsr['psr_vs_zero']*100:.1f}%**",
         f"- Expected max Sharpe from pure luck across {dsr['n_trials']} trials: "
         f"{ann(dsr['expected_max_sharpe_luck']):.2f} annualised",
         f"- **Deflated Sharpe Ratio (P skill survives multiple-testing): "
         f"{dsr['deflated_sharpe']*100:.1f}%**", "",
         "## Probability of Backtest Overfitting (PBO via CSCV)", "",
         f"- Strategies in matrix: {pbo['n_strategies']} | train/test combos: {pbo['n_combos']}",
         f"- **PBO: {pbo['pbo']*100:.1f}%** (fraction of splits where the in-sample-best "
         "strategy lands below the out-of-sample median)",
         f"- Median performance-degradation logit: {pbo['median_logit']:.2f} "
         f"({'positive = robust' if pbo['median_logit'] > 0 else 'negative = overfit'})", "",
         "## Verdict", ""]
    dsr_ok = dsr["deflated_sharpe"] > 0.95
    pbo_ok = pbo["pbo"] < 0.20
    L.append(f"- DSR {'PASS' if dsr_ok else 'WEAK'}: deflated-Sharpe "
             f"{dsr['deflated_sharpe']*100:.0f}% — "
             f"{'the Sharpe survives multiple-testing; >95% confidence it is skill, not the luckiest draw.' if dsr_ok else 'caution — significance is borderline after deflation.'}")
    L.append(f"- PBO {'PASS' if pbo_ok else 'WEAK'}: {pbo['pbo']*100:.0f}% — "
             f"{'low overfitting risk; the config generalises out-of-sample.' if pbo_ok else 'elevated overfitting risk.'}")
    L.append("")
    L.append("These are the statistics professional quant funds use to vet a strategy "
             "before risking capital — most retail/student backtests never compute them.")

    os.makedirs(REPORTS, exist_ok=True)
    open(os.path.join(REPORTS, "OVERFITTING_ANALYSIS.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print("\nSaved -> reports/OVERFITTING_ANALYSIS.md")


if __name__ == "__main__":
    main()
