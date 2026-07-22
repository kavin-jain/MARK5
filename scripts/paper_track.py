"""
MARK6 — Paper-trading tracker: the bridge between backtest and real money.
==========================================================================
Records the deployed portfolio once, then measures LIVE divergence between the
paper book and the backtest's expectation. The decision rule this enables:
only consider real capital after 6-12 months of paper NAV tracking the
backtest within its expected noise band — never before.

  python3 scripts/paper_track.py init --capital 500000   # snapshot today's book
  python3 scripts/paper_track.py status                  # mark-to-market + log

State: data/paper/paper_book.json (holdings) + data/paper/paper_nav.csv (history).
"""
import os, sys, json, argparse, csv
from datetime import date

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, FactorLibrary, composite_score)

PAPER_DIR = os.path.join(_ROOT, "data", "paper")
BOOK = os.path.join(PAPER_DIR, "paper_book.json")
NAV_LOG = os.path.join(PAPER_DIR, "paper_nav.csv")
SLEEVES = {"GOLDBEES": 0.25, "MON100": 0.25}


def live_prices(tickers: list[str]) -> dict[str, float]:
    """Latest close for each ticker via yfinance (single batched call)."""
    import yfinance as yf
    data = yf.download([f"{t}.NS" for t in tickers], period="5d",
                       auto_adjust=True, progress=False, threads=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(f"{tickers[0]}.NS")
    out = {}
    for t in tickers:
        s = data.get(f"{t}.NS")
        if s is not None and s.dropna().size:
            out[t] = float(s.dropna().iloc[-1])
    return out


def cmd_init(capital: float):
    if os.path.exists(BOOK):
        sys.exit(f"ERROR: {BOOK} already exists — delete it explicitly to restart "
                 f"the paper experiment (that resets the track record).")
    tickers = discover_tickers()
    if not tickers:
        sys.exit("ERROR: empty data cache — run scripts/refetch_all.py first.")
    panel = DataPanel(tickers, str(date.today()))
    asof = panel.close.index[-1]
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08,
                             factor_weights={"momentum": 0.45, "low_vol": 0.15,
                                             "trend": 0.25, "stability": 0.15})
    elig = panel.eligible(asof, 252, 0.40)
    raw = {f: {} for f in FactorLibrary.DEFAULT_FACTORS}
    vol = {}
    for t in elig:
        row = FactorLibrary.compute_all(panel.close[t]).loc[:asof]
        if row.empty:
            continue
        last = row.iloc[-1]
        for f in raw:
            raw[f][t] = last.get(f, float("nan"))
        vol[t] = -last.get("low_vol", float("nan"))
    comp = composite_score({f: pd.Series(v) for f, v in raw.items()}, cfg.factor_weights)
    w_eq = PortfolioConstructor(cfg).target_weights(comp, pd.Series(vol), [])

    eq_frac = 1 - sum(SLEEVES.values())
    holdings = {t: w * eq_frac for t, w in w_eq.items()}
    holdings.update(SLEEVES)
    px = live_prices(list(holdings))
    missing = [t for t in holdings if t not in px]
    if missing:
        sys.exit(f"ERROR: no live price for {missing} — cannot snapshot the book.")

    os.makedirs(PAPER_DIR, exist_ok=True)
    book = {"start_date": str(date.today()), "capital": capital,
            "signal_asof": str(asof.date()),
            "positions": {t: {"weight": w, "entry_price": px[t],
                              "units": capital * w / px[t]}
                          for t, w in holdings.items()}}
    with open(BOOK, "w") as f:
        json.dump(book, f, indent=1)
    print(f"Paper book started {book['start_date']} (signal as-of {book['signal_asof']}), "
          f"₹{capital:,.0f}, {len(holdings)} instruments -> {BOOK}")
    print("Run 'status' weekly; compare the logged NAV against the backtest expectation.")


def cmd_status():
    if not os.path.exists(BOOK):
        sys.exit("ERROR: no paper book — run 'init' first.")
    with open(BOOK) as f:
        book = json.load(f)
    px = live_prices(list(book["positions"]))
    nav = sum(p["units"] * px.get(t, p["entry_price"])
              for t, p in book["positions"].items())
    ret = nav / book["capital"] - 1
    days = (date.today() - date.fromisoformat(book["start_date"])).days
    print(f"Paper book: day {days}  NAV ₹{nav:,.0f}  ({ret*100:+.2f}% since start)")
    print(f"{'ticker':<14}{'entry':>10}{'now':>10}{'P&L %':>8}")
    for t, p in sorted(book["positions"].items(), key=lambda kv: -kv[1]["weight"]):
        now = px.get(t, float("nan"))
        print(f"{t:<14}{p['entry_price']:>10.2f}{now:>10.2f}"
              f"{(now/p['entry_price']-1)*100:>+8.1f}")
    new = not os.path.exists(NAV_LOG)
    with open(NAV_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "day", "nav_inr", "return_pct"])
        w.writerow([str(date.today()), days, f"{nav:.0f}", f"{ret*100:.3f}"])
    print(f"Logged -> {NAV_LOG}")
    # honest context: expected band from the backtest (~20% CAGR, ~15% ann. vol
    # at the 3-sleeve level) — 1-sigma band around the expected path
    exp = (1.20 ** (days / 365.25) - 1) * 100
    sig = 15 * (days / 365.25) ** 0.5
    print(f"Backtest expectation for day {days}: {exp:+.1f}% ± {sig:.1f}% (1σ). "
          f"Sustained tracking OUTSIDE this band means the backtest does not "
          f"describe live reality — do not fund the strategy.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--capital", type=float, default=500000)
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "init":
        if not (0 < args.capital <= 1e12):
            sys.exit("ERROR: --capital must be positive.")
        cmd_init(args.capital)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
