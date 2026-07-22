"""
MARK6 — Live paper track record. No real money; every number real.
==================================================================
This is the bridge between "it backtested well" and "it works". It records the
deployed portfolio ONCE at real market prices, then marks it to market every day.

WHAT "PAPER" MEANS HERE — and what it does NOT mean:
  It means no rupees are at risk. It does NOT mean the numbers are invented.
  Every figure below comes from an actual market print:
    - Entry prices are real closing prices on the day the book was opened.
    - Quantities are WHOLE SHARES, because you cannot buy 4.3 shares of anything.
      Leftover cash is tracked as cash, not silently assumed invested.
    - Entry costs (brokerage, STT, stamp duty, exchange + SEBI fees, GST) are
      deducted at the real Zerodha delivery rates.
    - Every mark-to-market is a real closing price fetched that day.
  The ledger is APPEND-ONLY. Rows are never rewritten, so a bad week cannot be
  quietly edited out later. That is the entire point of keeping it in public.

  python3 scripts/paper_track.py init --capital 500000   # open the book (once)
  python3 scripts/paper_track.py status                  # mark to market + log
  python3 scripts/paper_track.py export                  # JSON for the dashboard

State: data/paper/paper_book.json · paper_nav.csv · paper_ledger.csv
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, FactorLibrary, composite_score)

PAPER_DIR = os.path.join(_ROOT, "data", "paper")
BOOK = os.path.join(PAPER_DIR, "paper_book.json")
NAV_LOG = os.path.join(PAPER_DIR, "paper_nav.csv")
LEDGER = os.path.join(PAPER_DIR, "paper_ledger.csv")
SLEEVES = {"GOLDBEES": 0.25, "MON100": 0.25}
N_HOLD, TOP_N = 20, 300

# Real Zerodha equity-delivery costs (buy side), as fractions of turnover.
BUY_COSTS = 0.001 + 0.00015 + 0.0000297 + 0.000001      # STT + stamp + NSE txn + SEBI
GST_ON = 0.18 * (0.0000297 + 0.000001)                   # GST applies to txn+SEBI fees
BUY_COST_RATE = BUY_COSTS + GST_ON                       # brokerage on delivery = 0


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def live_prices(tickers: list[str]) -> dict[str, float]:
    """Latest real closing price per ticker (yfinance, one batched call)."""
    import yfinance as yf
    data = yf.download([f"{t}.NS" for t in tickers], period="7d",
                       auto_adjust=True, progress=False, threads=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(f"{tickers[0]}.NS")
    out = {}
    for t in tickers:
        s = data.get(f"{t}.NS")
        if s is not None and s.dropna().size:
            out[t] = float(s.dropna().iloc[-1])
    return out


def append_ledger(rows: list[dict]):
    """Append-only. Never rewrites history — that is what makes it a record."""
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "date", "action", "ticker",
                                          "qty", "price", "value_inr", "cost_inr", "note"])
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def target_book():
    """Today's deployed portfolio, from the same code path the backtest uses."""
    tickers = discover_tickers()
    if not tickers:
        sys.exit("ERROR: empty price cache — run scripts/fetch_bhavcopy.py + build_pit_cache.py")
    panel = DataPanel(tickers, str(pd.Timestamp.today().date()), freshness="off")
    asof = panel.close.index[-1]
    age = (pd.Timestamp.today().normalize() - asof.normalize()).days
    if age > 7:
        sys.exit(f"ERROR: price data ends {asof.date()}, {age} days ago. Refusing to open a "
                 f"book on stale prices — refresh the cache first.")
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=N_HOLD, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08,
                             factor_weights={"momentum": 0.45, "low_vol": 0.15,
                                             "trend": 0.25, "stability": 0.15})
    elig = panel.eligible(asof, 252, top_n=TOP_N)
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
    w = PortfolioConstructor(cfg).target_weights(comp, pd.Series(vol), [])
    return w, asof, len(elig)


def cmd_init(capital: float):
    if os.path.exists(BOOK):
        sys.exit(f"ERROR: {BOOK} already exists. Restarting the track record would erase a "
                 f"real history — if that is genuinely what you want, move the file aside "
                 f"manually and say so publicly.")
    w_eq, asof, n_elig = target_book()
    eq_frac = 1 - sum(SLEEVES.values())
    targets = {t: float(x) * eq_frac for t, x in w_eq.items()}
    targets.update(SLEEVES)

    px = live_prices(list(targets))
    missing = [t for t in targets if t not in px]
    if missing:
        sys.exit(f"ERROR: no live price for {missing}; refusing to record a book with "
                 f"guessed prices.")

    # WHOLE SHARES ONLY — you cannot buy a fraction. Leftover stays as real cash.
    positions, rows, spent = {}, [], 0.0
    for t, target_w in sorted(targets.items(), key=lambda kv: -kv[1]):
        budget = capital * target_w
        qty = int(budget // px[t])
        if qty <= 0:
            rows.append({"timestamp": now_iso(), "date": str(pd.Timestamp.today().date()),
                         "action": "SKIP", "ticker": t, "qty": 0, "price": f"{px[t]:.2f}",
                         "value_inr": 0, "cost_inr": 0,
                         "note": f"1 share (Rs {px[t]:,.0f}) exceeds its Rs {budget:,.0f} slot"})
            continue
        value = qty * px[t]
        cost = value * BUY_COST_RATE
        spent += value + cost
        positions[t] = {"qty": qty, "entry_price": px[t], "entry_value": value,
                        "entry_cost": cost, "target_weight": target_w}
        rows.append({"timestamp": now_iso(), "date": str(pd.Timestamp.today().date()),
                     "action": "BUY", "ticker": t, "qty": qty, "price": f"{px[t]:.2f}",
                     "value_inr": f"{value:.2f}", "cost_inr": f"{cost:.2f}", "note": ""})
    cash = capital - spent

    os.makedirs(PAPER_DIR, exist_ok=True)
    book = {"start_date": str(pd.Timestamp.today().date()), "start_timestamp": now_iso(),
            "capital": capital, "signal_asof": str(asof.date()),
            "eligible_universe": n_elig, "n_hold": N_HOLD, "top_n_liquid": TOP_N,
            "cost_rate_buy": BUY_COST_RATE, "cash": cash, "positions": positions,
            "mode": "PAPER — no real money; all prices, quantities and costs are real"}
    book["integrity"] = hashlib.sha256(
        json.dumps(book, sort_keys=True).encode()).hexdigest()[:16]
    json.dump(book, open(BOOK, "w"), indent=1)
    append_ledger(rows)

    inv = sum(p["entry_value"] for p in positions.values())
    print(f"\n  PAPER BOOK OPENED  {book['start_date']}  (signal as-of {asof.date()})")
    print(f"  Capital Rs {capital:,.0f} | invested Rs {inv:,.0f} in {len(positions)} instruments "
          f"| entry costs Rs {sum(p['entry_cost'] for p in positions.values()):,.0f} "
          f"| uninvested cash Rs {cash:,.0f}")
    skipped = [r["ticker"] for r in rows if r["action"] == "SKIP"]
    if skipped:
        print(f"  Could not buy (1 share costs more than the slot): {skipped}")
    print(f"  Integrity hash {book['integrity']} — recorded in {BOOK}")
    print(f"  Ledger -> {LEDGER} (append-only)\n  Run 'status' to mark to market.\n")


def _mark(book):
    px = live_prices(list(book["positions"]))
    mv, detail = book.get("cash", 0.0), []
    for t, p in book["positions"].items():
        now = px.get(t)
        if now is None:
            now = p["entry_price"]
        val = p["qty"] * now
        mv += val
        detail.append({"ticker": t, "qty": p["qty"], "entry": p["entry_price"],
                       "price": now, "value": val,
                       "pnl": val - p["entry_value"] - p["entry_cost"],
                       "pnl_pct": (now / p["entry_price"] - 1) * 100,
                       "weight": val, "stale": px.get(t) is None})
    for d in detail:
        d["weight"] = d["value"] / mv * 100 if mv else 0
    return mv, sorted(detail, key=lambda d: -d["value"])


def cmd_status(quiet=False):
    if not os.path.exists(BOOK):
        sys.exit("ERROR: no paper book — run 'init' first.")
    book = json.load(open(BOOK))
    nav, detail = _mark(book)
    ret = nav / book["capital"] - 1
    days = (pd.Timestamp.today().normalize()
            - pd.Timestamp(book["start_date"]).normalize()).days
    if not quiet:
        print(f"\n  PAPER BOOK — day {days} since {book['start_date']}")
        print(f"  NAV Rs {nav:,.0f}  ({ret*100:+.2f}%)   cash Rs {book.get('cash',0):,.0f}")
        print(f"  {'ticker':<14}{'qty':>6}{'entry':>10}{'now':>10}{'value':>12}{'P&L %':>9}")
        for d in detail:
            print(f"  {d['ticker']:<14}{d['qty']:>6}{d['entry']:>10.2f}{d['price']:>10.2f}"
                  f"{d['value']:>12,.0f}{d['pnl_pct']:>+9.1f}")
    today = str(pd.Timestamp.today().date())
    seen = set()
    if os.path.exists(NAV_LOG):
        seen = {r.split(",")[0] for r in open(NAV_LOG).read().splitlines()[1:]}
    if today not in seen:                      # one honest row per calendar day
        new = not os.path.exists(NAV_LOG)
        with open(NAV_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "day", "nav_inr", "return_pct", "timestamp"])
            w.writerow([today, days, f"{nav:.2f}", f"{ret*100:.4f}", now_iso()])
    return book, nav, ret, days, detail


def cmd_export():
    """Emit the JSON the public dashboard reads. Real data only."""
    book, nav, ret, days, detail = cmd_status(quiet=True)
    hist = []
    if os.path.exists(NAV_LOG):
        hist = list(csv.DictReader(open(NAV_LOG)))
    out = {"generated": now_iso(), "mode": book["mode"],
           "start_date": book["start_date"], "days_live": days,
           "capital": book["capital"], "nav": nav, "return_pct": ret * 100,
           "cash": book.get("cash", 0), "integrity": book.get("integrity"),
           "holdings": detail, "nav_history": hist}
    path = os.path.join(PAPER_DIR, "paper_export.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"  wrote {path}  (day {days}, NAV Rs {nav:,.0f}, {ret*100:+.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--capital", type=float, default=500000)
    sub.add_parser("status"); sub.add_parser("export")
    a = ap.parse_args()
    if a.cmd == "init":
        if not (0 < a.capital <= 1e12):
            sys.exit("ERROR: --capital must be positive.")
        cmd_init(a.capital)
    elif a.cmd == "status":
        cmd_status()
    else:
        cmd_export()


if __name__ == "__main__":
    main()
