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
from core.portfolio import (DataPanel, discover_tickers, load_ohlcv,
                            PortfolioConstructor,
                            ConstructionConfig, FactorLibrary, composite_score,
                            load_sector_map, load_delivery_factors)

PAPER_DIR = os.path.join(_ROOT, "data", "paper")
BOOK = os.path.join(PAPER_DIR, "paper_book.json")
NAV_LOG = os.path.join(PAPER_DIR, "paper_nav.csv")
LEDGER = os.path.join(PAPER_DIR, "paper_ledger.csv")
# Four equal sleeves. Equity is the remainder, so this dict alone sets the
# allocation: 1 - 0.75 = 25% equity.
#
# Staged in DEPLOYMENT_2027-01 and lands at the scheduled rebalance (~2027-01-24),
# NOT on the day it was written — Mandate §6. Evidence, 2007-2026 (19 years, so it
# contains 2008, which the 2016-2026 window does not):
#
#     50 eq / 25 gold / 25 US   +17.27% CAGR   Sharpe 1.25   MaxDD -41.80%
#     25 / 25 / 25 / 25 gilt    +16.16% CAGR   Sharpe 1.43   MaxDD -17.09%
#
# 1.1pp of return for less than half the drawdown. That trade is invisible on a
# window without a financial crisis in it, which is why the shorter window had
# undersold this change.
#
# LTGILTBEES verified 2026-08-09: Rs 30.07/share, Rs 3.78cr median daily turnover.
# The low share price matters as much as the liquidity — a Rs 1.3 lakh sleeve buys
# ~4,300 whole shares, so rounding error is negligible.
#
# N_HOLD stays 20. The 20->60 breadth change was staged and DECLINED (K43): IR
# peaks near 60 but at t=1.90, under this project's 3.0 bar, and 25% equity across
# 60 names is ~Rs 2,166 a position here — less than one share of several holdings.
SLEEVES = {"GOLDBEES": 0.25, "MON100": 0.25, "LTGILTBEES": 0.25}
N_HOLD, TOP_N = 20, 300
# Hoisted out of target_book so the export can publish the rules next to the
# numbers they produced. Anything that needs them reads them from here; a second
# copy in a display layer is a copy that drifts.
MAX_WEIGHT = 0.08
FACTOR_WEIGHTS = {"momentum": 0.45, "low_vol": 0.15, "trend": 0.25, "stability": 0.15}
DELIV_WEIGHT = 0.10          # added only when the delivery archive is present

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


def last_session_date() -> pd.Timestamp | None:
    """Date of the most recent NSE session that actually printed a close.

    live_prices() takes the last non-NaN bar in a 7-day window and says nothing
    about WHEN it was, so on a market holiday it happily returns Friday's close
    as if it were today's. The weekday check catches Saturday and Sunday and
    nothing else; Diwali, Holi and every other trading holiday fall on weekdays.
    Booking fills at a stale close is the one thing an append-only ledger can
    never take back, so the rebalance asks the index what day the market last
    traded rather than assuming.
    """
    import yfinance as yf
    try:
        h = yf.download("^NSEI", period="10d", auto_adjust=True,
                        progress=False)["Close"].dropna()
    except Exception:                                        # noqa: BLE001
        return None
    return pd.Timestamp(h.index[-1]).normalize() if len(h) else None


def reconcile_corporate_actions(book) -> list[str]:
    """Adjust held quantities for splits/bonuses that happened AFTER entry.

    yfinance back-adjusts history, so today's quote is the real market price. If a
    holding splits 1:5 after we bought it, the quote drops ~80% while our stored
    quantity stays put — the book would read a catastrophic fake loss forever. In
    reality the holder now owns 5x the shares. So on every mark we check for splits
    since entry, scale quantity up and entry price down (position value unchanged,
    P&L unaffected), and LOG it. A corporate action is a real event in the record,
    not something to silently absorb.
    """
    import yfinance as yf
    start = pd.Timestamp(book["start_date"])
    notes, rows = [], []
    for t, p in book["positions"].items():
        try:
            sp = yf.Ticker(f"{t}.NS").splits
        except Exception as e:
            # Silently skipping this leaves an unreconciled split looking like a
            # 50-90% single-name loss. healthcheck.py catches the symptom (">30%
            # suspicious move"), but only AFTER it has been marked and published.
            # If yfinance is rate-limited every name skips at once and the whole
            # reconciliation stops running with nothing in the log to show it.
            print(f"  ⚠ split check FAILED for {t}: {type(e).__name__} — "
                  f"corporate actions NOT reconciled for this name")
            continue
        if sp is None or not len(sp):
            continue
        idx = sp.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        applied = float(p.get("split_factor_applied", 1.0))
        total = 1.0
        for d, ratio in zip(idx, sp.values):
            if d > start and float(ratio) > 0:
                total *= float(ratio)
        if abs(total / applied - 1) < 1e-9:
            continue
        new = total / applied
        p["qty"] = int(round(p["qty"] * new))
        p["entry_price"] = p["entry_price"] / new
        p["split_factor_applied"] = total
        notes.append(f"{t} {new:g}x")
        rows.append({"timestamp": now_iso(), "date": str(pd.Timestamp.today().date()),
                     "action": "SPLIT", "ticker": t, "qty": p["qty"],
                     "price": f"{p['entry_price']:.4f}", "value_inr": "", "cost_inr": "",
                     "note": f"corporate action {new:g}x — qty scaled, entry rebased"})
    if rows:
        append_ledger(rows)
        json.dump(book, open(BOOK, "w"), indent=1)
    return notes


def benchmark_value(capital: float, start: str) -> float | None:
    """What the same rupees in the index (NIFTYBEES) would be worth now.

    Without this the live return is unreadable: +5% means nothing if the index did
    +8%. Fetching from `start` alone returns an EMPTY frame on the first day —
    yfinance treats start==end as a zero-width window — so we pull from a buffer
    before it and take the first bar at or after the start date. Failures are
    reported, never swallowed: a silently-missing benchmark is how a dashboard
    ends up quietly flattering itself.
    """
    import yfinance as yf
    buf = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        h = yf.download("NIFTYBEES.NS", start=buf, auto_adjust=True,
                        progress=False, threads=False)["Close"].dropna()
    except Exception as e:
        print(f"  WARN: benchmark fetch failed ({type(e).__name__}: {e})")
        return None
    if isinstance(h, pd.DataFrame):
        h = h.iloc[:, 0].dropna()
    idx = h.index
    if getattr(idx, "tz", None) is not None:
        h.index = idx.tz_localize(None)
    at = h.loc[h.index >= pd.Timestamp(start)]
    if len(at) < 1:
        print(f"  WARN: no benchmark bar at or after {start}")
        return None
    return capital * float(h.iloc[-1]) / float(at.iloc[0])


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


def allocate(targets: dict, px: dict, capital: float) -> dict:
    """Whole-share quantities that track the target weights as closely as the
    share prices allow — largest-remainder apportionment.

    The obvious method, qty = floor(slot / price), rounds EVERY position down.
    At Rs 5 lakh that strands ~1.9% of the book in idle cash and pulls the
    weights 6.6pp away from target, always in the same direction. Measured
    against real forward prices (scripts/capital_flexibility.py) it costs
    -0.35pp/yr at Rs 5L and -1.26pp/yr at Rs 1L.

    Largest-remainder fixes it for free: floor first, then spend the residual
    cash one share at a time on whichever name is furthest below its target and
    still affordable. Same capital, same names, no strategy change — measured
    drag falls to +0.11pp at Rs 5L, i.e. zero. This is the single change that
    lets a small book behave like a large one.

    Returns {ticker: qty}. Never overspends: every purchase is checked against
    remaining cash INCLUDING costs.
    """
    qty, spent = {}, 0.0
    for t, w in targets.items():
        p = px.get(t)
        if not p or p <= 0:
            continue
        q = int((capital * w) // (p * (1 + BUY_COST_RATE)))
        qty[t] = q
        spent += q * p * (1 + BUY_COST_RATE)
    cash = capital - spent
    for _ in range(10000):
        best, best_short = None, 0.0
        for t, w in targets.items():
            p = px.get(t)
            if not p or p <= 0:
                continue
            cost = p * (1 + BUY_COST_RATE)
            if cost > cash + 1e-9:
                continue
            short = capital * w - qty.get(t, 0) * p     # rupees still owed
            if short > best_short:
                best, best_short = t, short
        if best is None:
            break
        qty[best] = qty.get(best, 0) + 1
        cash -= px[best] * (1 + BUY_COST_RATE)
    return qty


MAX_HOLDING_STALE_DAYS = 14


def assert_cache_covers_holdings(book):
    """Refuse to re-pick the book when the cache cannot see what we already own.

    The existing freshness guard in `target_book` checks the PANEL's newest date —
    the maximum across all names. That passes as long as any single ticker is
    current, so it says nothing about breadth. On 2026-08-09 this cache was
    missing RBLBANK, HFCL, AEROFLEX and NYKAA entirely and had BHARATFORG two
    months stale, and the guard was perfectly happy.

    Why that matters more at a rebalance than anywhere else: a name absent from
    the cache cannot be ranked, so it is not in the target book, so the rebalance
    SELLS it. Fifteen holdings would have been liquidated because a fetch failed —
    and the trade log would read like fifteen deliberate decisions. A rebalance
    driven by missing data is indistinguishable, after the fact, from one driven
    by signal, which is what makes it worth stopping rather than warning about.

    Declining is safe: the cadence is still due tomorrow, so a fixed cache simply
    rebalances on the next session (the same contract the weekend and holiday
    guards use).
    """
    cached = set(discover_tickers())
    held = [t for t in book.get("positions", {}) if t not in SLEEVES]
    today = pd.Timestamp.today().normalize()
    absent, stale = [], []
    for t in held:
        if t not in cached:
            absent.append(t)
            continue
        d = load_ohlcv(t)
        if d is None or d.empty:
            absent.append(t)
        elif (today - d.index[-1].normalize()).days > MAX_HOLDING_STALE_DAYS:
            stale.append(f"{t}({d.index[-1].date()})")
    if not (absent or stale):
        return None
    return (f"the price cache does not cover {len(absent) + len(stale)} of "
            f"{len(held)} holdings\n"
            f"  missing entirely : {', '.join(absent) or 'none'}\n"
            f"  stale > {MAX_HOLDING_STALE_DAYS}d      : {', '.join(stale) or 'none'}")


def _num(x, nd=4):
    """JSON-safe float. NaN and inf are not valid JSON and json.dump emits them
    anyway, producing a file that every strict parser rejects."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if f == f and abs(f) != float("inf") else None


SIGNALS = os.path.join(_ROOT, "data", "paper", "signals.json")


def write_signals(signals, basis):
    """Persist the ranking that chose the book, with an explicit `basis` naming
    where the numbers came from. The basis is not decoration: scores captured at
    the rebalance ARE the reasoning; scores recomputed later merely describe how
    the same name ranks today. Presenting the second as the first would be a
    fabricated explanation, which is worse than no explanation."""
    signals = dict(signals, basis=basis)
    os.makedirs(os.path.dirname(SIGNALS), exist_ok=True)
    json.dump(signals, open(SIGNALS, "w"), indent=1, allow_nan=False)
    print(f"  wrote {SIGNALS}  ({len(signals['scores'])} names, {basis})")


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
    # v7.7: deliv_chg added at 10% — PROVISIONAL (RESEARCH_LOG 4l). It wins 4/4
    # walk-forward windows on return and is orthogonal to every price factor AND to
    # size, but its underlying IC is not statistically significant and it makes
    # drawdown ~1.5pp worse. deliv_per_z is deliberately EXCLUDED: it tested worse
    # than baseline. Degrades to the price-only book if the archive is absent.
    dfac = load_delivery_factors(universe=panel.tickers)
    fw = dict(FACTOR_WEIGHTS)
    if dfac:
        fw["deliv_chg"] = DELIV_WEIGHT
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=N_HOLD, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=MAX_WEIGHT, factor_weights=fw)
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
    # delivery factor as-of the signal date, strictly causal
    if dfac:
        raw["deliv_chg"] = {}
        for t in elig:
            e = dfac.get(t)
            if e is not None:
                e = e.loc[:asof]
                if not e.empty:
                    raw["deliv_chg"][t] = e["deliv_chg"].iloc[-1]
    comp = composite_score({f: pd.Series(v) for f, v in raw.items()}, cfg.factor_weights)
    w = PortfolioConstructor(cfg, sector_map=load_sector_map()).target_weights(
        comp, pd.Series(vol), [])

    # Record WHY, not just what. These numbers chose the book and were previously
    # dropped on the floor, which left "why do we hold this" answerable only by
    # re-deriving them later — and a later re-derivation does NOT reproduce the
    # original: corporate actions retroactively adjust price history and the
    # cross-sectional ranks move as names enter and leave the universe. A rebuild
    # of the 2026-07-21 signal returned 5 of the 20 names actually picked. So the
    # reasoning has to be captured AT the decision or it is gone.
    #
    # Percentiles, not raw factor values. composite_score blends cross-sectional
    # rank z-scores with higher always better, so a percentile is directly
    # readable as "stronger than N% of the field" — whereas a raw momentum of
    # 0.34 means nothing without the distribution it sat in.
    pctl = {f: pd.Series(v).rank(pct=True) for f, v in raw.items()}
    scores = {}
    for i, (t, sc) in enumerate(comp.sort_values(ascending=False).items(), 1):
        scores[t] = {"rank": i, "composite": _num(sc),
                     "factors": {f: _num(pctl[f].get(t)) for f in raw}}
    signals = {"asof": str(asof.date()), "n_eligible": len(elig),
               "factor_weights": cfg.factor_weights, "scores": scores}
    return w, asof, len(elig), signals


def cmd_init(capital: float):
    if os.path.exists(BOOK):
        sys.exit(f"ERROR: {BOOK} already exists. Restarting the track record would erase a "
                 f"real history — if that is genuinely what you want, move the file aside "
                 f"manually and say so publicly.")
    w_eq, asof, n_elig, signals = target_book()
    write_signals(signals, f"recorded when the book was opened on "
                           f"{pd.Timestamp.today().date()}")
    eq_frac = 1 - sum(SLEEVES.values())
    targets = {t: float(x) * eq_frac for t, x in w_eq.items()}
    targets.update(SLEEVES)

    px = live_prices(list(targets))
    missing = [t for t in targets if t not in px]
    if missing:
        sys.exit(f"ERROR: no live price for {missing}; refusing to record a book with "
                 f"guessed prices.")

    # WHOLE SHARES ONLY — you cannot buy a fraction. Leftover stays as real cash.
    # Quantities come from largest-remainder apportionment, not naive floor: see
    # allocate(). Same capital and same names, ~6.6pp closer to the target book.
    alloc = allocate(targets, px, capital)
    positions, rows, spent = {}, [], 0.0
    for t, target_w in sorted(targets.items(), key=lambda kv: -kv[1]):
        budget = capital * target_w
        qty = alloc.get(t, 0)
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
                        "entry_cost": cost, "target_weight": target_w,
                        "entry_date": str(pd.Timestamp.today().date())}
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


def net_fy_tax(book) -> float:
    """Tax owed on gains realised so far this fiscal year, AFTER netting losses.

    Indian law (Sec 70/74) nets losses against gains within the FY: STCL offsets
    STCG then LTCG; LTCL offsets LTCG only. The backtest engine has modelled this
    since P11; this is the same rule applied to the live book, so the live record
    and the research record use the same tax law rather than two different ones.

    The result is a LIABILITY. It is money the book owes and cannot spend, so it
    is deducted from NAV — otherwise every rebalance that books a gain would make
    the track record look better than it is, permanently and invisibly.
    """
    st = book.get("fy_stcg", 0.0)
    lt = book.get("fy_ltcg", 0.0)
    lt_112a = book.get("fy_ltcg_112a", 0.0)        # equity-sourced part of `lt`
    stl = max(0.0, -st) + book.get("cf_stcl", 0.0)
    ltl = max(0.0, -lt) + book.get("cf_ltcl", 0.0)
    st, lt = max(0.0, st), max(0.0, lt)
    use = min(stl, st); st -= use; stl -= use      # STCL vs STCG
    use = min(stl, lt); lt -= use; stl -= use      # then vs LTCG
    use = min(ltl, lt); lt -= use                  # LTCL vs LTCG only
    # Sec 112A: the first Rs 1.25 lakh of long-term gain on LISTED INDIAN EQUITY is
    # exempt each fiscal year. It does NOT cover the gold or US-Nasdaq ETF sleeves,
    # so only the equity-sourced share of the surviving long-term gain qualifies;
    # exempting the whole figure would understate the bill. At this book's size the
    # exemption is worth roughly +0.6pp/yr, which is why the scale-free headline
    # UNDERSTATES what a small book actually keeps.
    exempt = min(LTCG_EXEMPTION, max(0.0, lt_112a), lt)
    return st * STCG + max(0.0, lt - exempt) * LTCG


def _mark(book):
    px = live_prices(list(book["positions"]))
    mv, detail = book.get("cash", 0.0) - net_fy_tax(book), []
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
    ca = reconcile_corporate_actions(book)
    if ca and not quiet:
        print(f"  corporate actions applied: {', '.join(ca)}")
    nav, detail = _mark(book)
    bench = benchmark_value(book["capital"], book["start_date"])
    ret = nav / book["capital"] - 1
    days = (pd.Timestamp.today().normalize()
            - pd.Timestamp(book["start_date"]).normalize()).days
    if not quiet:
        print(f"\n  PAPER BOOK — day {days} since {book['start_date']}")
        print(f"  NAV Rs {nav:,.0f}  ({ret*100:+.2f}%)   cash Rs {book.get('cash',0):,.0f}")
        if bench:
            print(f"  Nifty on the same rupees: Rs {bench:,.0f} "
                  f"({(bench/book['capital']-1)*100:+.2f}%)  ->  "
                  f"relative {ret*100-(bench/book['capital']-1)*100:+.2f}pp")
        print(f"  {'ticker':<14}{'qty':>6}{'entry':>10}{'now':>10}{'value':>12}{'P&L %':>9}")
        for d in detail:
            print(f"  {d['ticker']:<14}{d['qty']:>6}{d['entry']:>10.2f}{d['price']:>10.2f}"
                  f"{d['value']:>12,.0f}{d['pnl_pct']:>+9.1f}")
    # One row per TRADING SESSION, dated by the session — not one per calendar day.
    #
    # The old rule was "one honest row per calendar day", and it was not honest on
    # a day the market was shut. _mark() prices the book from the last non-NaN bar
    # in a 7-day window and never says WHEN that bar was, so running this on a
    # Sunday writes Friday's closes under Sunday's date, and on Diwali it writes
    # the previous session's closes as if they were the holiday's. Both are
    # invented sessions in a record whose entire value is that nothing invented is
    # in it (Mandate §6). Proven the hard way on 2026-08-09: a manual `export` on
    # a Sunday appended a 2026-08-09 row carrying Friday's prices.
    #
    # The weekday cron hid it — trading holidays are weekdays, so the daily job
    # would have written the same thing on the next NSE holiday, unattended.
    #
    # Keying on the session date also makes this idempotent: re-running any number
    # of times can only ever produce the row that session already has.
    sess = last_session_date()
    if sess is None:
        if not quiet:
            print("  could not confirm the last NSE session — not recording a mark")
        return book, nav, ret, days, detail, bench
    stamp = str(sess.date())
    days = (sess.normalize() - pd.Timestamp(book["start_date"]).normalize()).days
    seen = set()
    if os.path.exists(NAV_LOG):
        seen = {r.split(",")[0] for r in open(NAV_LOG).read().splitlines()[1:]}
    if stamp not in seen:
        new = not os.path.exists(NAV_LOG)
        with open(NAV_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "day", "nav_inr", "return_pct", "bench_inr",
                            "bench_return_pct", "timestamp"])
            br = (bench / book["capital"] - 1) * 100 if bench else None
            w.writerow([stamp, days, f"{nav:.2f}", f"{ret*100:.4f}",
                        f"{bench:.2f}" if bench else "",
                        f"{br:.4f}" if br is not None else "", now_iso()])
    return book, nav, ret, days, detail, bench


PASSIVE = {"GOLDBEES": "Gold ETF", "MON100": "US Nasdaq-100"}


def sleeve_attribution(book, detail, nav_now):
    """Split the headline into its three sleeves. Real money, two honest measures.

    The blended headline hides the only thing worth knowing: half the book is two
    passive ETFs anyone can buy, and only the other half is the system. When Nifty
    rallies and both diversifiers fall, the total looks broken while nothing is.

    Two DIFFERENT measures, because they answer different questions and disagree:

      P&L / NAV impact — cash-on-cash: current value minus every rupee put in,
        including brokerage. Exact by construction, since
            capital = SUM(net invested) + cash   and   NAV = SUM(value) + cash
        so the sleeve impacts sum to the headline return with nothing left over.
        This is "what did it do to my money".

      Return % — TIME-WEIGHTED, chain-linked across rebalances. This is "how did
        it perform", and it is the only one comparable to an index. The naive
        holdings-vs-entry figure is NOT used: a rebalance resets entry prices and
        sweeps idle cash in, and counting a deposit as profit flattered the equity
        sleeve by +0.74pp on 2026-08-03 (RESEARCH_LOG 5a). Reconstructed as
            equity value = NAV - cash - ETF quantities x ETF closes
        from the append-only ledger, so it needs only two price series and stays
        exact however many names the equity sleeve has traded.
    """
    led = list(csv.DictReader(open(LEDGER))) if os.path.exists(LEDGER) else []
    if not led:
        return None
    cap = book["capital"]

    # cash-on-cash: net rupees committed per sleeve, brokerage included
    invested = {}
    for r in led:
        k = PASSIVE.get(r["ticker"], "eq")
        sgn = 1 if r["action"] == "BUY" else -1
        invested[k] = invested.get(k, 0.0) + sgn * float(r["value_inr"]) + float(r["cost_inr"])

    value = {}
    for h in detail:
        k = PASSIVE.get(h["ticker"], "eq")
        value[k] = value.get(k, 0.0) + h["value"]

    twr = _sleeve_twr(book, led)
    labels = {"eq": "Indian equity", "Gold ETF": "Gold ETF",
              "US Nasdaq-100": "US Nasdaq-100"}
    rows = []
    for k in ("eq", "Gold ETF", "US Nasdaq-100"):
        val, inv = value.get(k, 0.0), invested.get(k, 0.0)
        rows.append({
            "key": k, "label": labels[k], "passive": k != "eq",
            "role": ("Factor-ranked stock selection — this is the system"
                     if k == "eq" else "Passive ETF — anyone can buy this"),
            "n_holdings": sum(1 for h in detail if PASSIVE.get(h["ticker"], "eq") == k),
            "value_inr": val, "invested_inr": inv,
            "weight_pct": val / nav_now * 100 if nav_now else None,
            "pnl_inr": val - inv,
            "nav_impact_pp": (val - inv) / cap * 100,
            "return_pct": twr.get(k),
        })
    return {"rows": rows, "cash_inr": book.get("cash", 0.0),
            "total_pnl_inr": sum(r["pnl_inr"] for r in rows),
            "total_impact_pp": sum(r["nav_impact_pp"] for r in rows),
            "method": ("NAV impact is cash-on-cash and sums exactly to the headline. "
                       "Return % is time-weighted and chain-linked across rebalances, "
                       "so cash swept in at a rebalance is never counted as profit.")}


def _sleeve_twr(book, led):
    """{sleeve -> time-weighted return %} over the live window, flow-neutral."""
    import yfinance as yf
    if not os.path.exists(NAV_LOG):
        return {}
    hist = [r for r in csv.DictReader(open(NAV_LOG)) if r.get("nav_inr")]
    if len(hist) < 2:
        return {}
    dates = [pd.Timestamp(r["date"]) for r in hist]
    etf = list(PASSIVE)
    try:
        px = yf.download([f"{t}.NS" for t in etf], start=book["start_date"],
                         end=str((dates[-1] + pd.Timedelta(days=2)).date()),
                         auto_adjust=True, progress=False, threads=False)["Close"]
        px.index = px.index.tz_localize(None)
    except Exception:
        return {}

    qty, cash, states = {t: 0 for t in etf}, float(book["capital"]), {}
    for d in sorted({r["date"] for r in led}):
        for r in led:
            if r["date"] != d:
                continue
            sgn = 1 if r["action"] == "BUY" else -1
            cash -= sgn * float(r["value_inr"]) + float(r["cost_inr"])
            if r["ticker"] in etf:
                qty[r["ticker"]] += sgn * int(r["qty"])
        states[pd.Timestamp(d)] = (dict(qty), cash)
    keys = sorted(states)

    series = {"eq": [], **{PASSIVE[t]: [] for t in etf}}
    for i, d in enumerate(dates):
        q, csh = states[max(k for k in keys if k <= d)]
        tot = 0.0
        for t in etf:
            try:
                p = float(px[f"{t}.NS"].reindex([d], method="ffill").iloc[0])
            except Exception:
                return {}
            v = q[t] * p
            series[PASSIVE[t]].append(v)
            tot += v
        series["eq"].append(float(hist[i]["nav_inr"]) - csh - tot)

    # chain-link across rebalance dates — the only points cash enters a sleeve
    breaks = sorted({pd.Timestamp(r["date"]) for r in book.get("rebalances", [])}
                    & set(dates))
    cuts = [0] + [dates.index(b) for b in breaks] + [len(dates)]
    out = {}
    for k, s in series.items():
        r = 1.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - 1 <= a or not s[a]:
                continue
            r *= s[b - 1] / s[a]
        out[k] = (r - 1) * 100
    return out


def rebalance_events(book) -> list[dict]:
    """Every reconstitution of the book, with the ones that fired early flagged.

    An off-cadence rebalance resets entry prices and re-picks names mid-flight, so
    the window either side of it is not one continuous test of one book. That has
    to be visible on the page, not buried in the diagnosis JSON.
    """
    start = pd.Timestamp(book["start_date"])
    out = []
    for r in book.get("rebalances", []):
        age = (pd.Timestamp(r["date"]) - start).days
        out.append({"date": r["date"], "signal_asof": r.get("signal_asof"),
                    "trades": r.get("trades", 0),
                    "realised_pnl": r.get("realised_pnl", 0.0),
                    "day_of_book": age,
                    "off_cadence": age < REBAL_DAYS})
    return out


def cmd_export():
    """Emit the JSON the public dashboard reads. Real data only."""
    book, nav, ret, days, detail, bench = cmd_status(quiet=True)
    hist = []
    if os.path.exists(NAV_LOG):
        hist = list(csv.DictReader(open(NAV_LOG)))
    navs = [float(r["nav_inr"]) for r in hist if r.get("nav_inr")]
    peak, mdd = 0.0, 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    bench_ret = (bench / book["capital"] - 1) * 100 if bench else None
    # The rules, published alongside the numbers they produced. Everything that
    # reads this file — the dashboard, the bot's position notes — otherwise has to
    # hardcode its own copy of the config, and a second copy is a copy that
    # disagrees eventually. A caption that describes a different portfolio from
    # the one in the table is the failure Mandate §6 names.
    out_config = {"n_hold": N_HOLD, "max_weight_per_name": MAX_WEIGHT,
                  "top_n_liquid": TOP_N, "cadence_days": REBAL_DAYS,
                  "factor_weights": FACTOR_WEIGHTS,
                  "sleeve_targets": SLEEVES,
                  "sizing": "inverse volatility within the equity sleeve"}
    out = {"generated": now_iso(), "mode": book["mode"], "config": out_config,
           "start_date": book["start_date"], "days_live": days,
           "capital": book["capital"], "nav": nav, "return_pct": ret * 100,
           "cash": book.get("cash", 0), "integrity": book.get("integrity"),
           "benchmark_nav": bench, "benchmark_return_pct": bench_ret,
           "relative_pct": (ret * 100 - bench_ret) if bench_ret is not None else None,
           "max_drawdown_pct": mdd * 100, "observations": len(navs),
           # tax owed on gains realised this fiscal year, after netting losses.
           # Already deducted from `nav` above — surfaced so the page can show
           # that the headline is net of it rather than leaving it implicit.
           "tax_liability": net_fy_tax(book),
           "realised_pnl": book.get("realised_pnl", 0.0),
           "rebalances": len(book.get("rebalances", [])),
           # Every reconstitution, dated. A track record that says "append-only
           # ledger" while silently hiding that the book was rebuilt on day 4 is
           # the same failure it claims to defend against, so the events ship
           # with the data and the page can render them on the NAV chart.
           "rebalance_events": rebalance_events(book),
           # the blended headline hides that half the book is passive ETFs
           "sleeves": sleeve_attribution(book, detail, nav),
           "holdings": detail, "nav_history": hist}
    path = os.path.join(PAPER_DIR, "paper_export.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"  wrote {path}  (day {days}, NAV Rs {nav:,.0f}, {ret*100:+.2f}%)")


SELL_COST_RATE = 0.001 + 0.0000297 + 0.000001 + 0.18 * (0.0000297 + 0.000001)  # STT+txn+SEBI+GST
REBAL_DAYS = 182          # the deployed cadence is 126 trading days ~ 6 calendar months
LTCG, STCG = 0.125, 0.20
LTCG_EXEMPTION = 125000.0   # Sec 112A, per person per fiscal year, listed Indian equity only


def cmd_rebalance_check() -> bool:
    """Is a rebalance due today? Answered from the book alone — no price cache.

    This exists because of the shape of the failure it prevents. The rebalance
    needs a ~200-name price panel that is gitignored and therefore ABSENT in CI;
    building it costs minutes, which is unacceptable in a job that runs daily and
    rebalances twice a year. So the daily job asks this question first, and only
    pays for the data on the one day the answer is yes.

    Getting this wrong is expensive and silent: the first rebalance falls due in
    January, and without this the job would have discovered its missing cache
    then — unattended, months after anyone was watching, with the mark-and-commit
    steps sitting behind the failure.

    Prints DUE / NOT-DUE and exits 0 either way; the workflow branches on stdout.
    """
    if not os.path.exists(BOOK):
        print("NOT-DUE  (no book yet)")
        return False
    book = json.load(open(BOOK))
    last = pd.Timestamp(book.get("last_rebalance", book["start_date"]))
    due_in = REBAL_DAYS - (pd.Timestamp.today().normalize() - last.normalize()).days
    weekend = pd.Timestamp.today().weekday() >= 5
    if due_in > 0 or weekend:
        # A third state, a week out. The rebalance is the only scheduled event in
        # a six-month absence and it used to arrive with no warning at all. On a
        # PREVIEW day the workflow does the full preparation — new listings,
        # sector labels, prices, ranking — WITHOUT trading. That makes the
        # heads-up message accurate, and it rehearses the machinery while there
        # is still a week to fix whatever it breaks on.
        state = ("PREVIEW" if 0 < due_in <= 7 and not weekend else "NOT-DUE")
        print(f"{state}  ({due_in}d to go" + (", weekend" if weekend else "") +
              f", next ~{(last + pd.Timedelta(days=REBAL_DAYS)).date()})")
        return False
    print(f"DUE  (last rebalance {last.date()}, cadence {REBAL_DAYS}d)")
    return True


def cmd_rebalance(force=False):
    """Refresh the equity book at real prices, logging every trade.

    The deployed system refreshes semi-annually. Without this the paper book would
    hold its opening names forever and stop being a test of the actual system.
    Sells realise P&L and accrue real tax; every leg is appended to the ledger.
    Nothing is ever rewritten — a rebalance ADDS rows, it does not replace history.
    """
    if not os.path.exists(BOOK):
        sys.exit("ERROR: no paper book — run 'init' first.")
    book = json.load(open(BOOK))
    last = pd.Timestamp(book.get("last_rebalance", book["start_date"]))
    due = (pd.Timestamp.today().normalize() - last.normalize()).days
    if due < REBAL_DAYS and not force:
        print(f"  not due — {due}d since last refresh, cadence is {REBAL_DAYS}d "
              f"(next ~{(last + pd.Timedelta(days=REBAL_DAYS)).date()}). Use --force to override.")
        return
    # --force overrides the CADENCE, never the market calendar. Two forced
    # rebalances once fired on a Sunday against Friday's closes: the ledger then
    # records fills at prices nobody could have traded at. There is no reason to
    # ever want that, so this guard has no override.
    # Exit-code contract, now that CI runs this unattended every weekday:
    #   0 = nothing to do, or done. 1 = something is actually wrong.
    # "The market is shut" is the guard SUCCEEDING, not a failure. Signalling it
    # as an error would fail the whole refresh job, and the steps that record the
    # day's mark and commit it come after — so a correctly-declined rebalance
    # would punch a hole in the append-only record it exists to protect.
    # The refusal itself is unchanged and still has no override.
    if pd.Timestamp.today().weekday() >= 5:
        print("  declined — market closed today (weekend); every quote is stale. "
              "Will retry on the next session.")
        return
    # …and the same for trading holidays, which are weekdays. Declining cannot
    # block permanently: `due` still exceeds the cadence tomorrow, so it simply
    # happens on the next real session.
    sess = last_session_date()
    if sess is None or sess != pd.Timestamp.today().normalize():
        print(f"  declined — no NSE session printed today (last close "
              f"{sess.date() if sess is not None else 'unknown'}). Refusing to book "
              f"fills at a price nobody could have traded at. Will retry next session.")
        return
    reconcile_corporate_actions(book)
    gap = assert_cache_covers_holdings(book)
    if gap:
        # DECLINE, do not fail. The exit-code contract is "0 = nothing to do, or
        # done; 1 = something is actually wrong" — but the consequence of exiting
        # 1 here is worse than the thing it reports. This step runs BEFORE the
        # mark-and-commit steps, so a non-zero exit strands them, and `due` stays
        # true every subsequent day: one bad cache would stop the append-only
        # record for as long as nobody looked. That is the exact hole the weekend
        # and holiday guards return early to avoid.
        #
        # Silence is handled elsewhere: healthcheck.py fails when a rebalance is
        # overdue, so the daily message says so instead of this being invisible.
        print(f"  declined — {gap}\n"
              f"  A name the cache cannot see is a name the ranking drops, and a "
              f"dropped name is SOLD. Rebuilding the cache and retrying on the "
              f"next session is the safe move; the cadence is still due.")
        return

    w_eq, asof, _, signals = target_book()
    eq_frac = 1 - sum(SLEEVES.values())
    targets = {t: float(x) * eq_frac for t, x in w_eq.items()}
    targets.update(SLEEVES)

    held = list(book["positions"])
    px = live_prices(sorted(set(held) | set(targets)))
    missing = [t for t in set(held) | set(targets) if t not in px]
    if missing:
        sys.exit(f"ERROR: no live price for {missing}; refusing to rebalance blind.")

    nav = book.get("cash", 0.0) + sum(p["qty"] * px[t] for t, p in book["positions"].items())
    rows, cash = [], book.get("cash", 0.0)
    today = str(pd.Timestamp.today().date())
    realised, tax_accrued = 0.0, 0.0

    # ── exits first, so their proceeds fund the entries ──────────────────────
    for t in held:
        if t in targets:
            continue
        p = book["positions"][t]
        gross = p["qty"] * px[t]
        cost = gross * SELL_COST_RATE
        pnl = gross - p["entry_value"] - p["entry_cost"] - cost
        days_held = (pd.Timestamp.today().normalize()
                     - pd.Timestamp(p.get("entry_date", book["start_date"])).normalize()).days
        rate = LTCG if days_held > 365 else STCG
        # accrue the GAIN (signed) to its fiscal-year bucket; the tax is computed
        # on the netted total, not per trade — a loss here really does reduce the
        # bill, which is what Indian law says and what the backtest engine models.
        bucket = "fy_ltcg" if days_held > 365 else "fy_stcg"
        book[bucket] = book.get(bucket, 0.0) + pnl
        # track how much of the long-term gain is Sec 112A-eligible (listed Indian
        # equity). The ETF sleeves are taxed under different provisions and get no
        # Rs 1.25L exemption, so they must not be counted toward it.
        if days_held > 365 and t not in SLEEVES:
            book["fy_ltcg_112a"] = book.get("fy_ltcg_112a", 0.0) + pnl
        tax = max(0.0, pnl) * rate
        realised += pnl
        tax_accrued += tax
        cash += gross - cost
        rows.append({"timestamp": now_iso(), "date": today, "action": "SELL", "ticker": t,
                     "qty": p["qty"], "price": f"{px[t]:.2f}", "value_inr": f"{gross:.2f}",
                     "cost_inr": f"{cost:.2f}",
                     "note": f"exit · P&L {pnl:+.0f} · held {days_held}d · "
                             f"{'LTCG' if rate == LTCG else 'STCG'} accrued {tax:.0f}"})
        del book["positions"][t]

    # ── entries / top-ups, whole shares, cash-constrained ───────────────────
    for t, wt in sorted(targets.items(), key=lambda kv: -kv[1]):
        want_val = nav * wt
        have = book["positions"].get(t, {"qty": 0})["qty"]
        want_qty = int(want_val // px[t])
        delta = want_qty - have
        if delta > 0:
            spend = delta * px[t]
            cost = spend * BUY_COST_RATE
            if spend + cost > cash:                      # never spend money we do not have
                delta = int(max(0, cash / (px[t] * (1 + BUY_COST_RATE))))
                spend, cost = delta * px[t], delta * px[t] * BUY_COST_RATE
            if delta <= 0:
                continue
            cash -= spend + cost
            pos = book["positions"].setdefault(
                t, {"qty": 0, "entry_price": px[t], "entry_value": 0.0, "entry_cost": 0.0,
                    "target_weight": wt, "entry_date": today})
            pos["entry_value"] += spend; pos["entry_cost"] += cost
            pos["qty"] += delta
            pos["entry_price"] = pos["entry_value"] / pos["qty"]
            pos["target_weight"] = wt
            rows.append({"timestamp": now_iso(), "date": today, "action": "BUY", "ticker": t,
                         "qty": delta, "price": f"{px[t]:.2f}", "value_inr": f"{spend:.2f}",
                         "cost_inr": f"{cost:.2f}", "note": "rebalance entry"})
        elif delta < 0 and want_qty >= 0:
            p = book["positions"][t]
            gross = -delta * px[t]; cost = gross * SELL_COST_RATE
            frac = -delta / p["qty"]
            pnl = gross - p["entry_value"] * frac - p["entry_cost"] * frac - cost
            realised += pnl; tax_accrued += max(0.0, pnl) * STCG
            book["fy_stcg"] = book.get("fy_stcg", 0.0) + pnl
            cash += gross - cost
            p["entry_value"] *= (1 - frac); p["entry_cost"] *= (1 - frac); p["qty"] = want_qty
            rows.append({"timestamp": now_iso(), "date": today, "action": "SELL", "ticker": t,
                         "qty": -delta, "price": f"{px[t]:.2f}", "value_inr": f"{gross:.2f}",
                         "cost_inr": f"{cost:.2f}", "note": f"trim · P&L {pnl:+.0f}"})

    # Residual-cash sweep: the loop above floors every entry, so it always leaves
    # cash idle. Spend it one share at a time on whichever name sits furthest below
    # its target — same largest-remainder logic as allocate(), applied to a book
    # that already holds positions. Without this the live book drifts further below
    # target at every rebalance instead of converging on it.
    for _ in range(10000):
        best, best_short = None, 0.0
        for t, wt in targets.items():
            p = px.get(t)
            if not p or p <= 0 or p * (1 + BUY_COST_RATE) > cash + 1e-9:
                continue
            short = nav * wt - book["positions"].get(t, {"qty": 0})["qty"] * p
            if short > best_short:
                best, best_short = t, short
        if best is None:
            break
        p = px[best]
        cost = p * BUY_COST_RATE
        cash -= p + cost
        pos = book["positions"].setdefault(
            best, {"qty": 0, "entry_price": p, "entry_value": 0.0, "entry_cost": 0.0,
                   "target_weight": targets[best], "entry_date": today})
        pos["entry_value"] += p; pos["entry_cost"] += cost; pos["qty"] += 1
        pos["entry_price"] = pos["entry_value"] / pos["qty"]
        rows.append({"timestamp": now_iso(), "date": today, "action": "BUY",
                     "ticker": best, "qty": 1, "price": f"{p:.2f}",
                     "value_inr": f"{p:.2f}", "cost_inr": f"{cost:.2f}",
                     "note": "residual-cash sweep"})

    book["cash"] = cash
    book["last_rebalance"] = today
    book["realised_pnl"] = book.get("realised_pnl", 0.0) + realised
    book["tax_accrued"] = book.get("tax_accrued", 0.0) + tax_accrued
    book.setdefault("rebalances", []).append(
        {"date": today, "signal_asof": str(asof.date()), "trades": len(rows),
         "realised_pnl": realised, "tax_accrued": tax_accrued})
    json.dump(book, open(BOOK, "w"), indent=1)
    append_ledger(rows)
    write_signals(signals, f"recorded at the {today} rebalance — these are the "
                           f"numbers that chose this book")
    print(f"  REBALANCED {today}: {len(rows)} trades · realised P&L Rs {realised:+,.0f} · "
          f"tax accrued Rs {tax_accrued:,.0f} · cash Rs {cash:,.0f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--capital", type=float, default=500000)
    sub.add_parser("status"); sub.add_parser("export")
    pr = sub.add_parser("rebalance")
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--check", action="store_true",
                    help="print DUE/NOT-DUE from the book alone; touches no price data")
    sub.add_parser("signals", help="recompute today's ranking and record it "
                                   "(labelled as recomputed, not as the pick reasoning)")
    a = ap.parse_args()
    if a.cmd == "init":
        if not (0 < a.capital <= 1e12):
            sys.exit("ERROR: --capital must be positive.")
        cmd_init(a.capital)
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd == "signals":
        # The live book was opened before any of this was recorded, so there is
        # nothing to explain it with until the next rebalance. This fills the gap
        # honestly: it answers "how does the system rank this name TODAY", which
        # is a true statement and arguably the more useful one for anyone deciding
        # now — but it is NOT why the name was bought, and the basis string says so.
        # Same guard as the rebalance. A snapshot computed while names are missing
        # would score the survivors against a field that accidentally lost its
        # peers — a measurement artefact presented as the system's reasoning,
        # which is worse than having no snapshot at all.
        gap = assert_cache_covers_holdings(json.load(open(BOOK)))
        if gap:
            sys.exit(f"ERROR: {gap}\n  Refusing to score names against a field "
                     f"that is accidentally missing its peers.")
        _, _, _, sig = target_book()
        write_signals(sig, f"recomputed {pd.Timestamp.today().date()} — how these "
                           f"names rank TODAY, not the numbers used to pick them")
    elif a.cmd == "rebalance":
        if a.check:
            cmd_rebalance_check()
        else:
            cmd_rebalance(a.force)
    else:
        cmd_export()


if __name__ == "__main__":
    main()
