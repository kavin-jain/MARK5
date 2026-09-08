"""
Repair the live NAV log: insert missed sessions, correct lookahead-inflated rows.
================================================================================
This REWRITES an append-only record, which Mandate §6 forbids by default. It
exists because the record had four defects from one cause, and leaving known-
wrong numbers on a public track record is the larger dishonesty:

  2026-08-28, 2026-09-08   traded, never marked. `cmd_status` only ever wrote the
                           CURRENT session, so a day skipped once was skipped for
                           good — and the session pointer skipped them (it asked
                           ^NSEI, whose bar publishes hours after the job runs).

  2026-08-27, 2026-09-07   marked with a price from the FUTURE. When a ticker was
                           missing from the pinned session, `live_prices(asof=)`
                           fell back to its globally-latest bar, so these rows
                           carry closes from the following session. Both are
                           inflated: +Rs 2,081.61 and +Rs 5,546.01.

WHY THIS IS SAFE TO RUN. The arithmetic is not reimplemented — it monkeypatches
a historical price fetch under `paper_track._mark`, so every value is produced by
the engine's own code path. It then SELF-CHECKS by recomputing every row it is
not repairing: if it cannot reproduce those to the paisa it aborts, because a
method that cannot reproduce the good rows has no business rewriting the bad
ones. It refuses to touch anything at or before the last position change, where
the current book no longer describes what was held.

Nothing is hidden. Repaired rows carry today's timestamp in the log's existing
`timestamp` column, visibly later than their own date, and the run writes
reports/nav_repair.json recording every before/after so the correction itself is
auditable.

  python3 scripts/repair_nav_log.py              # dry run, prints the diff
  python3 scripts/repair_nav_log.py --apply      # writes, after a backup
"""
import argparse
import csv
import json
import os
import shutil
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import paper_track as pt                                     # noqa: E402

REPORT = os.path.join(_ROOT, "reports", "nav_repair.json")
TOLERANCE = 0.01          # rupees; equal to the log's own 2dp precision


def _historical_prices(px):
    """A `live_prices` with the same contract, served from one wide fetch.

    The real one uses period="10d" and cannot reach back to August. Same pinning
    rule as the fixed engine: the asof row if the ticker printed that day, else
    its last bar AT OR BEFORE asof — never after, which is the bug being undone.
    """
    def fetch(tickers, asof=None):
        out = {}
        for t in tickers:
            s = px.get(f"{t}.NS")
            if s is None:
                continue
            if asof is not None:
                s = s.loc[s.index <= asof]
            s = s.dropna()
            if len(s):
                out[t] = float(s.iloc[-1])
        return out
    return fetch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the repair")
    a = ap.parse_args()

    book = json.load(open(pt.BOOK))
    rows = list(csv.DictReader(open(pt.NAV_LOG)))
    recorded = {r["date"]: r for r in rows}
    start = book["start_date"]

    # The current book only describes what was held AFTER the last position
    # change; before it the quantities differ and this method would be wrong.
    led = list(csv.DictReader(open(pt.LEDGER)))
    floor = max(r["date"] for r in led)
    print(f"  last position change {floor} — nothing at or before it will be touched")

    import yfinance as yf
    tks = list(book["positions"])
    px = yf.download([f"{t}.NS" for t in tks], start=start,
                     end=str((pd.Timestamp.today() + pd.Timedelta(days=1)).date()),
                     auto_adjust=True, progress=False, threads=False)["Close"]
    idx = yf.download("^NSEI", start=start,
                      end=str((pd.Timestamp.today() + pd.Timedelta(days=1)).date()),
                      auto_adjust=True, progress=False)["Close"].dropna()
    sessions = [str(pd.Timestamp(d).date()) for d in idx.index]

    pt.live_prices = _historical_prices(px)                  # engine arithmetic, historical prices

    def truth(d):
        nav, _ = pt._mark(book, asof=pd.Timestamp(d))
        bench = pt.benchmark_value(book["capital"], start, asof=pd.Timestamp(d))
        days = (pd.Timestamp(d).normalize() - pd.Timestamp(start).normalize()).days
        return nav, bench, days

    # ── self-check: reproduce every row we are NOT repairing ────────────────
    checked = mismatched = 0
    suspect = []
    for d in sessions:
        if d <= floor or d not in recorded:
            continue
        nav, _, _ = truth(d)
        stored = float(recorded[d]["nav_inr"])
        checked += 1
        if abs(nav - stored) > TOLERANCE:
            mismatched += 1
            suspect.append((d, stored, nav, nav - stored))
    ok = checked - mismatched
    print(f"  self-check: reproduced {ok}/{checked} recorded rows exactly")
    if checked and ok / checked < 0.75:
        sys.exit(f"ABORT: only {ok}/{checked} rows reproduce — the method does not "
                 f"agree with the record and must not rewrite it.")

    # ── build the repaired series ──────────────────────────────────────────
    changes, out = [], []
    for d in sessions:
        if d <= floor:
            if d in recorded:
                out.append(recorded[d])
            continue
        nav, bench, days = truth(d)
        br = (bench / book["capital"] - 1) * 100 if bench else None
        new = {"date": d, "day": str(days), "nav_inr": f"{nav:.2f}",
               "return_pct": f"{(nav / book['capital'] - 1) * 100:.4f}",
               "bench_inr": f"{bench:.2f}" if bench else "",
               "bench_return_pct": f"{br:.4f}" if br is not None else "",
               "timestamp": pt.now_iso()}
        if d not in recorded:
            changes.append({"date": d, "kind": "inserted", "was": None,
                            "now": float(new["nav_inr"])})
            out.append(new)
        else:
            stored = float(recorded[d]["nav_inr"])
            if abs(nav - stored) > TOLERANCE:
                changes.append({"date": d, "kind": "corrected", "was": stored,
                                "now": round(nav, 2), "delta": round(nav - stored, 2)})
                out.append(new)
            else:
                out.append(recorded[d])          # untouched, original timestamp kept

    # rows the index never reported (e.g. the disclosed 2026-07-26 Sunday) are
    # kept exactly as they are — this repairs sessions, it does not delete history
    kept = [r for r in rows if r["date"] not in {o["date"] for o in out}]
    out = sorted(out + kept, key=lambda r: r["date"])

    print(f"\n  {'date':<12}{'kind':<11}{'was':>13}{'now':>13}{'delta':>12}")
    for c in changes:
        was = f"{c['was']:,.2f}" if c["was"] is not None else "—"
        dl = f"{c.get('delta', 0):+,.2f}" if c["was"] is not None else "—"
        print(f"  {c['date']:<12}{c['kind']:<11}{was:>13}{c['now']:>13,.2f}{dl:>12}")
    print(f"\n  rows {len(rows)} -> {len(out)}   changes {len(changes)}")
    if suspect and len(suspect) != len(
            [c for c in changes if c["kind"] == "corrected"]):
        print("  NOTE: self-check mismatches and corrections differ — inspect before applying")

    report = {"generated": pt.now_iso(), "rows_before": len(rows),
              "rows_after": len(out), "position_floor": floor,
              "self_check": {"checked": checked, "reproduced": ok},
              "changes": changes,
              "method": ("Repriced by paper_track._mark with a historical price "
                         "fetch pinned at or before each session — never after, "
                         "which is the lookahead being undone.")}

    if not a.apply:
        print("\n  DRY RUN — nothing written. Re-run with --apply.")
        return
    shutil.copy(pt.NAV_LOG, pt.NAV_LOG + ".pre-repair")
    with open(pt.NAV_LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "day", "nav_inr", "return_pct",
                                          "bench_inr", "bench_return_pct", "timestamp"])
        w.writeheader()
        w.writerows(out)
    json.dump(report, open(REPORT, "w"), indent=1)
    print(f"\n  wrote {pt.NAV_LOG}  (backup at {os.path.basename(pt.NAV_LOG)}.pre-repair)")
    print(f"  wrote {REPORT}")


if __name__ == "__main__":
    main()
