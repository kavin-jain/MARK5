"""
Daily health check for the live system and its public page.
===========================================================
Answers one question with evidence: is anything quietly broken?

"Quietly" is the important word. A crashed script is easy to notice; the failures
that actually matter here are silent — a feed that stops updating while still
serving yesterday's numbers, an unhandled stock split that turns into a permanent
fake loss, a ledger that stops being appended to. Each check below exists because
that specific failure would otherwise go unseen.

Exit code is 0 only if every check passes; 1 if anything FAILs. WARNs do not fail
the run but are printed for a human to judge.

  python3 scripts/healthcheck.py            # full check
  python3 scripts/healthcheck.py --json     # machine-readable summary
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(_ROOT, "data", "paper")
FEED = "https://kavin-jain.github.io/MARK5/data/mark6.json"
PAGE = "https://kavinjain.in/mark6"
UA = {"User-Agent": "MARK6-healthcheck/1.0"}

results = []


def check(name, ok, detail="", warn=False):
    results.append({"check": name, "status": "WARN" if (warn and not ok) else
                    ("PASS" if ok else "FAIL"), "detail": detail})
    icon = "✓" if ok else ("!" if warn else "✗")
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))
    return ok


def get(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    print(f"\nMARK6 health check — {datetime.now().astimezone().isoformat(timespec='seconds')}\n")

    # ── 1. public surfaces ───────────────────────────────────────────────
    print("PUBLIC SURFACES")
    feed = None
    try:
        r = get(FEED)
        feed = json.loads(r.read())
        check("data feed reachable", r.status == 200, f"HTTP {r.status}")
        check("feed sends CORS header",
              r.headers.get("access-control-allow-origin") == "*",
              r.headers.get("access-control-allow-origin") or "missing")
    except Exception as e:
        check("data feed reachable", False, f"{type(e).__name__}: {e}")
    try:
        check("page /mark6 serves", get(PAGE).status == 200)
    except Exception as e:
        check("page /mark6 serves", False, f"{type(e).__name__}: {e}")

    # ── 2. is the feed actually being refreshed? ─────────────────────────
    print("\nFRESHNESS  (the failure that looks like success)")
    if feed:
        gen = pd.Timestamp(feed.get("generated", "").replace("Z", "+00:00"))
        age_h = (pd.Timestamp.now(tz=gen.tz) - gen).total_seconds() / 3600
        # allow a long weekend: Fri evening -> Mon evening is ~72h
        check("feed refreshed recently", age_h < 96,
              f"{age_h:.0f}h old (generated {feed.get('generated','?')[:16]})", warn=age_h < 120)
        nh = feed.get("live", {}).get("nav_history", [])
        check("nav history is accumulating", len(nh) >= 1, f"{len(nh)} daily marks")
        dates = [r.get("date") for r in nh]
        check("no duplicate days in nav history", len(dates) == len(set(dates)),
              f"{len(dates) - len(set(dates))} duplicates")

    # ── 3. the scheduled job ─────────────────────────────────────────────
    print("\nAUTOMATION")
    try:
        out = subprocess.run(["gh", "run", "list", "--workflow", "refresh.yml",
                              "--limit", "3", "--json", "conclusion,createdAt,status"],
                             capture_output=True, text=True, cwd=_ROOT, timeout=60).stdout
        runs = json.loads(out) if out.strip() else []
        if not runs:
            check("refresh workflow has run", False,
                  "no runs yet — first fire is the next scheduled weekday 17:00 IST", warn=True)
        else:
            last = runs[0]
            check("last refresh run succeeded",
                  last.get("conclusion") in ("success", None),
                  f"{last.get('conclusion')} at {last.get('createdAt','')[:16]}")
    except Exception as e:
        check("refresh workflow queryable", False, f"{type(e).__name__}", warn=True)

    # ── 4. accounting identity on the live book ──────────────────────────
    print("\nLIVE BOOK INTEGRITY")
    bookp = os.path.join(PAPER, "paper_book.json")
    if os.path.exists(bookp) and feed:
        book = json.load(open(bookp))
        live = feed.get("live", {})
        holds = live.get("holdings", [])
        # NAV must equal cash + sum(position values). If this drifts, something is
        # being double-counted or dropped.
        recomputed = live.get("cash", 0) + sum(h["value"] for h in holds)
        drift = abs(recomputed - live.get("nav", 0))
        check("NAV = cash + positions", drift < 1.0, f"drift Rs {drift:.2f}")
        check("position count matches book",
              len(holds) == len(book.get("positions", {})),
              f"feed {len(holds)} vs book {len(book.get('positions', {}))}")
        check("no null prices in holdings",
              all(h.get("price") for h in holds),
              f"{sum(1 for h in holds if not h.get('price'))} null")
        stale = [h["ticker"] for h in holds if h.get("stale")]
        check("all holdings priced from live quotes", not stale,
              f"stale: {stale}" if stale else "", warn=True)
        # a >30% single-name move against entry, this early, almost always means an
        # unhandled corporate action rather than a real market move
        wild = [f"{h['ticker']} {h['pnl_pct']:+.0f}%" for h in holds
                if abs(h.get("pnl_pct", 0)) > 30]
        check("no suspicious single-name moves", not wild,
              f"check for unhandled splits: {wild}" if wild else "")
        has_bench = live.get("benchmark_return_pct") is not None
        check("benchmark present", has_bench,
              "" if has_bench else "missing — the vs-index number is the whole point")

    # ── 5. the record must only ever grow ────────────────────────────────
    print("\nRECORD DURABILITY  (nothing may be deleted)")
    led = os.path.join(PAPER, "paper_ledger.csv")
    nav = os.path.join(PAPER, "paper_nav.csv")
    for path, label in ((led, "ledger"), (nav, "nav log")):
        if not os.path.exists(path):
            check(f"{label} exists", False, path)
            continue
        rows = list(csv.DictReader(open(path)))
        check(f"{label} non-empty", len(rows) > 0, f"{len(rows)} rows")
    tracked = subprocess.run(["git", "ls-files", "data/paper/"], capture_output=True,
                             text=True, cwd=_ROOT).stdout.split()
    for f in ("data/paper/paper_ledger.csv", "data/paper/paper_nav.csv",
              "data/paper/paper_book.json"):
        check(f"{os.path.basename(f)} tracked in git", f in tracked,
              "untracked = lives on one machine only" if f not in tracked else "")
    # history must never shrink between runs
    hist_file = os.path.join(PAPER, ".healthcheck_watermark.json")
    marks = {"ledger": len(list(csv.DictReader(open(led)))) if os.path.exists(led) else 0,
             "nav": len(list(csv.DictReader(open(nav)))) if os.path.exists(nav) else 0}
    if os.path.exists(hist_file):
        prev = json.load(open(hist_file))
        shrunk = [k for k in marks if marks[k] < prev.get(k, 0)]
        check("record never shrinks", not shrunk,
              f"LOST ROWS in {shrunk}: {prev} -> {marks}" if shrunk else
              f"ledger {prev.get('ledger')}→{marks['ledger']}, nav {prev.get('nav')}→{marks['nav']}")
    else:
        check("watermark initialised", True, str(marks))
    json.dump(marks, open(hist_file, "w"))

    # ── verdict ──────────────────────────────────────────────────────────
    fails = [r for r in results if r["status"] == "FAIL"]
    warns = [r for r in results if r["status"] == "WARN"]
    print(f"\n{'─'*64}")
    print(f"  {len(results)} checks · {len(fails)} FAIL · {len(warns)} WARN")
    if fails:
        print("  FAILING:")
        for f in fails:
            print(f"    ✗ {f['check']} — {f['detail']}")
    print()
    if args.json:
        print(json.dumps({"generated": datetime.now(timezone.utc).isoformat(),
                          "fails": len(fails), "warns": len(warns),
                          "results": results}, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
