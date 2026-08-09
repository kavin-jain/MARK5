"""Refresh config/sector_map.json from NSE's own index constituent files.
=========================================================================
The sector cap is the only thing stopping the book from putting everything in
one industry. It is enforced by `PortfolioConstructor` ONLY for names it can
label, and sector_map.json's own note says the rest are "treated as their own
sector and therefore escape the cap".

On 2026-08-09 that was 30 of the 300 eligible names — and they were not evenly
spread. SEVEN of the top TWENTY were unlabelled, three of those capital-goods
companies (AEROFLEX, ASTRAMICRO, TDPOWERSYS). The cap was being evaded precisely
where it would otherwise have bound. A risk control that quietly switches itself
off for the highest-scoring names is worse than not having one, because the
report still says the cap is on.

The map had 500 entries and was built once, by hand, on 2026-07-22. Nothing
refreshed it, so every company listed since was unlabelled by construction — the
same "can only shrink" failure the price universe had.

WHY THESE FILES. NSE's quote API returns 403 to anything that is not a browser.
The index constituent CSVs are static archive files, carry an `Industry` column
straight from NSE's own classification, and together cover ~1000 names.

Never removes an existing label: a name that drops out of every index keeps the
sector it had. Sector membership barely moves, so a stale label is a far weaker
assumption than no label at all.

  python3 scripts/fetch_sector_map.py
"""
import csv
import io
import json
import os
import sys
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "config", "sector_map.json")

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
      "Accept": "*/*", "Referer": "https://www.nseindia.com/"}

# Ordered widest-first. Total Market is ~750 names and is the primary source;
# the rest fill the small end, where new listings live.
LISTS = ["ind_niftytotalmarket_list", "ind_nifty500list",
         "ind_niftymicrocap250_list", "ind_niftysmallcap250_list",
         "ind_niftymidcap150_list", "ind_niftynext50list"]


def fetch(name):
    url = f"https://nsearchives.nseindia.com/content/indices/{name}.csv"
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=45).read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw)))
    return {r["Symbol"].strip().upper(): r["Industry"].strip()
            for r in rows if r.get("Symbol") and r.get("Industry", "").strip()}


def main():
    try:
        old = json.load(open(OUT))
        existing = dict(old.get("sectors") or {})
    except (OSError, ValueError):
        old, existing = {}, {}
    print(f"existing map: {len(existing)} names")

    fetched, ok = {}, 0
    for name in LISTS:
        try:
            got = fetch(name)
            # First source wins: the widest list is the most authoritative, and
            # later files only fill gaps rather than relabel.
            for k, v in got.items():
                fetched.setdefault(k, v)
            ok += 1
            print(f"  {name:<32} {len(got):>4} names")
        except Exception as e:                               # noqa: BLE001
            print(f"  {name:<32} FAILED {type(e).__name__}: {str(e)[:60]}")

    if not ok:
        sys.exit("ERROR: every NSE list failed — refusing to rewrite the map with "
                 "nothing. The old labels are better than none.")

    # Never drop a label. A name absent from every index today keeps yesterday's
    # sector; membership is far more stable than listing status.
    merged = dict(existing)
    added = [k for k in fetched if k not in merged]
    changed = [k for k in fetched if k in merged and merged[k] != fetched[k]]
    merged.update(fetched)

    json.dump({"note": "NSE industry classification, from NSE's own index "
                       "constituent files. Sector membership is far more stable than "
                       "price, so applying today's labels historically is a much "
                       "weaker assumption than survivorship; unmapped names are "
                       "treated as their own sector and therefore escape the cap — "
                       "which is why this is refreshed before every rebalance.",
               "fetched": str(__import__("datetime").date.today()),
               "sources": LISTS[:ok],
               "count": len(merged),
               "sectors": dict(sorted(merged.items()))},
              open(OUT, "w"), indent=1)
    print(f"\n  {len(merged)} names  (+{len(added)} new, {len(changed)} relabelled)")

    # Coverage against what the book can actually choose from — the number that
    # decides whether the sector cap works.
    try:
        sig = json.load(open(os.path.join(_ROOT, "data", "paper", "signals.json")))
        elig = list(sig["scores"])
        miss = [t for t in elig if t not in merged]
        print(f"  eligible universe covered: {len(elig) - len(miss)}/{len(elig)}"
              f"  ({len(miss)} still unlabelled{': ' + ', '.join(miss[:8]) if miss else ''})")
    except (OSError, ValueError, KeyError):
        pass


if __name__ == "__main__":
    main()
