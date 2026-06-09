"""
Fetch DEEP historical shareholding (Promoter / FII / DII / Public) from the
OFFICIAL NSE corporate-filings archive — FREE, no subscription, ~32 quarters
back to mid-2018 (vs screener.in free tier's ~12 quarters).

Pipeline per ticker:
  1. GET nseindia.com/api/corporate-share-holdings-master  -> list of every
     quarterly filing with as-on date, promoter/public split, the actual
     BROADCAST (public-disclosure) date, and an XBRL link.
  2. For each filing, download the XBRL and parse the institutional breakdown
     (FII = InstitutionsForeign, DII = InstitutionsDomestic) from the summary
     statement contexts.

Zero look-ahead: we store the real `broadcastDate` (when the filing actually
became public), not an assumed lag. Resumable; caches to
data/cache/shareholding_nse/{TICKER}.json in a screener-compatible schema plus
`dates`/`disclosure` arrays.

Run:  python3 scripts/fetch_shareholding_nse.py            # full universe
      python3 scripts/fetch_shareholding_nse.py HAL TRENT  # specific tickers
"""
import os
import re
import sys
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

XBRL_WORKERS = 12   # NSE archive is per-IP throttled; ~12 is the sweet spot (~2x)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio.universe import discover_tickers

OUT = os.path.join(_ROOT, "data", "cache", "shareholding_nse")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
MASTER = ("https://www.nseindia.com/api/corporate-share-holdings-master"
          "?index=equities&symbol={sym}")
SHP_REF = "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"

MON = {"JAN": "Mar", "MAR": "Mar", "JUN": "Jun", "SEP": "Sep", "DEC": "Dec"}
MON_NUM = {"JAN": "03", "MAR": "03", "JUN": "06", "SEP": "09", "DEC": "12"}
DAY = {"03": "31", "06": "30", "09": "30", "12": "31"}

# SEBI shareholding XBRL has TWO taxonomy generations with different context IDs
# and different value scales (older = percent like 75.15; newer = fraction 0.7515).
# We list accepted context IDs per field (newest first) and self-calibrate scale
# from Promoter+Public (which always partitions the whole -> 100% or 1.0).
PCT_TAG = "ShareholdingAsAPercentageOfTotalNumberOfShares"
# Three taxonomy generations seen: suffix `I` (≤2022 and 2022-2025) and `_ContextI`
# (2025+). List accepted context IDs per field, TOTAL contexts first so we never grab a
# sub-category by mistake. DII/FII totals are then cross-checked against Institutions.
CTX = {
    "Promoters": ["ShareholdingOfPromoterAndPromoterGroup_ContextI",
                  "ShareholdingOfPromoterAndPromoterGroupI"],
    "Public":    ["PublicShareholding_ContextI", "PublicShareholdingI"],
    "FIIs":      ["InstitutionsForeign_ContextI", "InstitutionsForeignI",
                  "InstitutionsForeignPortfolioInvestorI", "ForeignPortfolioInvestorI"],
    "DIIs":      ["InstitutionsDomestic_ContextI", "InstitutionsDomesticI"],
    "Institutions": ["Institutions_ContextI", "InstitutionsI"],  # total; else summed
}


class NSE:
    """Thin NSE client with cookie warm-up and transparent refresh."""

    def __init__(self):
        self.cj = urllib.request.HTTPCookieProcessor()
        self.op = urllib.request.build_opener(self.cj)
        self.warm()

    def warm(self):
        for url in ("https://www.nseindia.com/", SHP_REF):
            try:
                self._get(url, ref="https://www.nseindia.com/", raw=True)
            except Exception:
                pass
            time.sleep(0.3)

    def _get(self, url, ref=SHP_REF, raw=False, timeout=20):
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Accept": "application/json, text/plain, */*" if not raw else "text/html,*/*",
            "Referer": ref,
            "Accept-Language": "en-US,en;q=0.9",
        })
        with self.op.open(req, timeout=timeout) as r:
            return r.read()

    def json(self, url):
        for attempt in range(3):
            try:
                body = self._get(url)
                return json.loads(body)
            except (urllib.error.HTTPError, json.JSONDecodeError, urllib.error.URLError):
                self.warm()
                time.sleep(1.0 + attempt)
        return None

    def xbrl(self, url):
        # nsearchives host: plain GET, no cookie needed. Use module urlopen (thread-safe
        # for independent GETs) so this can run inside a ThreadPoolExecutor.
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": UA, "Referer": "https://www.nseindia.com/"})
                with urllib.request.urlopen(req, timeout=25) as r:
                    return r.read().decode("utf-8", "ignore")
            except Exception:
                time.sleep(0.6 + attempt)
        return None


def _grab(xml: str, ctxs: list[str]):
    """First '% of total shares' fact across accepted context IDs (taxonomy-robust)."""
    for c in ctxs:
        m = re.search(
            r'<in-bse-shp:' + PCT_TAG + r'\s+[^>]*contextRef="'
            + re.escape(c) + r'"[^>]*>\s*([0-9.]+)', xml)
        if m:
            return float(m.group(1))
    return None


def parse_xbrl(xml: str) -> dict | None:
    """Return {Promoters, Public, FIIs, DIIs, Institutions} in PERCENT, or None."""
    promo = _grab(xml, CTX["Promoters"])
    public = _grab(xml, CTX["Public"])
    fii = _grab(xml, CTX["FIIs"])
    dii = _grab(xml, CTX["DIIs"])
    inst = _grab(xml, CTX["Institutions"])
    if promo is None or public is None:
        return None
    # self-calibrate scale: promoter+public is 100 (percent) or 1.0 (fraction)
    scale = 100.0 if (promo + public) < 2.0 else 1.0
    promo, public = promo * scale, public * scale
    fii = fii * scale if fii is not None else None
    dii = dii * scale if dii is not None else None
    inst = inst * scale if inst is not None else None
    # reconcile institutional totals across taxonomies (any two give the third)
    if inst is None and fii is not None and dii is not None:
        inst = fii + dii
    if dii is None and inst is not None and fii is not None:
        dii = round(inst - fii, 4)
    if fii is None and inst is not None and dii is not None:
        fii = round(inst - dii, 4)
    if inst is None and fii is not None and dii is None:
        inst = fii  # last resort (foreign only seen)
    # guard: a 0% institutional total on a liquid NSE name == parse failure, not truth.
    # Never emit a fake zero (it poisons the Δ signal). Drop the quarter instead.
    if inst is not None and inst <= 0.01:
        inst = fii = dii = None
    return {"Promoters": round(promo, 4), "Public": round(public, 4),
            "FIIs": round(fii, 4) if fii is not None else None,
            "DIIs": round(dii, 4) if dii is not None else None,
            "Institutions": round(inst, 4) if inst is not None else None}


def date_to_labels(d: str):
    """'31-MAR-2026' -> ('Mar 2026', '2026-03-31')."""
    dd, mon, yr = d.split("-")
    mon = mon.upper()
    return f"{MON.get(mon, mon.title())} {yr}", f"{yr}-{MON_NUM.get(mon,'03')}-{DAY.get(MON_NUM.get(mon,'03'),'31')}"


def fetch_one(nse: NSE, ticker: str) -> dict | None:
    master = nse.json(MASTER.format(sym=ticker))
    if not isinstance(master, list) or not master:
        return None
    # keep only XBRL-backed filings (institutional breakdown exists), newest dedup
    cand = []
    for r in master:
        d = r.get("date")
        xu = r.get("xbrl") or ""
        if not d or not xu.endswith(".xml"):
            continue
        try:
            iso = date_to_labels(d)[1]
        except Exception:
            continue
        cand.append((iso, r))
    # sort by REAL date asc; dedup quarter keeping latest-broadcast (revisions)
    cand.sort(key=lambda x: (x[0], x[1].get("recordId", "")))
    by_q = {}
    for iso, r in cand:
        by_q[iso] = r  # later (higher recordId) wins => uses revised filing
    rec = {"quarters": [], "dates": [], "disclosure": [], "Promoters": [],
           "FIIs": [], "DIIs": [], "Institutions": [], "Public": []}
    isos = sorted(by_q)
    # download all this ticker's XBRLs concurrently (per-IP throttled, ~2x faster)
    with ThreadPoolExecutor(max_workers=XBRL_WORKERS) as ex:
        xmls = dict(zip(isos, ex.map(lambda i: nse.xbrl(by_q[i]["xbrl"]), isos)))
    for iso in isos:
        xml = xmls.get(iso)
        if not xml:
            continue
        facts = parse_xbrl(xml)
        if not facts or facts.get("Institutions") is None:
            continue
        r = by_q[iso]
        rec["quarters"].append(date_to_labels(r["date"])[0])
        rec["dates"].append(iso)
        rec["disclosure"].append(_disc(r.get("broadcastDate"), iso))
        rec["Promoters"].append(facts["Promoters"])
        rec["Public"].append(facts["Public"])
        rec["FIIs"].append(facts["FIIs"])
        rec["DIIs"].append(facts["DIIs"])
        rec["Institutions"].append(facts["Institutions"])
    return rec if rec["quarters"] else None


def _disc(broadcast, iso):
    """'13-APR-2026 11:12:58' -> '2026-04-13'; fallback quarter-end + 45d marker."""
    if broadcast:
        m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{4})", broadcast)
        if m:
            mm = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05",
                  "JUN": "06", "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10",
                  "NOV": "11", "DEC": "12"}[m.group(2).upper()]
            return f"{m.group(3)}-{mm}-{m.group(1)}"
    return iso  # caller applies lag if disclosure missing


def main():
    os.makedirs(OUT, exist_ok=True)
    tickers = sys.argv[1:] if len(sys.argv) > 1 else discover_tickers()
    nse = NSE()
    ok = fail = skip = 0
    for k, t in enumerate(tickers):
        path = os.path.join(OUT, f"{t}.json")
        if os.path.exists(path):
            skip += 1
            continue
        try:
            rec = fetch_one(nse, t)
        except Exception as e:
            rec = None
            print(f"  {t}: error {e}", flush=True)
        if rec and any(v is not None for v in rec["Institutions"]):
            json.dump(rec, open(path, "w"))
            ok += 1
        else:
            json.dump({"quarters": [], "error": "no data"}, open(path, "w"))
            fail += 1
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(tickers)}  ok={ok} fail={fail} skip={skip}", flush=True)
            nse.warm()  # refresh cookies periodically
        time.sleep(0.4)
    print(f"DONE: ok={ok} fail={fail} skip={skip} of {len(tickers)}", flush=True)


if __name__ == "__main__":
    main()
