"""
Fetch per-stock quarterly shareholding (FII/DII/Promoter/Public) from screener.in
— the only FREE, reachable per-stock ownership source. NOTE: free tier exposes only
~12 recent quarters (≈2023-2026), which bounds what we can backtest. Resumable; caches
data/cache/shareholding/{TICKER}.json.
"""
import os, sys, re, json, time
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio.universe import discover_tickers

OUT = os.path.join(_ROOT, "data", "cache", "shareholding")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
LABELS = ["Promoters", "FIIs", "DIIs", "Government", "Public"]


def fetch(ticker: str) -> str | None:
    for variant in ("consolidated/", ""):
        url = f"https://www.screener.in/company/{ticker}/{variant}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                html = r.read().decode("utf-8", "ignore")
            if "Shareholding Pattern" in html:
                return html
        except Exception:
            continue
    return None


def parse(html: str) -> dict | None:
    i = html.find("Shareholding Pattern")
    if i < 0:
        return None
    seg = html[i:i + 9000]
    hdr = seg[:seg.find("Promoters")] if "Promoters" in seg else seg[:2000]
    quarters = re.findall(r"((?:Mar|Jun|Sep|Dec) 20[12][0-9])", hdr)
    if not quarters:
        return None
    out = {"quarters": quarters}
    for label in LABELS:
        j = seg.find(label)
        if j < 0:
            continue
        row = seg[j:seg.find("</tr>", j)]
        nums = [float(x) for x in re.findall(r"(-?\d+\.\d+)%", row)]
        if len(nums) >= len(quarters):
            out[label] = nums[:len(quarters)]
    return out if len(out) > 1 else None


def main():
    os.makedirs(OUT, exist_ok=True)
    tickers = discover_tickers()
    ok = fail = skip = 0
    for k, t in enumerate(tickers):
        path = os.path.join(OUT, f"{t}.json")
        if os.path.exists(path):
            skip += 1
            continue
        html = fetch(t)
        rec = parse(html) if html else None
        if rec:
            json.dump(rec, open(path, "w"))
            ok += 1
        else:
            json.dump({"quarters": [], "error": "no data"}, open(path, "w"))
            fail += 1
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(tickers)}  ok={ok} fail={fail} skip={skip}", flush=True)
        time.sleep(0.4)
    print(f"DONE: ok={ok} fail={fail} skip={skip} of {len(tickers)}")


if __name__ == "__main__":
    main()
