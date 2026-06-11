"""
Fetch historical fundamentals from indianapi.in (stock.indianapi.in) for the F3
quality factor. Per ticker, pulls ratios + yoy_results + balancesheet + cashflow
(annual, ~2015-2026) and caches data/cache/fundamentals/{TICKER}.json. Resumable,
rate-limited. Key read from .env at runtime (never hard-coded).

  python3 scripts/fetch_fundamentals.py            # full universe
  python3 scripts/fetch_fundamentals.py HAL INFY   # specific
"""
import os, sys, json, time, urllib.request, urllib.error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio.universe import discover_tickers

BASE = "https://stock.indianapi.in"
OUT = os.path.join(_ROOT, "data", "cache", "fundamentals")
STATS = ["ratios", "yoy_results", "balancesheet", "cashflow"]


def load_key():
    for v in ("INDIAN_API_KEY", "INDIANAPI_KEY", "API_KEY"):
        if os.getenv(v):
            return os.getenv(v).strip()
    for path in (os.path.join(_ROOT, ".env"), os.path.expanduser("~/.env")):
        try:
            for raw in open(path):
                line = raw.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line or line.startswith("#"):
                    continue
                name, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if ("INDIAN" in name.upper() and "KEY" in name.upper()) or val.startswith("sk-live-"):
                    return val
        except FileNotFoundError:
            continue
    return None


QUOTA_EXHAUSTED = object()         # sentinel: stop the whole run, don't write stubs


def get(path, key):
    req = urllib.request.Request(BASE + path, headers={"X-Api-Key": key, "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(30 * (attempt + 1))   # long backoff — free-tier quota
                continue
            return None
        except Exception:
            time.sleep(1 + attempt)
    return QUOTA_EXHAUSTED


def fetch_one(ticker, key):
    rec = {}
    for stat in STATS:
        d = get(f"/historical_stats?stock_name={ticker}&stats={stat}", key)
        if d is QUOTA_EXHAUSTED:
            return QUOTA_EXHAUSTED
        if isinstance(d, dict) and d:
            rec[stat] = d
        time.sleep(0.25)
    return rec if rec else None


def main():
    key = load_key()
    if not key:
        print("NO KEY in .env (INDIAN_API_KEY=...). Stop."); return
    os.makedirs(OUT, exist_ok=True)
    tickers = sys.argv[1:] if len(sys.argv) > 1 else discover_tickers()
    ok = fail = skip = 0
    for i, t in enumerate(tickers):
        path = os.path.join(OUT, f"{t}.json")
        if os.path.exists(path):
            skip += 1; continue
        rec = fetch_one(t, key)
        if rec is QUOTA_EXHAUSTED:
            print(f"QUOTA EXHAUSTED at {t} ({i}/{len(tickers)}, ok={ok}) — "
                  f"stopping; re-run later to resume.", flush=True)
            return
        if rec:
            json.dump(rec, open(path, "w")); ok += 1
        else:
            fail += 1                      # NO stub — resume will retry this name
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tickers)} ok={ok} fail={fail} skip={skip}", flush=True)
        time.sleep(0.2)
    print(f"DONE: ok={ok} fail={fail} skip={skip} of {len(tickers)}", flush=True)


if __name__ == "__main__":
    main()
