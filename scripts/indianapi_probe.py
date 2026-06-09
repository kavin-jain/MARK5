"""
Probe indianapi.in (stock.indianapi.in) to (1) verify the API key works, (2) discover
the auth header, and (3) show the structure + historical depth of the fundamentals we'd
use for F3 (the quality factor). Reads the key from .env at RUNTIME (never hard-coded).

  python3 scripts/indianapi_probe.py
"""
import os, sys, json, urllib.request, urllib.error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://stock.indianapi.in"


def load_key():
    # prefer real env var; else parse .env tolerantly (handle `export`, spaces, quotes,
    # and any var name containing INDIAN + KEY or = an sk-live- token).
    for v in ("INDIAN_API_KEY", "INDIANAPI_KEY", "INDIAN_STOCK_API_KEY", "API_KEY"):
        k = os.getenv(v)
        if k:
            return k.strip()
    for path in (os.path.join(_ROOT, ".env"), os.path.expanduser("~/.env")):
        try:
            for raw in open(path):
                line = raw.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line or line.startswith("#"):
                    continue
                name, _, val = line.partition("=")
                name = name.strip().upper()
                val = val.strip().strip('"').strip("'")
                if ("INDIAN" in name and "KEY" in name) or val.startswith("sk-live-"):
                    return val
        except FileNotFoundError:
            continue
    return None


def call(path, key, header_name):
    url = BASE + path
    req = urllib.request.Request(url, headers={header_name: key,
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")[:200]
    except Exception as e:
        return None, repr(e)[:200]


def main():
    key = load_key()
    if not key:
        print("NO KEY found. Add to .env:  INDIAN_API_KEY=sk-live-...")
        return
    print(f"key loaded (len={len(key)}, starts {key[:7]}...)\n")

    # discover the working auth header
    header = None
    for h in ("X-Api-Key", "x-api-key", "api-key", "Authorization"):
        st, _ = call("/NSE_most_active", key, h)
        print(f"  auth header '{h}': HTTP {st}")
        if st == 200:
            header = h
            break
    if not header:
        print("\nNo auth header worked — key may be invalid or endpoint differs. Stop.")
        return
    print(f"\n==> working auth header: {header}\n")

    # most-active (the 'most active stocks' Kavin mentioned)
    st, body = call("/NSE_most_active", key, header)
    try:
        d = json.loads(body)
        print(f"NSE_most_active: {len(d) if isinstance(d,list) else 'obj'} items; "
              f"sample: {json.dumps(d[:1] if isinstance(d,list) else d, default=str)[:200]}")
    except Exception:
        print("NSE_most_active parse fail:", body[:150])

    # the F3 lever: historical ratios for one stock
    for stat in ("ratios", "yoy_results", "balancesheet", "cashflow"):
        st, body = call(f"/historical_stats?stock_name=HAL&stats={stat}", key, header)
        if st == 200:
            try:
                d = json.loads(body)
                keys = list(d.keys())[:12] if isinstance(d, dict) else f"list[{len(d)}]"
                print(f"\nhistorical_stats stats={stat}: HTTP 200")
                print(f"   top-level keys/shape: {keys}")
                print(f"   sample: {json.dumps(d, default=str)[:400]}")
            except Exception:
                print(f"\nstats={stat}: 200 but parse fail: {body[:150]}")
        else:
            print(f"\nstats={stat}: HTTP {st} {body[:120]}")


if __name__ == "__main__":
    main()
