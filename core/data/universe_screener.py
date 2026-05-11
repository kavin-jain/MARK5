"""
MARK5 Dynamic Universe Screener v1.0 — PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uses the ISE Indian stock API to build a fresh, data-driven universe of
high-momentum NSE stocks every session. No hardcoded ticker lists.

SOURCES (token-efficient):
  1. NSE Most Active    → high-volume / high-interest stocks
  2. Trending (gainers) → stocks breaking out TODAY
  3. Price Shockers     → big % movers
  4. 52-week highs      → stocks near multi-year breakout

COST: 4 ISE tokens per run (all results cached daily).

CHANGELOG:
- [2026-05-10] v1.0: Dynamic universe from ISE API, dedup + sanitise.

TRADING ROLE: Universe construction for portfolio rotation
SAFETY LEVEL: HIGH
"""

import logging
import os
import sys
import re
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load .env so ISE_API_KEY is available to ISEAdapter
_ENV_FILE = os.path.join(_ROOT, ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


logger = logging.getLogger("MARK5.UniverseScreener")

# Symbols to always exclude (ETFs, REITs, indices, illiquid)
_BLACKLIST = {
    "NIFTY", "SENSEX", "NIFTYBEES", "GOLDBEES", "SETFNIF50",
    "LIQUIDBEES", "SILVERBEES", "BANKBEES", "JUNIORBEES",
}

# Valid NSE symbol pattern: 1-20 uppercase alphanumeric chars + hyphens
_NSE_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9&\-]{1,19}$")


# ISE API sometimes returns non-standard ticker codes (e.g. 'SBI.NS' for SBIN,
# 'HDBK.NS' for HDFCBANK). Map these back to the real NSE trading symbol.
_ISE_TO_NSE: dict = {
    "SBI":   "SBIN",
    "HDBK":  "HDFCBANK",
    "ADAN":  "ADANIENT",
    "PNBK":  "PNB",
    "TISC":  "TATASTEEL",
    "SAMD":  "SAMMAANLOAN",   # likely illiquid — sanitiser will catch
    "ETEA":  "EIHOTEL",
    "CNBK":  "CANBK",
    "BAJA":  "BAJAJFINSV",
    "TEML":  "TECHM",
    "HLL":   "HINDUNILVR",
    "COAL":  "COALINDIA",
    "BJFN":  "BAJFINANCE",
    "AXBK":  "AXISBANK",
    "ULTC":  "ULTRACEMCO",
    "ONGC":  "ONGC",
    "TREN":  "TRENT",
    "REDY":  "DRREDDY",
    "VDAN":  "VEDL",
    "JAINREC": "JAINREC",
}


def _sanitise_symbol(raw: str) -> Optional[str]:
    """
    Normalise and validate a raw ticker string from ISE API.
    Strips .NS/.BO suffixes, applies ISE→NSE symbol mapping, uppercases.
    """
    if not isinstance(raw, str):
        return None
    sym = raw.strip().upper().replace(".NS", "").replace(".BO", "")
    sym = sym.replace(" ", "").replace("\t", "")
    if not sym or sym in _BLACKLIST:
        return None
    # Apply ISE→NSE mapping
    sym = _ISE_TO_NSE.get(sym, sym)
    if not _NSE_SYMBOL_RE.match(sym):
        return None
    return sym


def build_dynamic_universe(
    min_size: int = 30,
    max_size: int = 80,
    include_fallback: bool = True,
) -> List[str]:
    """
    Build a fresh universe of NSE momentum stocks from the ISE API.

    Strategy:
      1. NSE Most Active  → volume leaders (broad market participation)
      2. Trending gainers → stocks with today's price momentum
      3. Price Shockers   → big daily % movers (event-driven breakouts)
      4. 52-week highs    → stocks at multi-year breakout levels (RULE 29)

    All 4 endpoints use daily caching → 4 tokens total per day.

    Returns:
        Deduplicated, sanitised list of NSE ticker symbols.
    """
    from core.data.data_pipeline import DataPipeline
    from core.data.adapters.ise_adapter import TokenBudgetExceeded

    pipeline = DataPipeline()
    seen: dict[str, int] = {}   # ticker → appearance count (for priority ranking)

    def _add_tickers(symbols: List[str], source: str, weight: int = 1) -> None:
        for sym in symbols:
            clean = _sanitise_symbol(sym)
            if clean:
                seen[clean] = seen.get(clean, 0) + weight
                logger.debug(f"  [{source}] {clean}")

    # ── 1. NSE Most Active ────────────────────────────────────────────────
    # API returns: ticker='SBI.NS', company='State Bank of India'
    try:
        df_active = pipeline.get_most_active("NSE")
        if not df_active.empty:
            col = next((c for c in ["ticker", "symbol", "name", "stock_name", "company"] if c in df_active.columns), None)
            if col:
                _add_tickers(df_active[col].tolist(), "most_active", weight=2)
                logger.info(f"Most active: {len(df_active)} entries — cols: {list(df_active.columns[:5])}")
    except (TokenBudgetExceeded, Exception) as e:
        logger.warning(f"Most active fetch failed: {e}")

    # ── 2. Trending (top_gainers + top_losers from trending endpoint) ───────────
    # API: {top_gainers:[{ric:'BAJA.NS', company_name:...}], top_losers:[...]}
    try:
        raw_trend = pipeline.ise.fetch_trending()
        gainers_raw = raw_trend.get("top_gainers", []) if isinstance(raw_trend, dict) else []
        # Use ric field (e.g. 'BAJA.NS') — sanitiser strips .NS and maps to NSE code
        if gainers_raw and isinstance(gainers_raw[0], dict):
            col = next((k for k in ["ric", "ticker", "symbol", "nseCode"] if k in gainers_raw[0]), None)
            if col:
                _add_tickers([r[col] for r in gainers_raw if col in r], "trending_gainer", weight=3)
                logger.info(f"Trending gainers: {len(gainers_raw)} entries")
    except (TokenBudgetExceeded, Exception) as e:
        logger.warning(f"Trending fetch failed: {e}")

    # ── 3. Price Shockers ─────────────────────────────────────────────────
    # API: {BSE_PriceShocker:[{nseCode, ric, ...}], NSE_PriceShocker:[...]}
    try:
        raw_shock = pipeline.ise.fetch_price_shockers()
        shock_list = []
        if isinstance(raw_shock, dict):
            shock_list = raw_shock.get("NSE_PriceShocker", raw_shock.get("BSE_PriceShocker", []))
        elif isinstance(raw_shock, list):
            shock_list = raw_shock
        if shock_list:
            col = next((k for k in ["nseCode", "ticker", "symbol", "ric"] if k in shock_list[0]), None)
            if col:
                _add_tickers([r[col] for r in shock_list if col in r], "price_shocker", weight=2)
                logger.info(f"Price shockers (NSE): {len(shock_list)} entries")
    except (TokenBudgetExceeded, Exception) as e:
        logger.warning(f"Price shockers fetch failed: {e}")

    # ── 4. 52-week highs ──────────────────────────────────────────────────
    try:
        import pandas as pd
        raw_52 = pipeline.ise.fetch_52_week_high_low()
        # API returns dict with keys like '52WeekHigh', '52WeekLow'
        items_52 = []
        if isinstance(raw_52, dict):
            items_52 = raw_52.get("52WeekHigh", raw_52.get("data", []))
        elif isinstance(raw_52, list):
            items_52 = raw_52
        if items_52 and isinstance(items_52[0], dict):
            col = next((k for k in ["ticker", "nseCode", "symbol", "name"] if k in items_52[0]), None)
            col_curr = next((k for k in ["price", "currentPrice", "ltp", "close"] if k in items_52[0]), None)
            col_high = next((k for k in ["52_week_high", "yearHigh", "high52", "week52High"] if k in items_52[0]), None)
            if col:
                if col_curr and col_high:
                    # Only stocks within 5% of 52w high (RULE 29 — breakout zone)
                    near = [r for r in items_52
                            if pd.to_numeric(r.get(col_curr, 0), errors='coerce') >=
                               pd.to_numeric(r.get(col_high, 0), errors='coerce') * 0.95]
                    _add_tickers([r[col] for r in near if col in r], "52w_high", weight=3)
                    logger.info(f"52w highs (within 5%): {len(near)} / {len(items_52)} entries")
                else:
                    _add_tickers([r[col] for r in items_52 if col in r], "52w_high", weight=1)
    except (TokenBudgetExceeded, Exception) as e:
        logger.warning(f"52w high/low fetch failed: {e}")

    # ── Rank by appearance count, then cap ───────────────────────────────
    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    universe = [sym for sym, _ in ranked][:max_size]

    logger.info(f"Dynamic universe: {len(universe)} stocks "
                f"(raw candidates: {len(seen)})")

    # ── Fallback if API delivered too little ─────────────────────────────
    if include_fallback and len(universe) < min_size:
        logger.warning(f"API returned only {len(universe)} stocks — augmenting with fallback")
        fallback = [
            # ── Large-cap anchors (NIFTY50 core) ──
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ITC", "LT", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "ASIANPAINT", "TITAN",
            "SUNPHARMA", "WIPRO", "HCLTECH", "TECHM", "TATASTEEL", "JSWSTEEL",
            "COALINDIA", "ONGC", "NTPC", "POWERGRID", "MARUTI", "M&M", "BAJAJ-AUTO",
            # ── Defence / PSU Capital Goods ──
            "BEL", "HAL", "BHEL", "CUMMINSIND", "ABB", "SIEMENS",
            # ── PSU Banks (sector rotation alpha) ──
            "PNB", "CANBK", "BANKINDIA", "UNIONBANK", "BANKBARODA",
            # ── Private Mid Banks ──
            "AUBANK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB",
            # ── Pharma ──
            "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "TORNTPHARM", "AUROPHARMA", "ZYDUSLIFE",
            # ── IT Mid-cap ──
            "TATAELXSI", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE",
            # ── Auto & Auto-ancillaries ──
            "TATAMOTORS", "EICHERMOT", "HEROMOTOCO", "MOTHERSON",
            # ── Consumer / FMCG ──
            "NESTLEIND", "BRITANNIA", "PIDILITIND", "TRENT", "VOLTAS",
            # ── Metals / Mining ──
            "VEDL", "ADANIPORTS", "ADANIENT",
            # ── New Economy ──
            "ZOMATO", "IRCTC", "DMART",
        ]
        existing = set(universe)
        for sym in fallback:
            if sym not in existing:
                universe.append(sym)
                existing.add(sym)
            if len(universe) >= min_size:
                break

    if not universe:
        raise RuntimeError("Universe is empty — all ISE endpoints failed and fallback disabled")

    logger.info(f"Final universe ({len(universe)} stocks): {universe[:10]}{'...' if len(universe)>10 else ''}")
    return universe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    uni = build_dynamic_universe()
    print(f"\n{'='*60}")
    print(f"Dynamic Universe: {len(uni)} NSE stocks")
    print(f"{'='*60}")
    for i, sym in enumerate(uni, 1):
        print(f"  {i:>3}. {sym}")
