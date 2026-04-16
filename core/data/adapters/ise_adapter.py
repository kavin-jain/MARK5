"""
MARK5 INDIAN STOCK API ADAPTER v1.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provider  : stock.indianapi.in
Auth      : x-api-key header  (env var: ISE_API_KEY)
OpenAPI   : api-1.json (openapi: 3.0.1)

TRADING ROLE : Supplementary fundamental + event data (NOT OHLCV)
SAFETY LEVEL : HIGH — drives RULE 25 event gate and confidence scoring

TOKEN BUDGET : 500 req/month hard limit. Every call is logged to ledger.
CALL STRATEGY (Single-Call Efficiency):
  /stock               → prices + technicals + analyst + news + corp actions (1 token)
  /corporate_actions   → deduplicated RULE 25 gate with 7-day cache (1 token, weekly)
  /historical_stats    → quarterly P&L, shareholding pattern (1 token, weekly)

DO NOT call for OHLCV. Kite is the OHLCV source of truth.

RULE 43 compliance : 5s timeout, exponential backoff 1s→2s→4s, max 3 attempts.
RULE 44 compliance : No bare except. Catches requests.exceptions only.

ENV VARS REQUIRED:
  ISE_API_KEY  — secret token from stock.indianapi.in dashboard
"""

import http.client
import json
import logging
import os
import ssl
import time
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("MARK5.ISEAdapter")

# ── API Config ──────────────────────────────────────────────────────────────
_API_HOST = "stock.indianapi.in"
_ENV_KEY  = "ISE_API_KEY"

# ── Token Budget ─────────────────────────────────────────────────────────────
_MONTHLY_TOKEN_LIMIT = 500
_RAW_DATA_DIR = Path(__file__).parents[3] / "data" / "raw"
_BUDGET_FILE  = _RAW_DATA_DIR / ".token_budget.json"

# ── RULE 43: Retry policy ────────────────────────────────────────────────────
_TIMEOUT_SECS     = 5
_BACKOFF_SECS     = [1, 2, 4]   # max 3 attempts
_RATE_LIMIT_PAUSE = 60          # pause 60s on 429

# ── Cache TTLs ───────────────────────────────────────────────────────────────
_DAILY_CACHE_SUFFIX  = lambda: date.today().isoformat()                      # yyyy-mm-dd
_WEEKLY_CACHE_SUFFIX = lambda: f"w{date.today().isocalendar()[1]}_{date.today().year}"  # w14_2026


class TokenBudgetExceeded(RuntimeError):
    """Raised when monthly 500-request budget would be breached."""


class ISEAdapter:
    """
    Thin, token-efficient wrapper around stock.indianapi.in.

    Request lifecycle:
        1. Budget gate (raise before any network call if limit reached)
        2. Cache check (data/raw/{ticker}_{date}.json) — return if fresh
        3. Fetch with RULE 43 retry/timeout policy
        4. Persist JSON to data/raw/ (offline model training reuse)
        5. Decrement budget ledger

    Args:
        api_key : ISE secret token. Reads ISE_API_KEY env var if omitted.
        raw_dir : Override JSON persistence directory.
    """

    def __init__(
        self,
        api_key:  Optional[str]  = None,
        raw_dir:  Optional[Path] = None,
    ):
        self._api_key = api_key or os.getenv(_ENV_KEY, "")
        if not self._api_key:
            logger.warning(
                f"{_ENV_KEY} not set in .env. ISE data disabled until key is added. "
                "All calls will return empty dicts."
            )

        self._raw_dir = Path(raw_dir) if raw_dir else _RAW_DATA_DIR
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PUBLIC API — endpoint methods
    # =========================================================================

    def fetch_stock(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        1. Get Company Data by Name: /stock?name={ticker}
        Returns: price + technicals + analyst + shareholding + corp actions + news.
        Cost: 1 token.  Cache: daily.
        """
        cache_file = self._raw_dir / f"{ticker}_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/stock",
            params={"name": ticker},
            cache_file=cache_file,
            label=f"stock/{ticker}",
            force_refresh=force_refresh,
        )

    def fetch_industry_search(self, query: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        2. Industry Search: /industry_search?query={query}
        Cost: 1 token. Cache: weekly.
        """
        cache_file = self._raw_dir / f"industry_search_{query}_{_WEEKLY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/industry_search",
            params={"query": query},
            cache_file=cache_file,
            label=f"industry_search/{query}",
            force_refresh=force_refresh,
        )

    def fetch_mutual_fund_search(self, query: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        3. Mutual Fund Search: /mutual_fund_search?query={query}
        Cost: 1 token. Cache: weekly.
        """
        cache_file = self._raw_dir / f"mf_search_{query}_{_WEEKLY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/mutual_fund_search",
            params={"query": query},
            cache_file=cache_file,
            label=f"mutual_fund_search/{query}",
            force_refresh=force_refresh,
        )

    def fetch_trending(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        4. Trending: /trending
        Cost: 1 token.  Cache: daily.
        """
        cache_file = self._raw_dir / f"trending_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/trending",
            params={},
            cache_file=cache_file,
            label="trending",
            force_refresh=force_refresh,
        )

    def fetch_52_week_high_low(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        5. Fetch 52 Week High Low Data: /fetch_52_week_high_low_data
        Cost: 1 token. Cache: daily.
        """
        cache_file = self._raw_dir / f"52_week_high_low_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/fetch_52_week_high_low_data",
            params={},
            cache_file=cache_file,
            label="52_week_high_low",
            force_refresh=force_refresh,
        )

    def fetch_NSE_most_active(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        6. NSE Most Active: /NSE_most_active
        Cost: 1 token. Cache: daily.
        """
        cache_file = self._raw_dir / f"NSE_most_active_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/NSE_most_active",
            params={},
            cache_file=cache_file,
            label="NSE_most_active",
            force_refresh=force_refresh,
        )

    def fetch_BSE_most_active(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        7. BSE Most Active: /BSE_most_active
        Cost: 1 token. Cache: daily.
        """
        cache_file = self._raw_dir / f"BSE_most_active_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/BSE_most_active",
            params={},
            cache_file=cache_file,
            label="BSE_most_active",
            force_refresh=force_refresh,
        )

    def fetch_mutual_funds(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        8. Mutual Funds: /mutual_funds
        Cost: 1 token. Cache: daily.
        """
        cache_file = self._raw_dir / f"mutual_funds_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/mutual_funds",
            params={},
            cache_file=cache_file,
            label="mutual_funds",
            force_refresh=force_refresh,
        )

    def fetch_price_shockers(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        9. Price Shockers: /price_shockers
        Cost: 1 token. Cache: daily.
        """
        cache_file = self._raw_dir / f"price_shockers_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/price_shockers",
            params={},
            cache_file=cache_file,
            label="price_shockers",
            force_refresh=force_refresh,
        )

    def fetch_commodities(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        10. Commodity Futures: /commodities
        Cost: 1 token. Cache: daily.
        """
        cache_file = self._raw_dir / f"commodities_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/commodities",
            params={},
            cache_file=cache_file,
            label="commodities",
            force_refresh=force_refresh,
        )

    def fetch_stock_target_price(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        11. Analyst Recommendations: /stock_target_price?stock_id={ticker}
        Cost: 1 token.  Cache: weekly.
        """
        cache_file = self._raw_dir / f"{ticker}_target_{_WEEKLY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/stock_target_price",
            params={"stock_id": ticker},
            cache_file=cache_file,
            label=f"stock_target_price/{ticker}",
            force_refresh=force_refresh,
        )

    def fetch_stock_forecasts(
        self, 
        ticker: str, 
        measure_code: str = "EPS", 
        period_type: str = "Annual", 
        data_type: str = "Actuals", 
        age: str = "Current",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        12. Stock Forecasts: /stock_forecasts?stock_id={ticker}&measure_code={measure_code}&period_type={period_type}&data_type={data_type}&age={age}
        Cost: 1 token. Cache: weekly.
        """
        cache_file = self._raw_dir / f"{ticker}_forecast_{measure_code}_{_WEEKLY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/stock_forecasts",
            params={
                "stock_id": ticker,
                "measure_code": measure_code,
                "period_type": period_type,
                "data_type": data_type,
                "age": age
            },
            cache_file=cache_file,
            label=f"stock_forecasts/{ticker}/{measure_code}",
            force_refresh=force_refresh,
        )

    def fetch_historical_data(
        self, 
        ticker: str, 
        period: str = "1yr", 
        filter: str = "default",
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        13. Historical Data: /historical_data?stock_name={ticker}&period={period}&filter={filter}
        Cost: 1 token. Cache: daily.
        """
        cache_file = self._raw_dir / f"{ticker}_hist_{period}_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/historical_data",
            params={
                "stock_name": ticker,
                "period": period,
                "filter": filter
            },
            cache_file=cache_file,
            label=f"historical_data/{ticker}/{period}",
            force_refresh=force_refresh,
        )

    def fetch_historical_stats(
        self,
        ticker: str,
        stats:  str = "quarter_results",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        14. Historical Stats: /historical_stats?stock_name={ticker}&stats={stats}
        Cost: 1 token.  Cache: weekly.
        """
        cache_file = self._raw_dir / f"{ticker}_{stats}_{_WEEKLY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/historical_stats",
            params={"stock_name": ticker, "stats": stats},
            cache_file=cache_file,
            label=f"historical_stats/{ticker}/{stats}",
            force_refresh=force_refresh,
        )

    def fetch_corporate_actions(
        self, ticker: str, force_refresh: bool = False
    ) -> Union[Dict[str, Any], List]:
        """
        /corporate_actions?stock_name={ticker}
        Cost: 1 token.  Cache: weekly.
        """
        cache_file = self._raw_dir / f"{ticker}_corp_{_WEEKLY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/corporate_actions",
            params={"stock_name": ticker},
            cache_file=cache_file,
            label=f"corporate_actions/{ticker}",
            force_refresh=force_refresh,
        )

    def fetch_recent_announcements(self, ticker: str, force_refresh: bool = False) -> Union[Dict, List]:
        """
        /recent_announcements?stock_name={ticker}
        Cost: 1 token.  Cache: daily.
        """
        cache_file = self._raw_dir / f"{ticker}_announcements_{_DAILY_CACHE_SUFFIX()}.json"
        return self._get_cached_or_fetch(
            endpoint="/recent_announcements",
            params={"stock_name": ticker},
            cache_file=cache_file,
            label=f"recent_announcements/{ticker}",
            force_refresh=force_refresh,
        )


    def get_budget_status(self) -> Dict[str, int]:
        """Return current token budget: {used, remaining, limit}."""
        ledger = self._load_budget_ledger()
        used   = ledger.get(self._month_key(), 0)
        return {
            "used":      used,
            "remaining": max(0, _MONTHLY_TOKEN_LIMIT - used),
            "limit":     _MONTHLY_TOKEN_LIMIT,
        }

    # =========================================================================
    # INTERNAL: unified cache-or-fetch
    # =========================================================================

    def _get_cached_or_fetch(
        self,
        endpoint:     str,
        params:       Dict[str, str],
        cache_file:   Path,
        label:        str,
        force_refresh: bool = False,
    ) -> Any:
        """
        Return from cache if file exists and force_refresh is False.
        Otherwise fetch from API, save to cache, decrement budget.
        """
        if not force_refresh and cache_file.exists():
            logger.info(f"📦 ISE cache hit: {label} ({cache_file.name})")
            return self._load_json(cache_file)

        return self._request_with_budget(
            endpoint=endpoint,
            params=params,
            cache_file=cache_file,
            label=label,
        )

    # =========================================================================
    # INTERNAL: budget gate + HTTP + retry
    # =========================================================================

    def _request_with_budget(
        self,
        endpoint:   str,
        params:     Dict[str, str],
        cache_file: Path,
        label:      str,
    ) -> Any:
        """
        Core fetch with:
          - Budget gate (raise before any network hit)
          - RULE 43 retry policy via http.client (no requests dependency)
          - Cache persistence on success
          - Budget decrement on success
        """
        if not self._api_key:
            logger.warning(f"ISE fetch skipped (no API key): {label}")
            return {}

        # ── Budget gate ──────────────────────────────────────────────────────
        ledger     = self._load_budget_ledger()
        month_key  = self._month_key()
        used       = ledger.get(month_key, 0)

        if used >= _MONTHLY_TOKEN_LIMIT:
            raise TokenBudgetExceeded(
                f"ISE monthly limit ({_MONTHLY_TOKEN_LIMIT}) reached. "
                f"Resets 1st of next month. Blocked: {label}."
            )

        remaining = _MONTHLY_TOKEN_LIMIT - used
        logger.info(
            f"🔑 ISE fetch: {label} | budget {used}/{_MONTHLY_TOKEN_LIMIT} "
            f"({remaining} remaining)"
        )

        # ── Build query string ───────────────────────────────────────────────
        qs  = ("?" + urllib.parse.urlencode(params)) if params else ""
        path = endpoint + qs
        headers = {"x-api-key": self._api_key}

        # ── RULE 43 retry loop ───────────────────────────────────────────────
        last_exc: Optional[Exception] = None

        for attempt, backoff in enumerate(_BACKOFF_SECS, start=1):
            try:
                conn = http.client.HTTPSConnection(
                    _API_HOST,
                    timeout=_TIMEOUT_SECS,
                    context=ssl.create_default_context(),
                )
                conn.request("GET", path, headers=headers)
                res  = conn.getresponse()
                body = res.read().decode("utf-8")
                conn.close()

                if res.status == 429:
                    logger.warning(
                        f"ISE 429 rate-limit on {label} (attempt {attempt}). "
                        f"Pausing {_RATE_LIMIT_PAUSE}s…"
                    )
                    time.sleep(_RATE_LIMIT_PAUSE)
                    continue

                if res.status == 404:
                    logger.warning(f"ISE 404: {label} — ticker not found on API.")
                    return {}

                if res.status == 422:
                    logger.error(f"ISE 422 validation error for {label}: {body[:200]}")
                    return {}

                if res.status == 200:
                    data = json.loads(body)
                    self._save_json(cache_file, data)
                    ledger[month_key] = used + 1
                    self._save_budget_ledger(ledger)
                    logger.info(
                        f"✅ ISE OK: {label} | "
                        f"tokens used this month: {ledger[month_key]}"
                    )
                    return data

                logger.error(
                    f"ISE HTTP {res.status} on {label} (attempt {attempt}): "
                    f"{body[:120]}"
                )
                last_exc = IOError(f"HTTP {res.status}")
                time.sleep(backoff)

            except (http.client.HTTPException, OSError, TimeoutError) as exc:
                logger.warning(
                    f"ISE network error on {label} (attempt {attempt}/{len(_BACKOFF_SECS)}): {exc}"
                )
                last_exc = exc
                time.sleep(backoff)

            except json.JSONDecodeError as exc:
                logger.error(f"ISE invalid JSON from {label}: {exc}")
                return {}

        logger.error(
            f"ISE fetch FAILED after {len(_BACKOFF_SECS)} attempts: {label}. "
            f"Last error: {last_exc}. Returning empty dict."
        )
        return {}

    # =========================================================================
    # INTERNAL: cache + budget ledger helpers
    # =========================================================================

    @staticmethod
    def _month_key() -> str:
        return date.today().strftime("%Y-%m")

    def _load_budget_ledger(self) -> Dict[str, int]:
        if _BUDGET_FILE.exists():
            try:
                with _BUDGET_FILE.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_budget_ledger(self, ledger: Dict[str, int]) -> None:
        try:
            with _BUDGET_FILE.open("w") as f:
                json.dump(ledger, f, indent=2)
        except OSError as exc:
            logger.error(f"Budget ledger save failed: {exc}")

    @staticmethod
    def _load_json(path: Path) -> Any:
        try:
            with path.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"JSON load failed from {path}: {exc}")
            return {}

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        try:
            with path.open("w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"💾 ISE raw saved: {path.name}")
        except OSError as exc:
            logger.error(f"JSON save failed to {path}: {exc}")
