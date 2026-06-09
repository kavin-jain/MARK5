"""
MARK5 News Sentiment Provider v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fetches RSS news from free Indian financial news sources and computes
daily ticker-specific sentiment scores using financial keyword matching.

SOURCES (all free, no API key):
  1. MoneyControl Markets:   https://www.moneycontrol.com/rss/MCtopnews.xml
  2. Economic Times Markets: https://economictimes.indiatimes.com/markets/rss.cms
  3. Business Standard:      https://www.business-standard.com/rss/markets-106.rss
  4. LiveMint Markets:       https://www.livemint.com/rss/markets
  5. NSE Corporate Actions:  https://www.nseindia.com/api/corporate-announcements (scraped)

SENTIMENT METHOD:
  Pure keyword matching — no external ML models required.
  Score = (positive_count - negative_count) / (total_count + ε)
  Final range: [-1, +1] with 0 = neutral/no data

CACHE:
  data/sentiment/headlines_YYYY-MM-DD.json   — raw headlines per day
  data/sentiment/scores/TICKER.parquet       — per-ticker daily score series

TRADING ROLE: Inference-time enrichment ONLY — not used in offline training
              (RSS only goes back ~30 days; training uses 2015+ historical data)
SAFETY LEVEL: LOW RISK — read-only, no execution path dependency
"""

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

logger = logging.getLogger("MARK5.NewsSentiment")

# ── Directory layout ─────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_PROJECT    = _HERE.parent.parent
CACHE_BASE  = _PROJECT / "data" / "sentiment"
HEADLINE_DIR = CACHE_BASE / "headlines"
SCORE_DIR   = CACHE_BASE / "scores"

# ── Financial keyword dictionary ─────────────────────────────────────────────
# Curated for Indian equity markets (NSE/BSE context)
POSITIVE_WORDS = frozenset([
    "surge", "surges", "surged", "rally", "rallies", "rallied",
    "gain", "gains", "gained", "rise", "rises", "rose", "risen",
    "profit", "profits", "profitable", "record", "records",
    "beat", "beats", "outperform", "outperforms", "outperformed",
    "upgrade", "upgrades", "upgraded", "buy", "strong", "strength",
    "growth", "grew", "bullish", "bull", "positive", "upside",
    "breakout", "breakthrough", "milestone", "expansion", "boom",
    "order", "orders", "contract", "win", "wins", "winning",
    "revenue", "topline", "margin", "margins", "ebitda", "pat",
    "dividend", "bonus", "buyback", "acquisition", "deal",
    "approval", "approved", "launch", "launches", "launched",
    "recovery", "recover", "rebound", "bounce", "jump", "jumps",
    "all-time-high", "52-week-high", "ath", "target", "optimistic",
    "robust", "healthy", "confident", "momentum", "stellar",
    "outpace", "accelerate", "accelerates", "ramp", "scale",
])

NEGATIVE_WORDS = frozenset([
    "fall", "falls", "fell", "fallen", "decline", "declines", "declined",
    "drop", "drops", "dropped", "plunge", "plunges", "plunged",
    "crash", "crashes", "crashed", "slump", "slumps", "slumped",
    "loss", "losses", "losing", "miss", "misses", "missed",
    "downgrade", "downgrades", "downgraded", "sell", "weak", "weakness",
    "bearish", "bear", "negative", "downside", "concern", "concerns",
    "risk", "risks", "caution", "warning", "warns", "warned",
    "default", "defaults", "defaulted", "npa", "fraud", "probe",
    "investigation", "regulatory", "penalty", "fine", "fined",
    "miss", "disappoint", "disappoints", "disappointed",
    "cut", "cuts", "cutting", "reduce", "reduces", "reduced",
    "profit-warning", "guidance-cut", "slowdown", "contraction",
    "underperform", "underperforms", "lag", "lags", "lagging",
    "exit", "resign", "layoff", "layoffs", "retrenchment",
    "debt", "leverage", "overlevered", "covenant", "breach",
    "recall", "recall", "ban", "banned", "seized", "seizure",
])

# Ticker aliases for headline matching (common alternative names)
TICKER_ALIASES: Dict[str, List[str]] = {
    "HDFCBANK":    ["HDFC Bank", "HDFC bank", "hdfc bank"],
    "ICICIBANK":   ["ICICI Bank", "ICICI bank"],
    "KOTAKBANK":   ["Kotak Bank", "Kotak Mahindra", "Kotak bank"],
    "SBIN":        ["SBI", "State Bank", "state bank"],
    "AUBANK":      ["AU Small Finance", "AU Bank"],
    "BANDHANBNK":  ["Bandhan Bank", "Bandhan bank"],
    "IDFCFIRSTB":  ["IDFC First Bank", "IDFC First"],
    "PNB":         ["Punjab National Bank", "PNB"],
    "YESBANK":     ["Yes Bank", "YES Bank"],
    "BAJFINANCE":  ["Bajaj Finance", "Bajaj finance"],
    "BHARTIARTL":  ["Bharti Airtel", "Airtel", "airtel"],
    "IDEA":        ["Vi", "Vodafone Idea", "Vodafone"],
    "RELIANCE":    ["Reliance Industries", "RIL", "Reliance"],
    "TATAELXSI":   ["Tata Elxsi", "TATA Elxsi"],
    "TATASTEEL":   ["Tata Steel", "tata steel", "TATA Steel"],
    "TCS":         ["Tata Consultancy", "TCS"],
    "INFY":        ["Infosys", "infosys"],
    "COFORGE":     ["Coforge", "coforge"],
    "PERSISTENT":  ["Persistent Systems", "Persistent"],
    "HAL":         ["HAL", "Hindustan Aeronautics"],
    "BEL":         ["BEL", "Bharat Electronics"],
    "LT":          ["L&T", "Larsen", "Larsen & Toubro"],
    "VOLTAS":      ["Voltas", "voltas"],
    "MARUTI":      ["Maruti Suzuki", "Maruti", "Suzuki"],
    "MOTHERSON":   ["Motherson Sumi", "Motherson"],
    "LUPIN":       ["Lupin", "lupin"],
    "SUNPHARMA":   ["Sun Pharma", "Sun Pharmaceutical"],
    "ASIANPAINT":  ["Asian Paints", "Asian paints"],
    "HINDUNILVR":  ["HUL", "Hindustan Unilever"],
    "TITAN":       ["Titan", "titan"],
    "TRENT":       ["Trent", "trent", "Westside"],
    "ITC":         ["ITC", "Indian Tobacco"],
}

# RSS feed URLs (all free, no auth)
RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.livemint.com/rss/markets",
]


class NewsSentimentProvider:
    """
    Provides daily ticker-specific news sentiment scores from free RSS feeds.

    Usage:
        provider = NewsSentimentProvider()
        series = provider.get_sentiment("HAL", start="2024-01-01", end="2025-01-01")
        # pd.Series with DatetimeIndex, values in [-1.0, +1.0]
        # Returns pd.Series of zeros if data unavailable (zero is neutral)

    Architecture:
        1. Fetch all RSS feed headlines (cached per day)
        2. Match ticker names/aliases in headlines
        3. Score each headline: +1 positive, -1 negative, 0 neutral
        4. Aggregate to daily score per ticker
        5. Return rolling mean for smoothness
    """

    def __init__(self):
        HEADLINE_DIR.mkdir(parents=True, exist_ok=True)
        SCORE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_sentiment(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        lookback_days: int = 30,
        rolling_window: int = 3,
    ) -> pd.Series:
        """
        Get daily sentiment score series for a ticker.

        Args:
            ticker:         NSE ticker symbol (e.g. "HAL")
            start:          Start date "YYYY-MM-DD" (defaults to today - lookback_days)
            end:            End date "YYYY-MM-DD" (defaults to today)
            lookback_days:  How many days back to look if start not specified
            rolling_window: Rolling window for smoothing (default 3 days)

        Returns:
            pd.Series with DatetimeIndex, values in [-1.0, +1.0].
            Returns zeros if feedparser unavailable or no data found.
        """
        if not FEEDPARSER_AVAILABLE:
            logger.debug("feedparser not available — returning zero sentiment")
            return pd.Series(dtype=float, name='sentiment')

        end_dt   = pd.Timestamp(end) if end else pd.Timestamp(date.today())
        start_dt = pd.Timestamp(start) if start else (end_dt - pd.Timedelta(days=lookback_days))

        # Fetch recent headlines (RSS is limited to ~30 days)
        fetch_start = max(start_dt, pd.Timestamp(date.today()) - pd.Timedelta(days=25))
        self._ensure_headlines(fetch_start.date(), end_dt.date())

        # Load and score headlines
        daily_scores = self._score_ticker_from_cache(ticker, fetch_start.date(), end_dt.date())

        if not daily_scores:
            return pd.Series(0.0, name='news_sentiment')

        series = pd.Series(daily_scores, dtype=float).sort_index()
        series.index = pd.to_datetime(series.index)
        series.name = 'news_sentiment'

        # Smooth with rolling mean
        if rolling_window > 1 and len(series) >= rolling_window:
            series = series.rolling(rolling_window, min_periods=1).mean()

        return series.loc[start_dt:end_dt]

    def get_sentiment_for_date(self, ticker: str, as_of: pd.Timestamp) -> float:
        """Get the most recent available sentiment score as of a date."""
        series = self.get_sentiment(
            ticker,
            start=(as_of - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
            end=as_of.strftime('%Y-%m-%d'),
        )
        if series.empty:
            return 0.0
        return float(series.iloc[-1])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_headlines(self, start: date, end: date) -> None:
        """Fetch headlines for date range if not cached."""
        current = max(start, date.today() - timedelta(days=25))  # RSS limit
        while current <= end:
            if current.weekday() < 5:  # weekdays only
                self._fetch_day(current)
            current += timedelta(days=1)

    def _fetch_day(self, day: date) -> Dict[str, List[str]]:
        """Fetch and cache all headlines for a single day."""
        cache_file = HEADLINE_DIR / f"headlines_{day.isoformat()}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

        headlines = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '') or entry.get('description', '')
                    pub_date = entry.get('published_parsed')
                    if pub_date:
                        entry_date = date(*pub_date[:3])
                        if entry_date == day:
                            headlines.append(f"{title}. {summary}")
                    else:
                        # No date — include if within today ±1 day
                        headlines.append(f"{title}. {summary}")
                time.sleep(0.3)  # polite delay
            except Exception as e:
                logger.debug(f"RSS fetch failed for {url}: {e}")

        result = {day.isoformat(): headlines}
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass
        return result

    def _score_headline(self, text: str) -> float:
        """Score a single headline. Returns value in [-1, +1]."""
        words = text.lower().split()
        pos = sum(1 for w in words if w.strip('.,!?') in POSITIVE_WORDS)
        neg = sum(1 for w in words if w.strip('.,!?') in NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _matches_ticker(self, text: str, ticker: str) -> bool:
        """Check if headline mentions the ticker or its aliases."""
        text_lower = text.lower()
        # Check direct ticker mention
        if ticker.lower() in text_lower:
            return True
        # Check aliases
        aliases = TICKER_ALIASES.get(ticker, [])
        for alias in aliases:
            if alias.lower() in text_lower:
                return True
        return False

    def _score_ticker_from_cache(
        self, ticker: str, start: date, end: date
    ) -> Dict[str, float]:
        """Score all cached headlines for a ticker in date range."""
        daily_scores: Dict[str, float] = {}
        current = start
        while current <= end:
            cache_file = HEADLINE_DIR / f"headlines_{current.isoformat()}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    # Headlines may be stored under any key for the day
                    all_headlines = []
                    for v in data.values():
                        if isinstance(v, list):
                            all_headlines.extend(v)

                    ticker_headlines = [h for h in all_headlines if self._matches_ticker(h, ticker)]
                    if ticker_headlines:
                        scores = [self._score_headline(h) for h in ticker_headlines]
                        daily_scores[current.isoformat()] = float(np.mean(scores))
                    else:
                        daily_scores[current.isoformat()] = 0.0
                except Exception:
                    daily_scores[current.isoformat()] = 0.0
            current += timedelta(days=1)
        return daily_scores


def get_sentiment_series(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """
    Module-level convenience function.
    Returns pd.Series of daily sentiment scores in [-1, +1].
    Always returns a valid Series (zeros if no data).
    """
    provider = NewsSentimentProvider()
    series = provider.get_sentiment(ticker, start=start, end=end)
    if series.empty:
        idx = pd.date_range(
            start or (date.today() - timedelta(days=30)).isoformat(),
            end or date.today().isoformat(),
            freq='B',
        )
        return pd.Series(0.0, index=idx, name='news_sentiment')
    return series
