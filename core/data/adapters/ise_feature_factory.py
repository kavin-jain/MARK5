"""
MARK5 ISE FEATURE FACTORY v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG v2.0:
  - NEW: analyst_upgrade_momentum()     → 3-month net analyst upgrade/downgrade count
  - NEW: analyst_target_revision()      → 90-day change in mean target price
  - NEW: moving_average_alignment()     → price vs 5/10/20/50/100/300d MAs (from ISE)
  - NEW: fundamental_quality_score()    → EPS+revenue growth composite from keyMetrics
  - UPGRADED: compute_confidence_modifier()
      Range expanded from ±0.08 → ±0.20
      Now uses all 7 signals, not 4.

SIGNAL TAXONOMY:
  1. is_safe_to_trade      → RULE 25 corporate actions veto (bool)
  2. analyst_gap           → (mean_target - current) / current  [−0.5, +0.5]
  3. analyst_target_revision → 90d change in analyst mean target [float]
  4. analyst_conviction    → consensus rating normalised to [0, 1]
  5. analyst_upgrade_momentum → net analyst upgrades in 3mo [int]
  6. moving_average_alignment → price vs MAs score [−1, +1]
  7. fundamental_quality_score → growth + margin composite [0, 1]
  8. news_sentiment        → TextBlob news polarity [−1, +1]
  9. institutional_delta   → QoQ FII + Promoter holding change [float]

DESIGN PRINCIPLES:
  - Robustness: Handles None, list instead of dict, schema variations.
  - Fail-Safe: Returns neutral values (0.0 / True) instead of raising.
  - RULE 25: 3-day blackout period for hard veto.
  - Decimal precision: All money comparisons use Decimal (RULE 5/6).
"""

import logging
import re
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("MARK5.ISEFeatureFactory")

# ── Tuning constants ─────────────────────────────────────────────────────────
EVENT_BLACKOUT_DAYS = 3
PAISA = Decimal("0.01")

try:
    from textblob import TextBlob
    _TEXTBLOB_OK = True
except ImportError:
    logger.warning("textblob not installed. Sentiment will be 0.0.")
    _TEXTBLOB_OK = False


# =============================================================================
# 1. RULE 25 CORPORATE ACTION GATE
# =============================================================================

def is_safe_to_trade(
    stock_data: Dict[str, Any],
    reference_date: Optional[date] = None,
    blackout_days: int = EVENT_BLACKOUT_DAYS,
) -> Tuple[bool, str]:
    """
    Checks for corporate actions (dividends, splits, results) within window.
    Vetoes trade if an event is in the blackout period (Rule 25).
    """
    if not isinstance(stock_data, dict):
        return True, ""

    ref = reference_date or date.today()
    cutoff = ref + timedelta(days=blackout_days)

    # ── 1a. Parse corporate action dates ─────────────────────────────────────
    corp_data = stock_data.get("stockCorporateActionData") or {}
    if isinstance(corp_data, dict):
        for action_type, entries in corp_data.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                raw_d = entry.get("date") or entry.get("exDate", "")
                event_date = _parse_date(str(raw_d))
                if event_date and ref <= event_date <= cutoff:
                    return False, f"Rule 25: {action_type} on {event_date}"

    # ── 1b. Parse board meeting / results dates ───────────────────────────────
    profile = stock_data.get("companyProfile") or {}
    if isinstance(profile, dict):
        for key in ["nextResultDate", "boardMeetingDate", "resultsDate"]:
            raw_val = profile.get(key)
            if raw_val:
                adate = _parse_date(str(raw_val))
                if adate and ref <= adate <= cutoff:
                    return False, f"Rule 25: Board Meeting/Results on {adate}"

    # ── 1c. F&O Expiry ───────────────────────────────────────────────────────
    expiry_dates = stock_data.get("futureExpiryDates") or []
    if isinstance(expiry_dates, list):
        for raw_exp in expiry_dates:
            exp_date = _parse_date(str(raw_exp))
            if exp_date and ref <= exp_date <= (ref + timedelta(days=1)):
                return False, f"F&O Expiry {exp_date} imminent"

    return True, ""


# =============================================================================
# 2. ANALYST GAP (Price Consensus)
# =============================================================================

def analyst_gap(
    stock_data: Dict[str, Any],
    target_data: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    (Mean Target - Current Price) / Current Price.
    Positive = analysts see upside. Source: /stock_target_price Mean.
    """
    try:
        if not isinstance(stock_data, dict):
            return None
        price_dict = stock_data.get("currentPrice") or {}
        if not isinstance(price_dict, dict):
            return None
        curr_raw = price_dict.get("NSE") or price_dict.get("BSE")
        if not curr_raw:
            return None
        current = Decimal(str(curr_raw))
        if current <= 0:
            return None

        target_mean = None
        if isinstance(target_data, dict):
            pt = target_data.get("priceTarget")
            if isinstance(pt, dict):
                target_mean = pt.get("Mean")

        if target_mean is None:
            return None

        target = Decimal(str(target_mean))
        gap = float((target - current) / current)
        return max(-0.5, min(0.5, gap))

    except (InvalidOperation, TypeError, ValueError, ZeroDivisionError) as exc:
        logger.debug(f"analyst_gap failed: {exc}")
        return None


# =============================================================================
# 3. ANALYST TARGET PRICE REVISION (90-day trend)  ← NEW v2.0
# =============================================================================

def analyst_target_revision(
    target_data: Optional[Dict[str, Any]]
) -> Optional[float]:
    """
    90-day change in analyst MEAN target price as a fraction.
    e.g., 0.019 = analysts raised mean target +1.9% in 90 days.

    Rising targets = analysts becoming MORE bullish even if stock is falling.
    This is a strong forward-looking alpha signal.

    Schema:
      priceTargetSnapshots.PriceTargetSnapshot: list of {Age, Mean, ...}
      Age values: OneWeekAgo, ThirtyDaysAgo, SixtyDaysAgo, NinetyDaysAgo
    """
    if not isinstance(target_data, dict):
        return None
    try:
        current_mean = None
        pt = target_data.get("priceTarget")
        if isinstance(pt, dict):
            current_mean = pt.get("Mean")
        if current_mean is None:
            return None
        current_mean = float(current_mean)

        snapshots_wrap = target_data.get("priceTargetSnapshots") or {}
        snapshot_list  = snapshots_wrap.get("PriceTargetSnapshot", [])
        if not isinstance(snapshot_list, list):
            return None

        ninety_day_mean = None
        for snap in snapshot_list:
            if isinstance(snap, dict) and snap.get("Age") == "NinetyDaysAgo":
                ninety_day_mean = snap.get("Mean")
                break

        if ninety_day_mean is None or float(ninety_day_mean) == 0:
            return None

        revision = (current_mean - float(ninety_day_mean)) / float(ninety_day_mean)
        return float(max(-0.30, min(0.30, revision)))

    except (ValueError, TypeError, ZeroDivisionError) as exc:
        logger.debug(f"analyst_target_revision failed: {exc}")
        return None


# =============================================================================
# 4. ANALYST CONVICTION (Rating Level)
# =============================================================================

def analyst_conviction(
    stock_data: Dict[str, Any],
    target_data: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Converts 1.0 (Strong Buy) → 5.0 (Sell) mean rating to [0, 1] score.
    Returns 0.5 (neutral) on failure.
    """
    try:
        mean_rating = None

        if isinstance(target_data, dict):
            reco = target_data.get("recommendation")
            if isinstance(reco, dict):
                mean_rating = reco.get("Mean")

        if mean_rating is None and isinstance(stock_data, dict):
            recos_bar = stock_data.get("recosBar")
            if isinstance(recos_bar, dict):
                mean_rating = recos_bar.get("meanValue")

        if mean_rating is None:
            return 0.5

        mean_f = float(mean_rating)
        score = 1.0 - (mean_f - 1.0) / 4.0
        return max(0.0, min(1.0, score))

    except (TypeError, ValueError, KeyError):
        return 0.5


# =============================================================================
# 5. ANALYST UPGRADE MOMENTUM (3-month net changes)  ← NEW v2.0
# =============================================================================

def analyst_upgrade_momentum(stock_data: Dict[str, Any]) -> float:
    """
    Net analyst upgrades in the last 3 months.
    Source: analystView list with per-rating historical counts.

    Schema per item:
      {'ratingValue': 1, 'ratingName': 'Strong Buy',
       'numberOfAnalystsLatest': '13',
       'numberOfAnalysts3MonthAgo': '12', ...}

    ratingValue: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell

    Logic:
      +1 for each analyst moving TO bullish ratings (1 or 2)
      -1 for each analyst moving TO bearish ratings (4 or 5)
      Hold changes are neutral.

    A positive score means analysts are upgrading the stock.
    Analyst upgrade momentum has documented predictive power (1-3 months ahead).
    """
    analyst_view = stock_data.get("analystView", []) if isinstance(stock_data, dict) else []
    if not isinstance(analyst_view, list) or not analyst_view:
        return 0.0

    net_bullish_change = 0.0
    for item in analyst_view:
        if not isinstance(item, dict):
            continue
        rv = item.get("ratingValue", 0)
        try:
            latest    = float(item.get("numberOfAnalystsLatest",    0) or 0)
            three_ago = float(item.get("numberOfAnalysts3MonthAgo", 0) or 0)
            change    = latest - three_ago
        except (ValueError, TypeError):
            continue

        if rv in (1, 2):    # Strong Buy or Buy — bullish upgrades
            net_bullish_change += change
        elif rv in (4, 5):  # Sell or Strong Sell — bearish downgrades
            net_bullish_change -= change
        # Hold (rv=3) and Total (rv=6) are ignored

    return float(max(-10.0, min(10.0, net_bullish_change)))


# =============================================================================
# 6. MOVING AVERAGE ALIGNMENT  ← NEW v2.0
# =============================================================================

def moving_average_alignment(
    stock_data: Dict[str, Any],
    current_price: Optional[float] = None,
) -> float:
    """
    Score based on how many moving averages the stock is trading above.
    Source: stockTechnicalData list — 5, 10, 20, 50, 100, 300 day MAs.

    Returns score in [-1, +1]:
      +1.0  = stock above ALL moving averages (strong uptrend)
       0.0  = stock above half (mixed)
      -1.0  = stock below ALL moving averages (downtrend)

    This is a FREE alternative to computing MAs from OHLCV — ISE provides
    pre-computed MAs as authoritative data without any OHLCV history needed.
    """
    if not isinstance(stock_data, dict):
        return 0.0

    tech_data = stock_data.get("stockTechnicalData", [])
    if not isinstance(tech_data, list) or not tech_data:
        return 0.0

    # Get current price
    if current_price is None:
        price_dict = stock_data.get("currentPrice") or {}
        if isinstance(price_dict, dict):
            raw_p = price_dict.get("NSE") or price_dict.get("BSE")
            try:
                current_price = float(raw_p) if raw_p else None
            except (ValueError, TypeError):
                current_price = None

    # Also try stockDetailsReusableData for more precise intraday price
    if current_price is None:
        srd = stock_data.get("stockDetailsReusableData") or {}
        if isinstance(srd, dict):
            raw_p = srd.get("price") or srd.get("close")
            try:
                current_price = float(raw_p) if raw_p else None
            except (ValueError, TypeError):
                current_price = None

    if current_price is None or current_price <= 0:
        return 0.0

    # Parse MA values from ISE
    ma_values = []
    for item in tech_data:
        if not isinstance(item, dict):
            continue
        raw_ma = item.get("nsePrice") or item.get("bsePrice")
        try:
            ma_values.append(float(raw_ma))
        except (ValueError, TypeError):
            pass

    if not ma_values:
        return 0.0

    above = sum(1 for ma in ma_values if current_price > ma)
    total = len(ma_values)
    # Normalise to [-1, +1]
    return round((2 * above / total) - 1.0, 3)


# =============================================================================
# 7. FUNDAMENTAL QUALITY SCORE  ← NEW v2.0
# =============================================================================

def fundamental_quality_score(stock_data: Dict[str, Any]) -> float:
    """
    Composite fundamental quality from keyMetrics.growth and keyMetrics.margins.
    Returns score in [0, 1]:
      0.7+ = strong quality (high margin, growing earnings)
      0.5  = neutral / data unavailable
      0.3- = poor quality (declining margins, shrinking earnings)

    Uses:
      - EPS growth rate 5-year (ePSGrowthRate5Year) — long-term quality
      - Revenue growth rate 5-year (revenueGrowthRate5Year) — top-line momentum
      - Operating margin TTM (operatingMarginTrailing12Month) — profitability
    """
    if not isinstance(stock_data, dict):
        return 0.5

    km = stock_data.get("keyMetrics")
    if not isinstance(km, dict):
        return 0.5

    def _extract(section_key: str, metric_key: str) -> Optional[float]:
        section = km.get(section_key, [])
        if not isinstance(section, list):
            return None
        for item in section:
            if isinstance(item, dict) and item.get("key") == metric_key:
                try:
                    return float(item.get("value") or 0)
                except (ValueError, TypeError):
                    return None
        return None

    scores = []

    # EPS growth 5yr: 20%+ = excellent (1.0), 0% = neutral (0.5), negative = poor (0.2)
    eps_growth = _extract("growth", "ePSGrowthRate5Year")
    if eps_growth is not None:
        if eps_growth >= 20:   scores.append(1.0)
        elif eps_growth >= 10: scores.append(0.75)
        elif eps_growth >= 0:  scores.append(0.5)
        elif eps_growth >= -5: scores.append(0.35)
        else:                  scores.append(0.2)

    # Revenue growth 5yr: 12%+ = good, 5%+ = ok, <0% = poor
    rev_growth = _extract("growth", "revenueGrowthRate5Year")
    if rev_growth is not None:
        if rev_growth >= 12:   scores.append(0.9)
        elif rev_growth >= 7:  scores.append(0.7)
        elif rev_growth >= 3:  scores.append(0.55)
        elif rev_growth >= 0:  scores.append(0.45)
        else:                  scores.append(0.25)

    # Operating margin TTM: 20%+ = excellent, 10%+ = good, <5% = weak
    op_margin = _extract("margins", "operatingMarginTrailing12Month")
    if op_margin is not None:
        if op_margin >= 20:   scores.append(0.95)
        elif op_margin >= 12: scores.append(0.75)
        elif op_margin >= 7:  scores.append(0.55)
        elif op_margin >= 3:  scores.append(0.40)
        else:                  scores.append(0.25)

    if not scores:
        return 0.5
    return round(sum(scores) / len(scores), 4)


# =============================================================================
# 8. NEWS SENTIMENT
# =============================================================================

def news_sentiment(stock_data: Dict[str, Any]) -> float:
    """
    Computes polarity [-1, +1] using recent news from /stock.
    Falls back to 0.0 if textblob unavailable or no news.
    """
    if not _TEXTBLOB_OK or not isinstance(stock_data, dict):
        return 0.0

    news = stock_data.get("recentNews") or []
    if not isinstance(news, list) or not news:
        return 0.0

    scores = []
    for item in news:
        text = ""
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            headline = item.get("headline") or item.get("title") or ""
            body     = item.get("summary")  or item.get("description") or ""
            text     = f"{headline} {body}".strip()

        if text:
            try:
                scores.append(TextBlob(text).sentiment.polarity)
            except Exception:
                pass

    return round(sum(scores) / len(scores), 4) if scores else 0.0


# =============================================================================
# 9. INSTITUTIONAL DELTA
# =============================================================================

def institutional_delta(stock_data: Dict[str, Any]) -> Optional[float]:
    """
    QoQ delta in (FII + Promoter) combined holdings %.
    Positive = institutions are accumulating. Negative = distributing.
    """
    try:
        sh_list = stock_data.get("shareholding") or []
        if not isinstance(sh_list, list) or not sh_list:
            return None

        def _pct(entry: Dict, period: int = -1) -> float:
            cats = entry.get("categories") or []
            if not isinstance(cats, list) or len(cats) < abs(period):
                return 0.0
            raw = cats[period].get("percentage", 0) or 0
            return float(raw)

        target_names = {"fii", "promoter"}
        inst_entries = [
            e for e in sh_list
            if isinstance(e, dict)
            and str(e.get("displayName", "")).lower() in target_names
        ]

        if not inst_entries:
            return None

        all_have_prev = all(len(e.get("categories") or []) >= 2 for e in inst_entries)
        if not all_have_prev:
            return None

        latest = sum(_pct(e, -1) for e in inst_entries)
        prev   = sum(_pct(e, -2) for e in inst_entries)
        return max(-20.0, min(20.0, round(latest - prev, 4)))

    except Exception as exc:
        logger.debug(f"institutional_delta failed: {exc}")
        return None


# =============================================================================
# 10. COMPOSITE CONFIDENCE MODIFIER  (v2.0 — range ±0.20)
# =============================================================================

def compute_confidence_modifier(
    stock_data: Dict[str, Any],
    target_data: Optional[Dict[str, Any]],
    base_confidence: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Ties all 7 ISE signals together as a (+/- 0.20) modifier on ML confidence.

    v2.0 upgrade from v1.x:
      - Range expanded ±0.08 → ±0.20
      - Added target_revision, upgrade_momentum, ma_alignment, fundamental_quality
      - analyst_gap thresholds tiered (>5%, >10%, >20%) for better granularity

    Modifier table per signal:
      analyst_gap > 20%:          +0.07  (massive analyst upside)
      analyst_gap > 10%:          +0.05
      analyst_gap > 5%:           +0.03
      analyst_gap < -10%:         -0.08  (analysts below current = warning)
      analyst_gap < -5%:          -0.04
      target_revision_90d > 5%:   +0.04  (analysts raising targets)
      target_revision_90d > 2%:   +0.02
      target_revision_90d < -5%:  -0.04  (analysts cutting targets)
      target_revision_90d < -2%:  -0.02
      conviction >= 0.80:         +0.03  (strong buy consensus)
      conviction >= 0.65:         +0.02  (buy consensus)
      conviction < 0.30:          -0.03  (underperform/sell consensus)
      upgrade_momentum >= 3:      +0.04  (multiple upgrades in 3 months)
      upgrade_momentum >= 1:      +0.02  (some upgrades)
      upgrade_momentum <= -3:     -0.04  (multiple downgrades)
      upgrade_momentum <= -1:     -0.02  (some downgrades)
      news_sentiment > 0.15:      +0.02
      news_sentiment < -0.15:     -0.03
      institutional_delta > 1%:   +0.02  (FII+promoter accumulating)
      institutional_delta < -1%:  -0.02  (distributing)
      ma_alignment > 0.5:         +0.02  (above most MAs — confirmed trend)
      ma_alignment < -0.5:        -0.03  (below most MAs — downtrend)
      fundamental_quality > 0.75: +0.02  (high quality earnings/margins)
      fundamental_quality < 0.35: -0.02  (poor quality)

    Hard cap: modifier clamped to [-0.20, +0.20].
    """
    breakdown = {}
    modifier  = 0.0

    # ── 1. Analyst gap ────────────────────────────────────────────────────────
    gap = analyst_gap(stock_data, target_data)
    breakdown["analyst_gap"] = gap
    if gap is not None:
        if gap > 0.20:    modifier += 0.07
        elif gap > 0.10:  modifier += 0.05
        elif gap > 0.05:  modifier += 0.03
        elif gap < -0.10: modifier -= 0.08
        elif gap < -0.05: modifier -= 0.04

    # ── 2. Analyst target price revision (90d) ────────────────────────────────
    target_rev = analyst_target_revision(target_data)
    breakdown["target_revision_90d"] = round(target_rev, 4) if target_rev is not None else None
    if target_rev is not None:
        if target_rev > 0.05:    modifier += 0.04
        elif target_rev > 0.02:  modifier += 0.02
        elif target_rev < -0.05: modifier -= 0.04
        elif target_rev < -0.02: modifier -= 0.02

    # ── 3. Analyst conviction (rating level) ─────────────────────────────────
    conv = analyst_conviction(stock_data, target_data)
    breakdown["analyst_conviction"] = round(conv, 4)
    if conv >= 0.80:   modifier += 0.03
    elif conv >= 0.65: modifier += 0.02
    elif conv < 0.30:  modifier -= 0.03

    # ── 4. Analyst upgrade momentum (3-month trend) ───────────────────────────
    upg_mom = analyst_upgrade_momentum(stock_data)
    breakdown["upgrade_momentum"] = upg_mom
    if upg_mom >= 3:    modifier += 0.04
    elif upg_mom >= 1:  modifier += 0.02
    elif upg_mom <= -3: modifier -= 0.04
    elif upg_mom <= -1: modifier -= 0.02

    # ── 5. News sentiment ─────────────────────────────────────────────────────
    sent = news_sentiment(stock_data)
    breakdown["news_sentiment"] = sent
    if sent > 0.15:    modifier += 0.02
    elif sent < -0.15: modifier -= 0.03

    # ── 6. Institutional delta (FII + Promoter) ───────────────────────────────
    inst = institutional_delta(stock_data)
    breakdown["institutional_delta"] = inst
    if inst is not None:
        if inst > 1.0:   modifier += 0.02
        elif inst < -1.0: modifier -= 0.02

    # ── 7. Moving average alignment (pre-computed MAs from ISE) ──────────────
    ma_score = moving_average_alignment(stock_data)
    breakdown["ma_alignment"] = round(ma_score, 3)
    if ma_score > 0.5:    modifier += 0.02   # above most MAs — confirmed trend
    elif ma_score < -0.5: modifier -= 0.03   # below most MAs — downtrend signal

    # ── 8. Fundamental quality (growth + margins) ─────────────────────────────
    qual = fundamental_quality_score(stock_data)
    breakdown["fundamental_quality"] = round(qual, 4)
    if qual >= 0.75: modifier += 0.02
    elif qual < 0.35: modifier -= 0.02

    # ── Apply cap and compute adjusted confidence ─────────────────────────────
    modifier = max(-0.20, min(0.20, modifier))
    adjusted = max(0.0, min(1.0, base_confidence + modifier))

    breakdown["modifier"]             = round(modifier, 4)
    breakdown["base_confidence"]      = base_confidence
    breakdown["adjusted_confidence"]  = adjusted

    return adjusted, breakdown


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _parse_date(raw: str) -> Optional[date]:
    if not isinstance(raw, str):
        return None
    formats = ["%d %b %Y", "%Y-%m-%d", "%b %d, %Y", "%d/%m/%Y", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    match = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    if match:
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            pass
    return None
