"""
MARK5 ISE SIGNAL ENRICHER v1.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enriches ML-generated signals with premium ISE fundamentals.
Implements the DUAL-FETCH strategy: /stock + /stock_target_price.

POSITION IN PIPELINE:
  1. ML Model ranks candidates (confidence >= 55%)
  2. [THIS] ISE Enrichment Layer:
     - Automates RULE 25 (Veto if board meeting or results within 3 days)
     - Applies Fundamental Modifier (+/- 0.08 confidence)
  3. Final re-ranked candidates passed to order generator.

TOKEN BUDGET:
  2 calls per candidate x 5 candidates = 10 calls/day.
  10 x 22 trading days = 220 tokens/month (budget = 500).
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .ise_adapter import ISEAdapter, TokenBudgetExceeded
from .ise_feature_factory import (
    is_safe_to_trade,
    compute_confidence_modifier,
)

logger = logging.getLogger("MARK5.ISEEnricher")

MIN_CONFIDENCE = 0.55   # default kept for back-compat with direct callers
MAX_SIGNALS     = 10    # increased from 5: enricher now processes all top-10

@dataclass
class EnrichedSignal:
    ticker:              str
    base_confidence:     float
    adjusted_confidence: float
    is_safe:             bool
    veto_reason:         str
    ise_breakdown:       Dict[str, Any] = field(default_factory=dict)
    raw_stock_data:      Dict[str, Any] = field(default_factory=dict)
    raw_target_data:     Dict[str, Any] = field(default_factory=dict)
    _min_confidence:     float = field(default=MIN_CONFIDENCE, repr=False)

    @property
    def tradeable(self) -> bool:
        return self.is_safe and self.adjusted_confidence >= self._min_confidence



class ISESignalEnricher:
    """Orchestrates dual-endpoint ISE enrichment for candidate trades."""

    def __init__(self, api_key: Optional[str] = None):
        self._ise = ISEAdapter(api_key=api_key)

    def enrich(
        self,
        candidates: List[Tuple[str, float]],
        reference_date: Optional[date] = None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> Tuple[List[EnrichedSignal], List[EnrichedSignal], Dict[str, int]]:
        """
        Enrich candidates by fetching premium ISE data.

        Args:
            candidates:     list of (ticker, base_ml_confidence) pairs
            reference_date: date for Rule 25 blackout check
            min_confidence: threshold for tradeable property. Pass 0.0 from
                            rank_live.py to get ALL stocks back and apply the
                            final threshold externally (correct architecture).

        Returns: (tradeable_signals_ranked, vetoed_signals, budget)
        """
        tradeable: List[EnrichedSignal] = []
        vetoed:    List[EnrichedSignal] = []

        if not candidates:
            return tradeable, vetoed, self._ise.get_budget_status()

        logger.info(
            f"ISE Enrichment: {len(candidates)} candidates | "
            f"min_conf={min_confidence:.0%} | "
            f"Budget: {self._ise.get_budget_status()['remaining']} left"
        )

        for ticker, base_conf in candidates:
            bare = ticker.replace(".NS", "").replace(".BO", "")
            sig  = self._enrich_one(bare, ticker, base_conf, reference_date, min_confidence)

            if sig.tradeable:
                tradeable.append(sig)
            else:
                vetoed.append(sig)

        # Sort by adjusted confidence DESC
        tradeable.sort(key=lambda s: s.adjusted_confidence, reverse=True)
        return tradeable[:MAX_SIGNALS], vetoed, self._ise.get_budget_status()

    def _enrich_one(
        self,
        bare: str,
        full_ticker: str,
        base_confidence: float,
        reference_date: Optional[date],
        min_confidence: float = MIN_CONFIDENCE,
    ) -> EnrichedSignal:
        """Fetches dual endpoints and applies factory logic."""
        try:
            status = self._ise.get_budget_status()
            if status["remaining"] < 1:
                return self._neutral(full_ticker, base_confidence, "Budget Exhausted", min_confidence)

            stock_data  = self._ise.fetch_stock(bare)
            target_data = self._ise.fetch_stock_target_price(bare)

            if not stock_data:
                return self._neutral(full_ticker, base_confidence, "API returned no stock data", min_confidence)

            is_safe, veto_reason = is_safe_to_trade(stock_data, reference_date=reference_date)
            adjusted, breakdown  = compute_confidence_modifier(stock_data, target_data, base_confidence)

            return EnrichedSignal(
                ticker=full_ticker,
                base_confidence=base_confidence,
                adjusted_confidence=adjusted,
                is_safe=is_safe,
                veto_reason=veto_reason,
                ise_breakdown=breakdown,
                raw_stock_data=stock_data,
                raw_target_data=target_data,
                _min_confidence=min_confidence,
            )

        except TokenBudgetExceeded:
            return self._neutral(full_ticker, base_confidence, "Budget Exceeded", min_confidence)
        except Exception as exc:
            logger.error(f"Enrichment error for {bare}: {exc}", exc_info=True)
            return self._neutral(full_ticker, base_confidence, f"Code Error: {type(exc).__name__}", min_confidence)

    @staticmethod
    def _neutral(ticker: str, confidence: float, reason: str, min_confidence: float = MIN_CONFIDENCE) -> EnrichedSignal:
        """Fallback to original confidence if ISE fails or is blocked."""
        return EnrichedSignal(
            ticker=ticker,
            base_confidence=confidence,
            adjusted_confidence=confidence,
            is_safe=True,
            veto_reason=f"Neutral Fallback: {reason}",
            ise_breakdown={"modifier": 0.0, "note": reason},
            _min_confidence=min_confidence,
        )

