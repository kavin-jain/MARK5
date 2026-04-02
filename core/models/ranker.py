"""
MARK5 CROSS-SECTIONAL RANKER v2.1 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-03-16] v1.0: Initial implementation
- [2026-03-16] v2.0: Added get_ml_filtered_positions() — combines ranking
                     (Layer 1) with MARK5Predictor ML gate (Layer 2).
                     score_stock(), rank_universe(), get_target_positions()
                     are UNCHANGED.
- [2026-03-20] v2.1: Wired wick-confirmed entry multiplier from
                     AdvancedFeatureEngine.get_wick_confirmed_entry().
                     When the prior day was a bullish rejection candle and
                     today confirms with volume, ml_confidence is boosted
                     by 15%, capped at 0.95.

TRADING ROLE: Cross-sectional momentum ranking + ML entry filter.
              Layer 1: ranks 50 stocks, selects top-10 universe.
              Layer 2: ML confidence gate on top-10 only.
              Layer 3: triple barrier exits managed in test.py/rank_live.py.
SAFETY LEVEL: HIGH
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

from core.models.features import AdvancedFeatureEngine

logger = logging.getLogger("MARK5.Ranker")
_feature_engine = AdvancedFeatureEngine()


class CrossSectionalRanker:
    """
    Ranks universe of stocks by composite momentum score.
    Primary signal for portfolio construction.

    Based on:
    - 12-1 month momentum (Sehgal & Jain 2011, NSE documented alpha)
    - FII flow integration (NSE market pulse data)
    - Trend quality via Efficiency Ratio (Kaufman)
    """

    def __init__(self, top_n: int = 3, min_history_days: int = 252):
        self.top_n = top_n
        self.min_history_days = min_history_days

    def score_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        nifty_close: Optional[pd.Series],
        fii_net: Optional[pd.Series],
        current_date: pd.Timestamp,
    ) -> Optional[float]:
        """Legacy per-stock scorer kept for back-compat. rank_universe() now
        uses cross-sectional z-scoring via _raw_signals(). Calling this
        directly will not reflect the cross-sectional normalisation."""
        sigs = self._raw_signals(symbol, df, nifty_close, fii_net, current_date)
        if sigs is None:
            return None
        # Simple weighted sum without cross-sectional normalisation
        return (
            0.40 * sigs['rel_20d'] * 5.0 +
            0.20 * (sigs['dist_52w'] - 0.8) +
            0.20 * sigs['er'] +
            0.10 * (sigs['vol_ratio'] - 1.0) +
            0.10 * (sigs['fii_z'] / 3.0)
        )

    def _raw_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        nifty_close: Optional[pd.Series],
        fii_net: Optional[pd.Series],
        current_date: pd.Timestamp,
    ) -> Optional[dict]:
        """
        Compute raw (un-normalised) signal components for cross-sectional scoring.

        Returns a dict of signal values, or None if insufficient history.
        Cross-sectional z-scoring is applied in rank_universe() over the full
        universe — do NOT normalise here.
        """
        if current_date.tzinfo is not None:
            current_date = current_date.tz_localize(None)
        idx = df.index.tz_localize(None) if df.index.tz else df.index
        hist = df.loc[idx <= current_date]
        if len(hist) < self.min_history_days:
            return None

        close  = hist['close'].values.astype(float)
        high   = hist['high'].values.astype(float)
        low    = hist['low'].values.astype(float)
        volume = hist['volume'].values.astype(float)

        # ── Signal 1: Short-term relative momentum (20d) ───────────────
        # 20d return vs NIFTY — measures recent relative outperformance.
        # NOTE: 12-1m absolute momentum was proven anti-predictive (IC=-0.08)
        # for NIFTY50 in 2023+. 20d relative momentum has better signal quality.
        if len(close) < 21:
            return None
        ret_20d = close[-1] / close[-21] - 1.0
        if nifty_close is not None:
            nifty_a = nifty_close.reindex(hist.index, method='ffill').values
            nifty_ret_20d = nifty_a[-1] / nifty_a[-21] - 1.0 if len(nifty_a) >= 21 else 0.0
            rel_20d = ret_20d - nifty_ret_20d
        else:
            rel_20d = ret_20d

        # ── Signal 2: Distance from 52-week high (proximity factor) ────
        # Rule 29: stocks within 3% of 52w high get +0.05 confidence bonus.
        high_252 = np.max(high[-252:]) if len(high) >= 252 else np.max(high)
        dist_52w = close[-1] / high_252  # 1.0 = AT 52w high

        # ── Signal 3: Individual stock trend quality (Efficiency Ratio) ──
        window_er = min(50, len(close) - 1)
        net_move  = abs(close[-1] - close[-1 - window_er])
        total_path = np.sum(np.abs(np.diff(close[-1 - window_er:]))) + 1e-9
        er = float(np.clip(net_move / total_path, 0, 1))

        # ── Signal 4: Individual stock ADX (14) — simplified DX ────────
        if len(high) >= 15:
            tr_arr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])
                )
            )
            dm_plus  = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                                np.maximum(high[1:] - high[:-1], 0), 0)
            dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                                np.maximum(low[:-1] - low[1:], 0), 0)
            di14p = float(pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().iloc[-1])
            di14m = float(pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().iloc[-1])
            adx14 = abs(di14p - di14m) / (di14p + di14m + 1e-9) * 100
        else:
            adx14 = 15.0

        # ── Signal 5: Volume surge (5d vs 63d) ─────────────────────────
        vol_ratio = float(np.clip(
            np.mean(volume[-5:]) / (np.mean(volume[-63:]) + 1e-9), 0.3, 3.0
        ))

        # ── Signal 6: FII flow (systemic tailwind) ──────────────────────
        fii_z = 0.0
        if fii_net is not None and len(fii_net) >= 20:
            fii_a = fii_net.reindex(hist.index, method='ffill').fillna(0).values
            fii_10d = float(np.sum(fii_a[-10:]))
            fii_std = float(np.std(fii_a[-252:]) + 1e-9)
            fii_z = float(np.clip(fii_10d / fii_std, -3, 3))

        return {
            'rel_20d':    rel_20d,
            'dist_52w':   dist_52w,
            'er':         er,
            'adx14':      adx14,
            'vol_ratio':  vol_ratio,
            'fii_z':      fii_z,
            'symbol':     symbol,
        }

    def rank_universe(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        nifty_close: Optional[pd.Series],
        fii_net: Optional[pd.Series],
        current_date: pd.Timestamp,
    ) -> List[Tuple[str, float]]:
        """
        Rank all stocks by CROSS-SECTIONAL composite score (v3.0).

        Signals are z-scored across the universe before weighting.
        A stock ranks high only if it outperforms PEERS on that date —
        removing the absolute-momentum regime sign-flip (IC=-0.079 for
        NIFTY50 2023+ with 12-1m absolute momentum).

        Returns sorted list of (symbol, score) descending.
        """
        raw = {}
        for symbol, df in stocks_data.items():
            try:
                sig = self._raw_signals(symbol, df, nifty_close, fii_net, current_date)
                if sig is not None:
                    raw[symbol] = sig
            except Exception as e:
                logger.warning(f"Signal computation failed for {symbol}: {e}")

        if not raw:
            return []

        # Cross-sectional z-score each signal across the universe
        signal_keys = ['rel_20d', 'dist_52w', 'er', 'vol_ratio', 'fii_z']
        syms   = list(raw.keys())
        arrays = {k: np.array([raw[s][k] for s in syms]) for k in signal_keys}
        zs     = {}
        for k, arr in arrays.items():
            mu, sigma = arr.mean(), arr.std()
            zs[k] = (arr - mu) / (sigma + 1e-9)

        scores = {}
        for i, sym in enumerate(syms):
            composite = (
                0.40 * zs['rel_20d'][i]   +   # cross-sectional relative momentum
                0.20 * zs['dist_52w'][i]  +   # 52w high proximity vs peers (Rule 29)
                0.20 * zs['er'][i]        +   # trend quality
                0.10 * zs['vol_ratio'][i] +   # volume confirmation
                0.10 * zs['fii_z'][i]         # FII tailwind
            )
            # ADX multiplier: confirmed trend stocks get full score (Rule 23 at stock level)
            adx14 = raw[sym]['adx14']
            if adx14 >= 25:
                mult = 1.10
            elif adx14 >= 20:
                mult = 1.00
            elif adx14 >= 15:
                mult = 0.85
            else:
                mult = 0.70
            scores[sym] = float(composite * mult)

        ranked     = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        num_stocks = len(ranked)
        top_str    = [s for s, _ in ranked[:min(3, num_stocks)]]
        bottom_str = [s for s, _ in ranked[-min(3, num_stocks):]]
        logger.info(
            f"Ranked {num_stocks} stocks | "
            f"Top: {top_str} | "
            f"Bottom: {bottom_str}"
        )
        return ranked

    def get_target_positions(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        nifty_close: Optional[pd.Series],
        fii_net: Optional[pd.Series],
        current_date: pd.Timestamp,
        ml_vetoes: Optional[Dict[str, bool]] = None,
        nifty_bear: bool = False,
    ) -> List[str]:
        """
        Returns list of symbols to hold as of current_date.
        Applies ML veto and bear market gate.
        """
        if nifty_bear:
            effective_n = 1
        else:
            effective_n = self.top_n

        ranked = self.rank_universe(stocks_data, nifty_close, fii_net, current_date)
        ml_vetoes = ml_vetoes or {}

        selected = []
        for symbol, score in ranked:
            if len(selected) >= effective_n:
                break
            if ml_vetoes.get(symbol, False):
                logger.info(f"ML veto applied: {symbol} excluded from top-{effective_n}")
                continue
            selected.append(symbol)

        return selected

    # ── NEW v2.0 ────────────────────────────────────────────────────────────

    def get_ml_filtered_positions(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        nifty_close: Optional[pd.Series],
        fii_net: Optional[pd.Series],
        current_date: pd.Timestamp,
        all_predictors: Dict,
        nifty_bear: bool = False,
        confidence_threshold: float = 0.55,
        ranking_top_n: int = 10,
    ) -> List[Tuple[str, float, float]]:
        """
        Combined Layer 1 (ranking) + Layer 2 (ML gate).

        Step 1: Rank all stocks, take top-`ranking_top_n` (default 10).
        Step 2: For each top-10 stock, run MARK5Predictor.
                Only stocks with ML confidence >= threshold pass.
        Step 3: Sort passing stocks by ML confidence (not rank score).
                Return top-N by effective_n.

        Returns list of (symbol, rank_score, ml_confidence).
        Empty list = hold cash.

        Args:
            all_predictors: dict of {symbol: MARK5Predictor instance}
            nifty_bear:     if True, tightens ML threshold to 0.70 and
                            reduces effective_n to 1 (RULE 23)
            confidence_threshold: minimum ML confidence to enter (default 0.55)
            ranking_top_n:  how many top-ranked stocks to pass to ML gate
        """
        # Bear market overrides
        min_conf = 0.70 if nifty_bear else confidence_threshold
        effective_n = 1 if nifty_bear else self.top_n

        # Layer 1: rank full universe, take top-N eligible
        ranked = self.rank_universe(stocks_data, nifty_close, fii_net, current_date)
        top_eligible = ranked[:ranking_top_n]

        logger.info(
            f"ML filter | date={current_date.date()} | "
            f"bear={nifty_bear} | threshold={min_conf:.0%} | "
            f"eligible={len(top_eligible)}"
        )

        # Layer 2: ML gate on top-eligible only
        candidates = []
        passed = 0
        vetoed = 0

        for symbol, rank_score in top_eligible:
            predictor = all_predictors.get(symbol)

            # C-4: No model for this stock — EXCLUDE rather than fail-open.
            # Allowing entry at confidence_threshold would let unmodelled stocks
            # rank ahead of stocks that genuinely passed the ML gate.
            if predictor is None or predictor._container is None:
                vetoed += 1
                logger.info(f"  ⛔ {symbol}: no trained model — excluded (fail-closed)")
                continue

            try:
                # Feed historical data up to current_date to predictor
                hist = stocks_data[symbol]
                if hist.index.tz is not None:
                    hist = hist.copy()
                    hist.index = hist.index.tz_localize(None)
                hist_to_date = hist.loc[hist.index <= current_date].tail(300)

                if len(hist_to_date) < 30:
                    logger.debug(f"  {symbol}: insufficient history for ML")
                    continue

                result = predictor.predict(hist_to_date)
                confidence = float(result.get('confidence', 0.0))

                # Wick-confirmed entry boost (v2.1)
                # If the prior day showed a bullish rejection candle and today
                # confirms with volume, raise confidence by 15% (cap 0.95).
                # This never creates a BUY signal — it only reinforces one that
                # already cleared the ML threshold.
                try:
                    wick_confirm = _feature_engine.get_wick_confirmed_entry(hist_to_date)
                    if wick_confirm.iloc[-1]:
                        boosted = min(confidence * 1.15, 0.95)
                        logger.debug(
                            f"  📐 {symbol}: wick boost {confidence:.2%} → {boosted:.2%}"
                        )
                        confidence = boosted
                except Exception as _wick_err:
                    logger.debug(f"  {symbol}: wick check skipped ({_wick_err})")

                if confidence >= min_conf:
                    candidates.append((symbol, rank_score, confidence))
                    passed += 1
                    logger.info(
                        f"  ✅ {symbol:<20} rank={rank_score:+.3f} "
                        f"ml_conf={confidence:.2%} → PASS"
                    )
                else:
                    vetoed += 1
                    logger.info(
                        f"  ❌ {symbol:<20} rank={rank_score:+.3f} "
                        f"ml_conf={confidence:.2%} → VETO (< {min_conf:.0%})"
                    )

            except Exception as e:
                logger.warning(f"  {symbol}: ML prediction failed ({e})")
                continue

        logger.info(
            f"ML gate result: {passed} passed, {vetoed} vetoed "
            f"out of {len(top_eligible)} eligible"
        )

        if not candidates:
            logger.warning("ML gate: 0 stocks passed — holding cash this period")
            return []

        # Sort by ML confidence descending — confidence quality > momentum rank
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:effective_n]