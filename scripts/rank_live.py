"""
MARK5 LIVE RANKING SCRIPT v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run every Friday after 3:30 PM IST.

CHANGELOG:
- v1.0: Pure ranking output (top-3 by momentum score)
- v2.0: Added ML Layer 2 filter.
        Shows top-10 ranked universe with ML confidence per stock.
        Final BUY list = top-3 by ML confidence from within top-10.

OUTPUT:
  Layer 1 — Top 10 ranked stocks with momentum scores
  Layer 2 — ML confidence for each, pass/fail vs 0.55 threshold
  Final   — Stocks to enter Monday open (max 3, or 1 in bear market)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import pandas as pd
import numpy as np
from datetime import datetime

from core.models.ranker import CrossSectionalRanker
from core.models.predictor import MARK5Predictor
from core.data.adapters.kite_adapter import KiteFeedAdapter
from core.data.fii_data import FIIDataProvider
from scripts.nifty50_universe import MARK5_LIVE_TICKERS
from core.data.adapters import ISESignalEnricher

# ── CONFIG ─────────────────────────────────────────────────────────────────
ML_CONFIDENCE_THRESHOLD = 0.55   # normal market
ML_CONFIDENCE_BEAR      = 0.70   # bear market (RULE 23)
RANKING_TOP_N           = 15     # top-N from ranker — raised (larger universe)
MAX_POSITIONS           = 3      # max holdings in normal market
# ───────────────────────────────────────────────────────────────────────────



def _compute_atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    """ATR as fraction of price — used for position-level stop monitoring."""
    if len(df) < window + 1:
        return 0.02
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1])
    )
    tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
    atr = float(np.mean(tr[-window:]))
    price = float(closes[-1])
    return atr / price if price > 0 else 0.02


def _is_bear_market(nifty_close: pd.Series, current_date: pd.Timestamp) -> bool:
    """RULE 23: NIFTY below 200-day EMA AND 20-day return < -5%."""
    hist = nifty_close.loc[nifty_close.index <= current_date]
    if len(hist) < 200:
        return False
    ema200      = float(hist.ewm(span=200, adjust=False).mean().iloc[-1])
    curr_nifty  = float(hist.iloc[-1])
    prev_nifty  = float(hist.iloc[-20])
    ret_20d     = (curr_nifty / prev_nifty) - 1.0
    return curr_nifty < ema200 and ret_20d < -0.05


def get_current_rankings():
    # ── 1. Connect to Kite and download data ───────────────────────────────
    kite = KiteFeedAdapter({})
    if not kite.connect():
        print("❌ Failed to connect to Kite. Check .env credentials.")
        return []

    fii = FIIDataProvider()
    ranker = CrossSectionalRanker(top_n=MAX_POSITIONS)

    today = pd.Timestamp.today().tz_localize('Asia/Kolkata')
    end_dt = today
    start_dt = end_dt - pd.Timedelta(days=700)

    print(f"📥 Loading MARK5 Live Universe ({len(MARK5_LIVE_TICKERS)} stocks) via Kite up to {today.date()}...")
    all_data = {}
    for sym in MARK5_LIVE_TICKERS:
        df = kite.fetch_ohlcv(sym, from_date=start_dt, to_date=end_dt, interval='day')
        if df is not None and not df.empty:
            df.columns = [str(c).lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            all_data[sym] = df

    print(f"  Loaded {len(all_data)}/{len(MARK5_LIVE_TICKERS)} stocks")

    print("📥 Fetching NIFTY50 index...")
    nifty = kite.fetch_index_data('NIFTY50', interval='day', days_back=700)
    if nifty is None or nifty.empty:
        print("❌ Failed to fetch NIFTY50 index data.")
        return []
    nifty.columns = [str(c).lower() for c in nifty.columns]
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)
    nifty_close = nifty['close']

    print("📥 Fetching FII data...")
    today_naive = today.tz_localize(None)
    fii_net = fii.get_fii_flow(
        start_date=str((today_naive - pd.Timedelta(days=400)).date()),
        end_date=str(today_naive.date())
    )

    # ── 2. Bear market gate ────────────────────────────────────────────────
    nifty_bear = _is_bear_market(nifty_close, today_naive)
    min_conf   = ML_CONFIDENCE_BEAR if nifty_bear else ML_CONFIDENCE_THRESHOLD
    eff_n      = 1 if nifty_bear else MAX_POSITIONS

    if nifty_bear:
        print(f"\n🐻 BEAR MARKET GATE ACTIVE — ML threshold raised to {min_conf:.0%}, max 1 position")

    # ── 3. Layer 1: Rank full universe ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MARK5 LAYER 1 — Top {RANKING_TOP_N} Ranked Universe ({today_naive.date()})")
    print(f"{'='*60}")

    ranked = ranker.rank_universe(all_data, nifty_close, fii_net, today_naive)
    top_10 = ranked[:RANKING_TOP_N]

    for i, (sym, score) in enumerate(top_10, 1):
        print(f"  {i:2d}. {sym:<22} score={score:+.4f}")

    # ── 4. Layer 2: ML Confidence Scoring (ALL top-10) ──────────────────────
    # DESIGN: Do NOT apply threshold here. Collect raw ML confidence for ALL
    # top-10 stocks, then pass everything to ISE enricher (Layer 3).
    # The final threshold is applied to ISE-ADJUSTED confidence, not raw ML.
    # This prevents ISE from being shut out in bear market (where ML < 70%
    # for every stock, making the enricher never fire).
    print(f"\n{'='*60}")
    print(f"MARK5 LAYER 2 — ML Confidence Scoring (all top-{RANKING_TOP_N})")
    print(f"{'='*60}")

    # Load all predictors up front
    all_predictors = {}
    for sym, _ in top_10:
        try:
            p = MARK5Predictor(sym)
            p.reload_artifacts()
            if p._container is not None:
                all_predictors[sym] = p
        except Exception:
            pass

    # Score ALL top-10, threshold applied AFTER ISE enrichment
    ml_scored = []   # all stocks with ML confidence
    for sym, rank_score in top_10:
        predictor = all_predictors.get(sym)
        if predictor is None:
            print(f"  ⚠️  {sym:<22} no ML model — skipping")
            continue

        try:
            hist    = all_data[sym].tail(300)
            result  = predictor.predict(hist)
            ml_conf = float(result.get('confidence', 0.0))
            atr_pct = _compute_atr_pct(all_data[sym])

            print(f"  {sym:<22} rank={rank_score:+.4f} | ml={ml_conf:.2%}")
            ml_scored.append({
                'symbol':     sym,
                'rank_score': rank_score,
                'confidence': ml_conf,
                'atr_pct':    atr_pct,
            })

        except Exception as e:
            print(f"  {sym:<22} ML error: {e}")

    # ── 4b. Layer 3: ISE Enrichment on ALL ML-scored stocks ─────────────────
    # Runs on ALL top-10, not just ML-passers. ISE modifier can push stocks
    # above threshold even if ML alone was insufficient.
    print(f"\n{'='*60}")
    print(f"MARK5 LAYER 3 — ISE Enrichment + Final Gate (≥{min_conf:.0%})")
    print(f"{'='*60}")

    enriched_tradeable = []
    vetoed = []
    if ml_scored:
        enricher = ISESignalEnricher()
        # Pass min_confidence=0.0 — ALL stocks flow through.
        # Final threshold applied below to ISE-adjusted confidence.
        candidates_tuples = [(c['symbol'], c['confidence']) for c in ml_scored]
        all_enriched, _, budget = enricher.enrich(
            candidates_tuples,
            min_confidence=0.0,
        )
        print(f"  ISE Budget Remaining: {budget['remaining']} tokens")

        # Now apply the regime-appropriate threshold to ISE-adjusted confidence
        for sig in all_enriched + vetoed:  # vetoed is empty since we use min=0
            pass  # handled below

        # Re-split based on final threshold on adjusted confidence
        for sig in all_enriched:
            mod_str = f"{sig.ise_breakdown.get('modifier', 0):+.3f}"
            if not sig.is_safe:
                print(f"  🚫 VETOED:   {sig.ticker:<20} | Rule 25: {sig.veto_reason}")
                vetoed.append(sig)
            elif sig.adjusted_confidence >= min_conf:
                orig_candidate = next(
                    (c for c in ml_scored if c['symbol'] == sig.ticker),
                    {'confidence': sig.base_confidence}
                )
                print(
                    f"  ✅ PASS:     {sig.ticker:<20} "
                    f"ml={sig.base_confidence:.2%} → ise={sig.adjusted_confidence:.2%} "
                    f"(mod {mod_str})"
                )
                enriched_tradeable.append(sig)
            else:
                gap_to_gate = min_conf - sig.adjusted_confidence
                print(
                    f"  ⏸️  WAIT:     {sig.ticker:<20} "
                    f"ml={sig.base_confidence:.2%} → ise={sig.adjusted_confidence:.2%} "
                    f"(mod {mod_str} | {gap_to_gate:.1%} below gate)"
                )
                vetoed.append(sig)


    # ── 5. Final allocation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MARK5 FINAL ALLOCATION — Enter Monday open")
    print(f"{'='*60}")

    # enriched_tradeable is already sorted by adjusted_confidence DESC
    final = enriched_tradeable[:eff_n]

    if not final:
        print("🚨 NO STOCKS PASSED ALL GATES — HOLD 100% CASH THIS WEEK")
        if vetoed:
            print(f"   (ISE Vetoes/Threshold Drops: {len(vetoed)})")
    else:
        for rank, sig in enumerate(final, 1):
            # Find the original ATR from ml_scored
            orig_candidate = next((c for c in ml_scored if c['symbol'] == sig.ticker), {'atr_pct': 0.02})
            atr_pct = orig_candidate['atr_pct']
            
            pt_price = atr_pct * 2.5 * 100
            sl_price = atr_pct * 1.5 * 100
            
            modifier_str = f"{'+' if sig.ise_breakdown.get('modifier', 0) >=0 else ''}{sig.ise_breakdown.get('modifier', 0):.2f}"
            
            print(
                f"  {rank}. 🟢 BUY {sig.ticker:<20} "
                f"conf={sig.adjusted_confidence:.2%} "
                f"(ISE mod: {modifier_str}) "
                f"PT=+{pt_price:.1f}% SL=-{sl_price:.1f}%"
            )

    # ── 6. Weakest stocks (avoid) ──────────────────────────────────────────
    print(f"\n🔴 Weakest momentum (avoid / exit if held):")
    for sym, score in ranked[-3:]:
        print(f"     {sym:<22} score={score:+.4f}")

    print(f"\n{'='*60}")
    print(f"ML gate summary: {len(enriched_tradeable)} passed / {len(top_10)} eligible")
    if len(enriched_tradeable) < 1:
        print("⚠️  Low pass rate — potential cash drag. Monitor for 2 more weeks.")

    return [p.ticker for p in final]


if __name__ == '__main__':
    holdings = get_current_rankings()
    if holdings:
        print(f"\n✅ Action for Monday open: BUY {', '.join(holdings)}")
    else:
        print("\n✅ Action for Monday open: HOLD CASH")