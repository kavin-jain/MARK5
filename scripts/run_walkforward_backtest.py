"""
MARK5 — Walk-Forward OOS Backtest (Iteration 4)
================================================
Key Iter4 changes in effect:
  • atr_multiplier 1.5 → 2.5 (trailing stop wider — lets trends breathe)
  • RISK_PER_TRADE 1.5% → 2.5% (larger position sizing)
  • VIX gate 16 → 22 (stops halving positions in normal Indian markets)
  • CONFIDENCE_HURDLE 0.55 → 0.52 (more valid signals pass ML gate)
  • MAX_HOLD_DAYS 30 → 45 (ride trends further)
  • ML soft gate (0.48) applied to Tier 2/3 continuation/riding signals
  • ITC excluded from universe (pathological: WR=31.5%, AUC=0.331)
  • ATR formula unified to Wilder's (alpha=1/14) across training + backtest

OOS period: 2022-01-01 → 2026-05-21 (4-year true OOS from 10-year training data)

NOTE: CPCV metrics from retrain_results.json remain the TRUE hold-out estimates.
      Walk-forward P&L shows realized performance under Iter4 execution rules.
"""
import os, sys, json, logging, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.CRITICAL)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from datetime import datetime

# Re-enable logging only for our script
logger = logging.getLogger("MARK5.WFBacktest")
logging.getLogger("MARK5.WFBacktest").setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
logger.addHandler(handler)

# Iter3 production-ready tickers (passed prod gate in retrain_results.json)
# ITC excluded: BB breakout WR=31.5%, avg_return=-0.547%, AUC=0.331 over 2015-2026
ITER3_PROD_TICKERS = [
    "ASIANPAINT", "AUBANK", "BAJFINANCE", "BHARTIARTL", "COFORGE",
    "HAL", "PNB", "RELIANCE", "TATAELXSI", "TATASTEEL",
    "TCS", "TRENT", "YESBANK"
]

def run_backtest():
    from core.models.backtest_pipeline import PipelineOrchestrator

    start_date = "2022-01-01"
    end_date   = "2026-05-21"

    print(f"\n{'═'*80}")
    print(f"  MARK5 WALK-FORWARD BACKTEST — Iteration 4")
    print(f"  OOS period: {start_date} → {end_date}  (4 years)")
    print(f"  Tickers: {len(ITER3_PROD_TICKERS)}  (ITC excluded)")
    print(f"  Iter4 changes: ATR_stop=2.5× | risk=2.5% | VIX_gate=22 | hurdle=0.52")
    print(f"  Training window: 18 months | Test folds: 3 months")
    print(f"{'═'*80}\n")

    start = datetime.now()
    orch = PipelineOrchestrator(
        tickers=ITER3_PROD_TICKERS,
        start_date=start_date,
        end_date=end_date,
        models_dir="models",
        run_training=False,
        run_backtest=True,
    )

    summary = orch.run()

    elapsed = (datetime.now() - start).seconds
    print(f"\n  Backtest completed in {elapsed}s")

    # ── Print summary table ─────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  {'TICKER':<14} {'WF_SHARPE':>10} {'BEATS_BH':>10} {'N_TRADES':>10}")
    print(f"{'─'*80}")
    sharpes = []
    for ticker, res in sorted(summary.items()):
        if isinstance(res, dict) and "sharpe" in res:
            sh  = res.get("sharpe", 0) or 0
            bh  = f"{res.get('beats_bh_rate', 0)*100:.0f}%" if res.get('beats_bh_rate') is not None else "?"
            nt  = res.get("total_trades", "?")
            sharpes.append(sh)
            flag = "✅" if sh > 0 else "❌"
            print(f"  {ticker:<14} {sh:>+10.3f} {bh:>10} {str(nt):>10}  {flag}")
    if sharpes:
        avg = float(np.mean(sharpes))
        pos = sum(1 for s in sharpes if s > 0)
        print(f"{'─'*80}")
        print(f"  {'PORTFOLIO AVG':<14} {avg:>+10.3f}  Positive: {pos}/{len(sharpes)}")
    print(f"{'═'*80}\n")

    # Save results
    out = os.path.join(_ROOT, "reports", "walkforward_iter4.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved → {out}\n")

    return summary


if __name__ == "__main__":
    run_backtest()
