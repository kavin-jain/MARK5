#!/usr/bin/env python3
"""
MARK5 SURGICAL FIX APPLICATOR
==============================
Applies all 8 identified fixes from the codebase audit.
Run from project root: python apply_all_fixes.py

Fixes applied:
  FIX-1: Undertrained models — XGB/LGB/CAT iterations 50→500, early stop 10→50
  FIX-2: TCN undertrained — epochs 5→50 in fold training AND final retrain
  FIX-3: Production gate too lenient — P(Sharpe>1.5)≥40% → ≥70%, worst-5%≥-1.5 → ≥0.0
  FIX-4: Intrabar ambiguity — SL/PT ordering uses open-proximity, not elif order
  FIX-5: Degraded features — relative_strength_nifty/sector_rel_strength fallback 0.0 not raw returns
  FIX-6: Shadow models in backtests — backtest_150 refuses models that failed the gate
  FIX-7: Intraday simulation warning — compare_intraday_swing warns about daily-bar limitation
  FIX-8: Synthetic FII guard — trainer explicitly warns and zero-fills instead of randomising
"""

import os
import sys
import re
import shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT.endswith("mark5_fixes"):
    # Running from the fixes directory; assume project is one level up
    ROOT = os.path.dirname(ROOT)

BACKUP_DIR = os.path.join(ROOT, f"_fix_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

CHANGES: list[dict] = []  # populated by each fix function


# ─── helpers ──────────────────────────────────────────────────────────────────

def _path(*parts) -> str:
    return os.path.join(ROOT, *parts)


def _read(rel: str) -> str:
    with open(_path(rel), "r", encoding="utf-8") as f:
        return f.read()


def _write(rel: str, content: str) -> None:
    full = _path(rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    bak = os.path.join(BACKUP_DIR, rel)
    os.makedirs(os.path.dirname(bak), exist_ok=True)
    if os.path.exists(full):
        shutil.copy2(full, bak)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def _apply(rel: str, old: str, new: str, description: str) -> bool:
    src = _read(rel)
    if old not in src:
        print(f"  ⚠  {description} — pattern not found in {rel} (already patched?)")
        return False
    _write(rel, src.replace(old, new, 1))
    CHANGES.append({"file": rel, "description": description})
    print(f"  ✅ {description}")
    return True


def _apply_all(rel: str, replacements: list[tuple[str, str, str]]) -> None:
    """Apply multiple replacements to the same file in one pass."""
    src = _read(rel)
    for old, new, desc in replacements:
        if old not in src:
            print(f"  ⚠  {desc} — pattern not found (already patched?)")
            continue
        src = src.replace(old, new, 1)
        CHANGES.append({"file": rel, "description": desc})
        print(f"  ✅ {desc}")
    _write(rel, src)


# ─── FIX-1 & FIX-2 & FIX-3: trainer.py ───────────────────────────────────────

def fix_trainer() -> None:
    rel = "core/models/training/trainer.py"
    if not os.path.exists(_path(rel)):
        print(f"  ⚠  {rel} not found — skipping trainer fixes")
        return

    replacements = [
        # FIX-1a: XGBoost iterations
        (
            "XGB_N_ESTIMATORS: int = 50",
            "XGB_N_ESTIMATORS: int = 500  # FIX-1: was 50 — too few for financial TS signal",
            "FIX-1a: XGB_N_ESTIMATORS 50→500",
        ),
        # FIX-1b: LightGBM iterations
        (
            "LGB_N_ESTIMATORS: int = 50",
            "LGB_N_ESTIMATORS: int = 500  # FIX-1: was 50",
            "FIX-1b: LGB_N_ESTIMATORS 50→500",
        ),
        # FIX-1c: CatBoost iterations
        (
            "CAT_ITERATIONS: int = 50",
            "CAT_ITERATIONS: int = 500  # FIX-1: was 50",
            "FIX-1c: CAT_ITERATIONS 50→500",
        ),
        # FIX-1d: early stopping — must grow with tree count
        (
            "CAT_EARLY_STOP: int = 10",
            "CAT_EARLY_STOP: int = 50  # FIX-1: was 10 — too aggressive with 500 trees",
            "FIX-1d: CAT_EARLY_STOP 10→50",
        ),
        (
            "XGB_EARLY_STOP: int = 10",
            "XGB_EARLY_STOP: int = 50  # FIX-1: was 10",
            "FIX-1e: XGB_EARLY_STOP 10→50",
        ),
        (
            "LGB_EARLY_STOP: int = 10",
            "LGB_EARLY_STOP: int = 50  # FIX-1: was 10",
            "FIX-1f: LGB_EARLY_STOP 10→50",
        ),
        # FIX-3a: production gate — P(Sharpe>1.5) threshold
        (
            "PROD_GATE_P_SHARPE: float = 0.40     # P(Sharpe > SHARPE_TARGET) must exceed this",
            "PROD_GATE_P_SHARPE: float = 0.70     # FIX-3: institutional minimum (was 0.40 — too lenient)",
            "FIX-3a: PROD_GATE_P_SHARPE 0.40→0.70",
        ),
        # FIX-3b: worst-5% Sharpe floor
        (
            "PROD_GATE_WORST5PCT: float = -1.5     # worst-5% fold Sharpe must be ≥ -1.5 (Realism Gate)",
            "PROD_GATE_WORST5PCT: float = 0.0      # FIX-3: institutional minimum (was -1.5)",
            "FIX-3b: PROD_GATE_WORST5PCT -1.5→0.0",
        ),
        # FIX-2a: TCN epochs in fold training loop
        (
            "            tcn_m.train(\n"
            "                X_tr_tcn, y_tr, y_vol_tr,\n"
            "                X_es_tcn, y_es, y_vol_es,\n"
            "                epochs=5, batch_size=32,\n"
            "                callbacks=[early_stop]\n"
            "            )",
            "            tcn_m.train(\n"
            "                X_tr_tcn, y_tr, y_vol_tr,\n"
            "                X_es_tcn, y_es, y_vol_es,\n"
            "                epochs=50,  # FIX-2: was 5 — TCN needs 50+ epochs to converge on financial TS\n"
            "                batch_size=32,\n"
            "                callbacks=[early_stop]\n"
            "            )",
            "FIX-2a: TCN fold training epochs 5→50",
        ),
        # FIX-2b: TCN final retrain epochs
        (
            "        tcn_final.train(\n"
            "            X_ft_tcn, y_ft, y_vol_ft,\n"
            "            X_fe_tcn, y_fe, y_vol_fe,\n"
            "            epochs=5, batch_size=32,\n"
            "            callbacks=[early_stop_final]\n"
            "        )",
            "        tcn_final.train(\n"
            "            X_ft_tcn, y_ft, y_vol_ft,\n"
            "            X_fe_tcn, y_fe, y_vol_fe,\n"
            "            epochs=50,  # FIX-2: was 5\n"
            "            batch_size=32,\n"
            "            callbacks=[early_stop_final]\n"
            "        )",
            "FIX-2b: TCN final retrain epochs 5→50",
        ),
        # FIX-8: explicit synthetic FII guard in _build_context
        (
            "            fii_series = pipeline.fii_provider.get_fii_flow(start_date, end_date)\n"
            "            context['fii_net'] = fii_series",
            "            fii_series = pipeline.fii_provider.get_fii_flow(start_date, end_date)\n"
            "            # FIX-8: guard against trivially-short FII series that trigger synthetic fallback\n"
            "            if fii_series is None or len(fii_series) < 30:\n"
            "                logger.warning(\n"
            "                    f\"[{ticker}] Insufficient FII data ({len(fii_series) if fii_series is not None else 0} days). \"\n"
            "                    \"Using ZERO FII flows (not synthetic random). \"\n"
            "                    \"Re-connect Kite and refresh FII cache before production training.\"\n"
            "                )\n"
            "                import pandas as _pd\n"
            "                fii_series = _pd.Series(0.0, index=_pd.date_range(start_date, end_date, freq='B'), name='fii_net')\n"
            "            context['fii_net'] = fii_series",
            "FIX-8: zero-fill FII instead of random synthetic when data < 30 days",
        ),
    ]

    print(f"\n[trainer.py] Applying {len(replacements)} fixes …")
    _apply_all(rel, replacements)


# ─── FIX-4: backtester.py intrabar ambiguity ──────────────────────────────────

def fix_backtester() -> None:
    rel = "core/models/tcn/backtester.py"
    if not os.path.exists(_path(rel)):
        print(f"  ⚠  {rel} not found — skipping backtester fix")
        return

    # The bug: for a LONG, when BOTH curr_low<=stop AND curr_high>=target,
    # the `elif` means SL always wins.  Fix: check proximity to open first.
    LONG_OLD = (
                "                    # Scenario B: Intraday Stop Hit\n"
                "                    elif curr_low <= stop_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = stop_price * (1 - self.slippage) \n"
                "                        reason = \"SL_HIT\"\n"
                "                    # Scenario B2: Intraday Target Hit\n"
                "                    elif curr_high >= target_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = target_price * (1 - self.slippage)\n"
                "                        reason = \"PT_HIT\""
    )
    LONG_NEW = (
                "                    # Scenario B: Intraday SL/PT — FIX-4: resolve ambiguity\n"
                "                    # when BOTH levels are breached on the same daily bar,\n"
                "                    # use open-proximity to determine which was hit first.\n"
                "                    elif curr_low <= stop_price and curr_high >= target_price:\n"
                "                        dist_sl = abs(curr_open - stop_price)\n"
                "                        dist_pt = abs(curr_open - target_price)\n"
                "                        if dist_sl <= dist_pt:\n"
                "                            sl_hit = True; exit_price = stop_price * (1 - self.slippage); reason = \"SL_HIT\"\n"
                "                        else:\n"
                "                            sl_hit = True; exit_price = target_price * (1 - self.slippage); reason = \"PT_HIT\"\n"
                "                    elif curr_low <= stop_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = stop_price * (1 - self.slippage)\n"
                "                        reason = \"SL_HIT\"\n"
                "                    elif curr_high >= target_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = target_price * (1 - self.slippage)\n"
                "                        reason = \"PT_HIT\""
    )

    SHORT_OLD = (
                "                    # Scenario B: Intraday Stop Hit\n"
                "                    elif curr_high >= stop_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = stop_price * (1 + self.slippage)\n"
                "                        reason = \"SL_HIT\"\n"
                "                    # Scenario B2: Intraday Target Hit\n"
                "                    elif curr_low <= target_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = target_price * (1 + self.slippage)\n"
                "                        reason = \"PT_HIT\""
    )
    SHORT_NEW = (
                "                    # Scenario B: Intraday SL/PT — FIX-4 (short side)\n"
                "                    elif curr_high >= stop_price and curr_low <= target_price:\n"
                "                        dist_sl = abs(curr_open - stop_price)\n"
                "                        dist_pt = abs(curr_open - target_price)\n"
                "                        if dist_sl <= dist_pt:\n"
                "                            sl_hit = True; exit_price = stop_price * (1 + self.slippage); reason = \"SL_HIT\"\n"
                "                        else:\n"
                "                            sl_hit = True; exit_price = target_price * (1 + self.slippage); reason = \"PT_HIT\"\n"
                "                    elif curr_high >= stop_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = stop_price * (1 + self.slippage)\n"
                "                        reason = \"SL_HIT\"\n"
                "                    elif curr_low <= target_price:\n"
                "                        sl_hit = True\n"
                "                        exit_price = target_price * (1 + self.slippage)\n"
                "                        reason = \"PT_HIT\""
    )

    print(f"\n[backtester.py] Applying intrabar ambiguity fix …")
    src = _read(rel)
    changed = False
    if LONG_OLD in src:
        src = src.replace(LONG_OLD, LONG_NEW, 1)
        CHANGES.append({"file": rel, "description": "FIX-4: LONG intrabar SL/PT open-proximity ordering"})
        print("  ✅ FIX-4: LONG intrabar SL/PT ambiguity resolved")
        changed = True
    else:
        print("  ⚠  FIX-4 LONG pattern not found (already patched?)")

    if SHORT_OLD in src:
        src = src.replace(SHORT_OLD, SHORT_NEW, 1)
        CHANGES.append({"file": rel, "description": "FIX-4: SHORT intrabar SL/PT open-proximity ordering"})
        print("  ✅ FIX-4: SHORT intrabar SL/PT ambiguity resolved")
        changed = True
    else:
        print("  ⚠  FIX-4 SHORT pattern not found (already patched?)")

    if changed:
        _write(rel, src)


# ─── FIX-5: features.py fallback values ───────────────────────────────────────

def fix_features() -> None:
    rel = "core/models/features.py"
    if not os.path.exists(_path(rel)):
        print(f"  ⚠  {rel} not found — skipping features fix")
        return

    print(f"\n[features.py] Applying degraded-feature fallback fix …")

    _apply_all(rel, [
        # relative_strength_nifty fallback
        (
            "    else:\n"
            "        df['relative_strength_nifty'] = stock_ret_20",
            "    else:\n"
            "        # FIX-5: was stock_ret_20 — raw momentum has NO relative-strength information\n"
            "        # when NIFTY close is unavailable. Zero (neutral) is correct; the model\n"
            "        # will learn to ignore this feature rather than learn spurious stock momentum.\n"
            "        df['relative_strength_nifty'] = 0.0\n"
            "        import logging as _log\n"
            "        _log.getLogger('MARK5.Features').warning(\n"
            "            'relative_strength_nifty: NIFTY close unavailable — using 0.0 (neutral). '\n"
            "            'Connect Kite and run MarketDataProvider.get_nifty50_data() before training.'\n"
            "        )",
            "FIX-5a: relative_strength_nifty fallback 0.0 instead of raw stock returns",
        ),
        # sector_rel_strength fallback
        (
            "    else:\n"
            "        df['sector_rel_strength'] = stock_ret_10",
            "    else:\n"
            "        # FIX-5: was stock_ret_10 — same argument as relative_strength_nifty\n"
            "        df['sector_rel_strength'] = 0.0\n"
            "        import logging as _log\n"
            "        _log.getLogger('MARK5.Features').warning(\n"
            "            'sector_rel_strength: sector ETF close unavailable — using 0.0 (neutral). '\n"
            "            'Connect Kite and run MarketDataProvider.get_sector_etf_data() before training.'\n"
            "        )",
            "FIX-5b: sector_rel_strength fallback 0.0 instead of raw stock returns",
        ),
    ])


# ─── FIX-6: backtest_150.py — refuse shadow models ────────────────────────────

def fix_backtest_150() -> None:
    rel = "backtest_150.py"
    if not os.path.exists(_path(rel)):
        print(f"  ⚠  {rel} not found — skipping backtest_150 fix")
        return

    print(f"\n[backtest_150.py] Applying gate-enforcement fix …")

    # Replace allow_shadow=True with a hard gate check
    OLD = (
        "        try:\n"
        "            predictor = MARK5Predictor(ticker, allow_shadow=True)\n"
        "        except Exception as e:\n"
        "            return {\"ticker\": ticker, \"error\": f\"Could not load predictor after training: {e}\"}"
    )
    NEW = (
        "        # FIX-6: never run backtests on models that failed the production gate.\n"
        "        # allow_shadow=True was silently inflating results with certified-untrustworthy models.\n"
        "        import json as _json, pathlib as _pl\n"
        "        _base = _pl.Path('models') / ticker\n"
        "        _versions = sorted([v for v in _base.iterdir() if v.is_dir() and v.name.startswith('v')],\n"
        "                           key=lambda p: int(p.name[1:]), reverse=True) if _base.exists() else []\n"
        "        if _versions:\n"
        "            _meta_file = _versions[0] / 'metadata.json'\n"
        "            if _meta_file.exists():\n"
        "                _meta = _json.loads(_meta_file.read_text())\n"
        "                if not _meta.get('passes_gate', False):\n"
        "                    logger.warning(f'{ticker} GATE FAIL — skipping (model did not pass production gate)')\n"
        "                    return {'ticker': ticker, 'error': 'Gate failure — model not production-ready'}\n"
        "        try:\n"
        "            predictor = MARK5Predictor(ticker, allow_shadow=False)  # FIX-6: was allow_shadow=True\n"
        "        except Exception as e:\n"
        "            return {\"ticker\": ticker, \"error\": f\"Could not load predictor after training: {e}\"}"
    )

    src = _read(rel)
    if OLD in src:
        _write(rel, src.replace(OLD, NEW, 1))
        CHANGES.append({"file": rel, "description": "FIX-6: gate check before backtest, refuse shadow models"})
        print("  ✅ FIX-6: shadow models now blocked from backtest_150")
    else:
        print("  ⚠  FIX-6 pattern not found (already patched?)")


# ─── FIX-7: compare_intraday_swing.py — add honest warning ────────────────────

def fix_compare_intraday() -> None:
    rel = "compare_intraday_swing.py"
    if not os.path.exists(_path(rel)):
        print(f"  ⚠  {rel} not found — skipping compare_intraday fix")
        return

    print(f"\n[compare_intraday_swing.py] Adding daily-bar limitation warning …")

    OLD = "def run_comparison():\n    results = []"
    NEW = (
        "def run_comparison():\n"
        "    # FIX-7: IMPORTANT — this comparison is NOT a true intraday vs swing comparison.\n"
        "    # Both modes use DAILY OHLCV bars.  The EQUITY_INTRADAY segment flag only\n"
        "    # changes the STT rate; execution still enters at open and exits at close of\n"
        "    # the SAME DAILY bar.  Real intraday P&L requires 5-minute or 15-minute bars.\n"
        "    # Treat the 'Intraday' column as 'same-bar exit (different tax)' not true intraday.\n"
        "    import logging as _log\n"
        "    _log.getLogger('MARK5.Comparison').warning(\n"
        "        'compare_intraday_swing.py uses DAILY bars for both modes. '\n"
        "        'The intraday/swing distinction here is TAX REGIME only, not execution timing. '\n"
        "        'True intraday comparison requires 5m/15m Kite data.'\n"
        "    )\n"
        "    results = []"
    )

    src = _read(rel)
    if OLD in src:
        _write(rel, src.replace(OLD, NEW, 1))
        CHANGES.append({"file": rel, "description": "FIX-7: daily-bar limitation warning in intraday comparison"})
        print("  ✅ FIX-7: honest warning added to compare_intraday_swing")
    else:
        print("  ⚠  FIX-7 pattern not found (already patched?)")


# ─── summary ──────────────────────────────────────────────────────────────────

def print_summary() -> None:
    print("\n" + "=" * 70)
    print("MARK5 FIX APPLICATOR — SUMMARY")
    print("=" * 70)
    if not CHANGES:
        print("No changes applied (all patterns already patched or files missing).")
        return
    for i, c in enumerate(CHANGES, 1):
        print(f"  {i:2d}. [{c['file']}] {c['description']}")
    print(f"\nTotal: {len(CHANGES)} patches applied.")
    print(f"Backups saved to: {BACKUP_DIR}")
    print("\nNext steps:")
    print("  1. Re-train all models: python core/models/training/trainer.py --symbols ALL --years 3")
    print("  2. Run baseline:        python baseline_bb_breakout.py")
    print("  3. Run OOS backtest:    python backtest_150.py")
    print("  4. Compare results vs baseline to isolate ML alpha")


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(BACKUP_DIR, exist_ok=True)
    print("MARK5 Fix Applicator — running all patches …")
    print(f"Project root: {ROOT}")
    print(f"Backup directory: {BACKUP_DIR}")

    fix_trainer()
    fix_backtester()
    fix_features()
    fix_backtest_150()
    fix_compare_intraday()

    print_summary()
    print("\nRun `python baseline_bb_breakout.py` to generate the BB baseline report.")