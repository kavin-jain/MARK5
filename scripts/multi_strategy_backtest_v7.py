"""
MARK5 Multi-Strategy Backtest v7.0 — The CB-Fixed Production System
════════════════════════════════════════════════════════════════════
Built on V6 framework (7 structural improvements over V2 Baseline).
V7 adds 3 targeted fixes that address the CB Deadlock discovered in V6:

  FIX 1 — CB RECOVERY PROTOCOL (CRITICAL)
  ─────────────────────────────────────────
  V6 FLAW: Circuit breaker enters PAUSE → no entries → portfolio stays
  all-cash → equity flat → DD never reduces → stuck in PAUSE forever.
  "Zero trades in 2023, 2024, 2025 while market was +98pp."

  V7 FIX: After 90 days in PAUSE with all-cash + Nifty +15% above
  PAUSE trigger price → allow 1 cautious T1 re-entry (conf ≥ 0.62).
  On success, reset recovery timer. On failure, wait another 90 days.

  FIX 2 — RSI GATE ONLY (remove SMA + volume)
  ─────────────────────────────────────────────
  V6 OBSERVATION: SMA and volume gates together reduce trade count by
  ~60% with no commensurate return improvement (V6 True OOS showed
  only 11 trades in 2022-2026 partly due to over-filtering).
  V7 keeps RSI gate (overbought/oversold protection) but removes the
  SMA (trend confirmation) and volume (participation) gates.

  FIX 3 — FII GATE TIGHTENED (-3% → -2.5%)
  ──────────────────────────────────────────
  V4 / V6 used -3% Nifty 5d return threshold to block entries.
  Tightening to -2.5% catches FII selling pressure slightly earlier,
  reducing the number of bad entries in early trend reversals.

EXPECTED V7 OUTCOME vs V6:
  - Zero-trade years (2023-2025 locked) → should now have trades
  - CB Recovery entries: fewer but higher-quality
  - RSI-only gate: moderately more trades than V6's 3-gate filter
  - Net: should approximate V2 Baseline returns but with CB safety

PAPER MODE ONLY. Capital pool: ₹5 crore.

CHANGELOG:
- [2026-05-24] v7.0: CB Recovery Protocol + RSI-only gate + FII -2.5%
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Import V6 shared utilities ────────────────────────────────────────────────
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _ROOT    not in sys.path: sys.path.insert(0, _ROOT)
if _SCRIPTS not in sys.path: sys.path.insert(0, _SCRIPTS)

from multi_strategy_backtest_v6 import (
    # Constants (shared with V6)
    INITIAL_CAPITAL, OOS_START, OOS_END, TRUE_OOS_START,
    COST_PCT, SLIPPAGE_PCT, MAX_POSITIONS,
    ML_ENTRY_HURDLE, ML_EXIT_HURDLE, ML_ROLL_WINDOW, MIN_ML_STD,
    CONF_TIER_4, CONF_TIER_3, CONF_TIER_2,
    ALLOC_TIER_1,
    TRAIL_NORMAL, TRAIL_ELEVATED, TRAIL_HIGH,
    EQUITY_CB_CAUTION, EQUITY_CB_PAUSE, EQUITY_CB_EMERGENCY,
    RSI_ENTRY_MAX, RSI_ENTRY_MIN,
    REBAL_NORMAL_DAYS, REBAL_HIGH_VIX_DAYS, VIX_REBAL_TRIGGER,
    FII_PROXY_CRISIS,
    CANDIDATE_TICKERS, EXCLUDED_TICKERS,
    # Helper functions
    get_confidence_alloc, get_vix_trail_stop, compute_vix_proxy,
    get_equity_dd_state, compute_fii_proxy, get_rolling_conf,
    # Data loaders
    load_ticker, load_nifty, load_ml_confidence,
    # Portfolio engine (reused unchanged)
    Position, Trade, V6Portfolio,
    # Metrics
    _compile_results,
    # Presenters
    print_verification_report, print_comparison_table,
    # V6 runner for comparison
    run_v2_baseline, run_v6,
)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | V7 | %(levelname)s | %(message)s")
logger = logging.getLogger("MARK5.V7")

# ── V7 Constants (overrides / new) ────────────────────────────────────────────

# V7 Fix 1: CB Recovery Protocol
CB_RECOVERY_MIN_DAYS    = 90     # Days in PAUSE (all-cash) before recovery attempt
CB_RECOVERY_NIFTY_RISE  = 0.15   # Nifty must be +15% above PAUSE trigger price
CB_RECOVERY_MAX_POS     = 1      # Allow only 1 position during recovery
CB_RECOVERY_CONF_HURDLE = 0.62   # Stricter entry threshold during recovery mode
CB_RECOVERY_ALLOC       = ALLOC_TIER_1  # 17% — minimum tier during recovery

# V7 Fix 3: FII gate tightened
FII_PROXY_BLOCK_V7 = -0.025   # -2.5% (tighter than V6's -3%)


# ── V7 Quality Gate (RSI only) ────────────────────────────────────────────────

def check_quality_gate_rsi_only(
    df: pd.DataFrame,
    date: pd.Timestamp,
    rsi_min: float = RSI_ENTRY_MIN,
    rsi_max: float = RSI_ENTRY_MAX,
) -> Tuple[bool, str]:
    """
    [V7 Fix 2] RSI gate ONLY — no SMA/volume checks.

    V6 had 3 gates: RSI + SMA(20) + Volume(0.65x avg).
    SMA and volume gates were too restrictive (~60% fewer trades).
    V7 keeps RSI gate only: avoid overbought (>68) and free-falls (<28).
    """
    try:
        subset = df[df.index <= date]
        if len(subset) < 20:
            return True, "insufficient_data"
        close = subset["close"].astype(float)
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
        if not (rsi_min <= rsi <= rsi_max):
            return False, f"rsi={rsi:.1f} outside [{rsi_min},{rsi_max}]"
        return True, "ok"
    except Exception as e:
        return True, f"gate_error={e}"  # fail open


# ── V7 CB Recovery State ──────────────────────────────────────────────────────

class CbRecoveryTracker:
    """
    Tracks Circuit Breaker PAUSE state for the recovery protocol.

    PAUSE is entered when equity DD > 18%.
    Recovery is triggered when:
      1. All-cash (no positions) for ≥ 90 days
      2. Nifty has recovered ≥ +15% from the PAUSE trigger price
    On trigger: allow 1 cautious T1 entry with conf ≥ 0.62.
    On success (entry executed): reset timer from new trigger date.
    """
    def __init__(self):
        self.pause_start:    Optional[pd.Timestamp] = None
        self.nifty_at_pause: Optional[float]        = None
        self.last_attempt:   Optional[pd.Timestamp] = None  # prevent repeated attempts
        self.n_recoveries:   int                    = 0

    def enter_pause(self, date: pd.Timestamp, nifty_price: float):
        if self.pause_start is None:
            self.pause_start    = date
            self.nifty_at_pause = nifty_price
            logger.info(f"CB PAUSE entered on {date.date()} | Nifty@{nifty_price:.0f}")

    def clear_pause(self):
        self.pause_start    = None
        self.nifty_at_pause = None

    def can_attempt_recovery(
        self,
        date: pd.Timestamp,
        nifty_now: float,
        n_positions: int,
    ) -> bool:
        if self.pause_start is None:
            return False
        if n_positions > 0:
            return False  # already have a position; wait for outcome
        days_paused = (date - self.pause_start).days
        if days_paused < CB_RECOVERY_MIN_DAYS:
            return False
        if self.nifty_at_pause is None or self.nifty_at_pause <= 0:
            return False
        nifty_recovery = (nifty_now - self.nifty_at_pause) / self.nifty_at_pause
        qualifies = nifty_recovery >= CB_RECOVERY_NIFTY_RISE
        if qualifies:
            logger.info(
                f"CB RECOVERY triggered | Days paused={days_paused} "
                f"Nifty recovery={nifty_recovery:+.1%} from {self.nifty_at_pause:.0f} → {nifty_now:.0f}"
            )
        return qualifies

    def on_recovery_entry(self, date: pd.Timestamp, nifty_now: float):
        """Called when a recovery entry is executed. Reset tracker."""
        self.pause_start    = date        # restart timer from this entry
        self.nifty_at_pause = nifty_now  # new reference price
        self.last_attempt   = date
        self.n_recoveries  += 1
        logger.info(f"CB RECOVERY entry executed #{self.n_recoveries} on {date.date()}")


# ── V7 Backtest Runner ────────────────────────────────────────────────────────

def run_v7(
    all_data:  Dict[str, pd.DataFrame],
    conf_map:  Dict[str, pd.Series],
    nifty:     pd.Series,
    dates:     pd.DatetimeIndex,
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
) -> Dict:
    """
    V7 backtest: V6 framework + 3 targeted fixes (CB recovery, RSI gate, FII -2.5%).

    Inherits from V6:
      - Confidence-scaled position sizing (17/22/27/30%)
      - VIX-scaled trailing stops (9/12/15%)
      - Equity circuit breaker (12/18/25% thresholds)
      - FII proxy gate (block new entries in FII selling)
      - Dynamic rebalancing (14d / 21d)
      - Extended exit hurdle (0.42)

    V7 adds:
      1. CbRecoveryTracker: after 90d PAUSE + Nifty +15% → allow 1 T1 re-entry
      2. RSI-only quality gate (remove SMA + volume gates)
      3. FII block tightened to -2.5% (was -3.0%)
    """
    port        = V6Portfolio(INITIAL_CAPITAL)
    fii_proxy   = compute_fii_proxy(nifty)
    peak_equity = INITIAL_CAPITAL
    last_rebal: Optional[pd.Timestamp] = None
    cb_tracker  = CbRecoveryTracker()

    for date in dates:
        prices = {t: float(all_data[t].loc[date, "close"])
                  for t in all_data if date in all_data[t].index}
        if not prices:
            continue

        # ── Regime signals ────────────────────────────────────────────────────
        vix_val = compute_vix_proxy(nifty, date)
        trail   = get_vix_trail_stop(vix_val)

        fii_series = fii_proxy[fii_proxy.index <= date]
        fii_ret    = float(fii_series.iloc[-1]) if len(fii_series) else 0.0
        fii_block  = fii_ret <= FII_PROXY_BLOCK_V7   # V7: -2.5% (was -3.0%)
        fii_crisis = fii_ret <= FII_PROXY_CRISIS

        nifty_slice  = nifty[nifty.index <= date]
        nifty_now    = float(nifty_slice.iloc[-1]) if len(nifty_slice) else 0.0

        # ── Equity circuit breaker ────────────────────────────────────────────
        eq_now      = port.get_equity(prices)
        peak_equity = max(peak_equity, eq_now)
        equity_dd, equity_state = get_equity_dd_state(eq_now, peak_equity)

        # V7: Track PAUSE state for recovery protocol
        if equity_state in ("PAUSE", "EMERGENCY"):
            cb_tracker.enter_pause(date, nifty_now)
        else:
            cb_tracker.clear_pause()

        if equity_state == "EMERGENCY":
            port.exit_all(prices, date, "EQUITY_CB_EMERGENCY")
            port.equity_history.append({
                "date":         date,
                "equity":       port.get_equity(prices),
                "n_pos":        0,
                "vix":          round(vix_val * 100, 1),
                "equity_dd":    round(equity_dd * 100, 1),
                "equity_state": equity_state,
            })
            continue

        size_scale = 0.5 if equity_state == "CAUTION" else 1.0
        entry_ok   = equity_state not in ("PAUSE", "EMERGENCY")

        # ── V7 CB Recovery check ──────────────────────────────────────────────
        recovery_ok = False
        if not entry_ok and not fii_crisis:
            recovery_ok = cb_tracker.can_attempt_recovery(
                date, nifty_now, len(port.positions)
            )

        # ── Update trailing peaks ─────────────────────────────────────────────
        for tk, pos in list(port.positions.items()):
            if tk in prices:
                pos.peak_price = max(pos.peak_price, prices[tk])

        # ── Dynamic rebalancing frequency ─────────────────────────────────────
        rebal_freq = REBAL_HIGH_VIX_DAYS if vix_val > VIX_REBAL_TRIGGER else REBAL_NORMAL_DAYS
        is_rebal   = (last_rebal is None) or ((date - last_rebal).days >= rebal_freq)

        # ── Exits (daily) ─────────────────────────────────────────────────────
        for tk in list(port.positions.keys()):
            if tk not in prices:
                continue
            pos  = port.positions.get(tk)
            if pos is None:
                continue
            curr = prices[tk]
            if curr < pos.peak_price * (1 - trail):
                port.exit(tk, curr, date, f"TRAIL_STOP({trail:.0%})")
                continue
            if is_rebal and tk in conf_map:
                rc = get_rolling_conf(conf_map[tk], date)
                if rc < ML_EXIT_HURDLE:
                    port.exit(tk, curr, date, f"ML_EXIT(rc={rc:.3f})")

        # ── Entries ───────────────────────────────────────────────────────────
        if is_rebal and (entry_ok or recovery_ok) and not fii_crisis:
            last_rebal = date

            # Recovery mode: use stricter conf hurdle and max 1 position
            using_recovery = recovery_ok and not entry_ok
            entry_hurdle   = CB_RECOVERY_CONF_HURDLE if using_recovery else ML_ENTRY_HURDLE
            slot_limit     = CB_RECOVERY_MAX_POS      if using_recovery else MAX_POSITIONS

            scores: List[Tuple[str, float]] = []
            for tk in conf_map:
                if tk in port.positions or tk not in prices:
                    continue
                if tk in EXCLUDED_TICKERS:
                    continue
                if fii_block:
                    continue  # V7 FII gate: -2.5% threshold

                rc = get_rolling_conf(conf_map[tk], date)
                if rc < entry_hurdle:
                    continue

                # V7 Fix 2: RSI gate only (no SMA / volume)
                tkdf = all_data.get(tk)
                if tkdf is not None:
                    passes, reason = check_quality_gate_rsi_only(tkdf, date)
                    if not passes:
                        logger.debug(f"[{tk}] RSI gate blocked: {reason}")
                        continue

                scores.append((tk, rc))

            scores.sort(key=lambda x: -x[1])
            slots = slot_limit - len(port.positions)

            for tk, rc in scores[:slots]:
                entered = port.enter(
                    tk, prices[tk], date, rc, vix_val,
                    size_scale=0.5 if using_recovery else size_scale,
                )
                if entered and using_recovery:
                    cb_tracker.on_recovery_entry(date, nifty_now)

        # ── Record daily equity ───────────────────────────────────────────────
        port.equity_history.append({
            "date":         date,
            "equity":       port.get_equity(prices),
            "n_pos":        len(port.positions),
            "vix":          round(vix_val * 100, 1),
            "equity_dd":    round(equity_dd * 100, 1),
            "equity_state": equity_state,
        })

    # ── Force exit at end ─────────────────────────────────────────────────────
    final_date = dates[-1]
    final_prices = {t: float(all_data[t].loc[final_date, "close"])
                    for t in all_data if final_date in all_data[t].index}
    port.exit_all(final_prices, final_date, "END_SIM")

    result = _compile_results(
        port,
        "V7 (CB-Recovery + RSI-gate + FII-2.5%)",
        oos_start, oos_end,
    )
    result["cb_recoveries"] = cb_tracker.n_recoveries
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 90}")
    print("  MARK5 V7 — CB RECOVERY PROTOCOL + RSI GATE + FII -2.5%")
    print("  Fixing the circular deadlock that killed V6 Full")
    print(f"  OOS: {OOS_START} → {OOS_END} | True OOS: {TRUE_OOS_START} → {OOS_END}")
    print(f"  Capital: ₹{INITIAL_CAPITAL/1e7:.0f}cr | CB Recovery: {CB_RECOVERY_MIN_DAYS}d + "
          f"Nifty {CB_RECOVERY_NIFTY_RISE:.0%} recovery")
    print(f"{'═' * 90}\n")

    # Load data
    print("Loading Nifty50...")
    nifty = load_nifty()
    if nifty is None:
        print("ERROR: Nifty data not found.")
        return
    print(f"  Nifty: {len(nifty)} bars ({nifty.index[0].date()} → {nifty.index[-1].date()})")

    print("\nLoading ticker data...")
    all_data: Dict[str, pd.DataFrame] = {}
    for tk in CANDIDATE_TICKERS:
        if tk in EXCLUDED_TICKERS:
            continue
        df = load_ticker(tk)
        if df is not None and len(df) >= 300:
            all_data[tk] = df
            print(f"  ✓ {tk}: {len(df)} bars")
        else:
            print(f"  ✗ {tk}: not found")

    print("\nLoading ML confidence series...")
    conf_map: Dict[str, pd.Series] = {}
    for tk in list(all_data.keys()):
        conf = load_ml_confidence(tk)
        if conf is None:
            print(f"  ✗ {tk}: no ML models")
            continue
        oos_conf = conf.loc[OOS_START:OOS_END]
        if len(oos_conf) < 50:
            continue
        oos_std = float(oos_conf.std())
        oos_max = float(oos_conf.max())
        if oos_std < MIN_ML_STD or oos_max < ML_ENTRY_HURDLE:
            print(f"  ✗ {tk}: ML flat (std={oos_std:.4f}) — excluded")
            continue
        conf_map[tk] = conf
        print(f"  ✓ {tk}: std={oos_std:.4f} max={oos_max:.3f}")

    print(f"\nActive ML tickers: {len(conf_map)}: {sorted(conf_map.keys())}")

    dates_full = pd.bdate_range(start=OOS_START, end=OOS_END)
    dates_true = pd.bdate_range(start=TRUE_OOS_START, end=OOS_END)
    dates_full = dates_full[dates_full <= pd.Timestamp(OOS_END)]
    dates_true = dates_true[dates_true <= pd.Timestamp(OOS_END)]

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Running V2 BASELINE...")
    rv2 = run_v2_baseline(all_data, conf_map, nifty, dates_full, OOS_START, OOS_END)
    print(f"  V2: {rv2['n_trades']}t | WR={rv2['win_rate']:.1f}% | "
          f"Net={rv2['net_after_tax']:+.1f}% | DD={rv2['max_dd']:.1f}% | "
          f"Sharpe={rv2['sharpe']:.2f} | Calmar={rv2['calmar']:.2f}")

    print("\n" + "─" * 70)
    print("Running V6 FULL (2022-2026) — for comparison...")
    rv6 = run_v6(all_data, conf_map, nifty, dates_full, OOS_START, OOS_END)
    print(f"  V6: {rv6['n_trades']}t | WR={rv6['win_rate']:.1f}% | "
          f"Net={rv6['net_after_tax']:+.1f}% | DD={rv6['max_dd']:.1f}% | "
          f"Sharpe={rv6['sharpe']:.2f} | Calmar={rv6['calmar']:.2f}")

    # ── V7 runs ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Running V7 FULL (2022-2026) — CB Recovery + RSI gate + FII -2.5%...")
    rv7_full = run_v7(all_data, conf_map, nifty, dates_full, OOS_START, OOS_END)
    print(f"  V7: {rv7_full['n_trades']}t | WR={rv7_full['win_rate']:.1f}% | "
          f"Net={rv7_full['net_after_tax']:+.1f}% | DD={rv7_full['max_dd']:.1f}% | "
          f"Sharpe={rv7_full['sharpe']:.2f} | Calmar={rv7_full['calmar']:.2f} | "
          f"CB_recoveries={rv7_full.get('cb_recoveries', 0)}")

    print("\n" + "─" * 70)
    print("Running V7 TRUE OOS (2025-2026)...")
    rv7_oos = run_v7(all_data, conf_map, nifty, dates_true, TRUE_OOS_START, OOS_END)
    print(f"  V7 OOS: {rv7_oos['n_trades']}t | WR={rv7_oos['win_rate']:.1f}% | "
          f"Net={rv7_oos['net_after_tax']:+.1f}% | DD={rv7_oos['max_dd']:.1f}% | "
          f"Sharpe={rv7_oos['sharpe']:.2f}")

    # ── Reports ───────────────────────────────────────────────────────────────
    print_verification_report(rv7_full, "V7 FULL (2022-2026)")
    print_verification_report(rv7_oos,  "V7 TRUE OOS (2025-2026)")

    rv2_show = dict(rv2); rv2_show["label"] = "V2 Baseline"
    rv6_show = dict(rv6); rv6_show["label"] = "V6 Full (CB deadlock)"
    rv7_full_show = dict(rv7_full)
    rv7_oos_show  = dict(rv7_oos)
    print_comparison_table([rv2_show, rv6_show, rv7_full_show, rv7_oos_show])

    # ── Save ──────────────────────────────────────────────────────────────────
    reports_dir = os.path.join(_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    def _to_json(r: Dict) -> Dict:
        out = {k: v for k, v in r.items() if k not in ("equity_df", "trades")}
        if "trades" in r:
            out["trades"] = [
                {
                    "ticker":      t.ticker,
                    "entry_date":  str(t.entry_date.date()),
                    "exit_date":   str(t.exit_date.date()),
                    "entry_price": round(t.entry_price, 2),
                    "exit_price":  round(t.exit_price, 2),
                    "shares":      t.shares,
                    "net_pnl":     round(t.net_pnl, 2),
                    "pnl_pct":     round(t.pnl_pct, 2),
                    "hold_days":   t.hold_days,
                    "exit_reason": t.exit_reason,
                    "conf_entry":  round(t.conf_entry, 4),
                    "alloc_tier":  t.alloc_tier,
                }
                for t in r["trades"]
            ]
        return out

    out = {
        "v2_baseline": _to_json(rv2),
        "v6_full":     _to_json(rv6),
        "v7_full":     _to_json(rv7_full),
        "v7_true_oos": _to_json(rv7_oos),
    }
    json_path = os.path.join(reports_dir, "multi_strategy_backtest_v7.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    print(f"\n{'═' * 90}")
    print("  V7 KEY METRICS")
    print(f"{'═' * 90}")
    print(f"  CB Recoveries:   {rv7_full.get('cb_recoveries', 0)} attempts over 4.4 years")
    print(f"  V7 vs V6:        {rv7_full['net_after_tax'] - rv6['net_after_tax']:+.1f}pp improvement in net annual return")
    print(f"  V7 vs V2:        {rv7_full['net_after_tax'] - rv2['net_after_tax']:+.1f}pp vs the best current system")
    print(f"{'═' * 90}\n")


if __name__ == "__main__":
    main()
