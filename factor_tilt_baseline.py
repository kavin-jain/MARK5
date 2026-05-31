"""
MARK5 Factor-Tilt Buy-and-Hold Baseline — HONEST ALPHA VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS EXISTS
  The live MARK5 system underperforms a plain NIFTY buy-and-hold by ~10.8% CAGR
  over 2007→2026 (see data/backtest/deep_report.txt). The stock *selection*
  (momentum_score) is sound; the value destroyer is the OVERLAY — ML-gated entry
  timing plus ATR trailing-stops / time-stops / cooldowns (simulate_position).

  This script keeps the proven selection, STRIPS the overlay, and simply HOLDS
  the top-N momentum names equal-weight, rebalanced quarterly. It then validates
  the result the honest way: walk-forward by construction, net of costs, against
  two benchmarks (NIFTY B&H and an equal-weight buy-hold of the whole universe).

WHAT IT REPORTS
  Variant A  : always-invested top-N (max exposure, true "beat B&H" test)
  Variant B  : same selection, but go to cash when NIFTY < its 200-DMA
  Overlay A/B: the SAME Variant-A picks run through the existing stop/exit overlay
               → the CAGR delta = exactly what the stops/exits have been costing.

TRADING ROLE : Offline backtest / research only — never in the live signal path.
SAFETY LEVEL : MEDIUM (offline, additive; touches no live trading code)
"""
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.data.nse_data_provider import fetch_equity_ohlcv, fetch_nifty50_index
from core.models.portfolio_backtest import (
    momentum_score,
    simulate_position,
    SLIPPAGE_PCT,
    BROKERAGE_FLAT,
    STT_PCT,
    MIN_BARS,
)
from core.utils.constants import DEFAULT_WATCHLIST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [FACTOR-TILT] | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.FactorTiltBaseline")
# Quieten the chatty data/pipeline modules — one log line per fetch is enough noise.
for noisy in ("MARK5.NSEDataProvider", "MARK5.PortfolioRotation", "MARK5.WalkForwardBacktest"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

INITIAL_CAPITAL = 10_00_000.0   # ₹10 lakh notional — NAV is reported as % so the level is cosmetic
PER_SIDE_RATE = SLIPPAGE_PCT + STT_PCT   # cost charged per buy and per sell on traded value
TRADING_DAYS = 252.0


# ── Data loading ────────────────────────────────────────────────────────────────
def load_universe(args) -> List[str]:
    """Resolve the ticker universe from CLI flags, else the curated midcap watchlist."""
    if args.tickers:
        return list(args.tickers)
    if args.universe_file:
        with open(args.universe_file) as fh:
            data = json.load(fh)
        names = data.get("active_universe", data) if isinstance(data, dict) else data
        return list(names)
    return list(DEFAULT_WATCHLIST)


def load_prices(universe: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch (cached) daily OHLCV per ticker. Drops names with < MIN_BARS history."""
    out: Dict[str, pd.DataFrame] = {}
    for t in universe:
        df = fetch_equity_ohlcv(t, start, end)
        if df is not None and len(df) >= MIN_BARS:
            out[t] = df
        else:
            logger.info(f"[{t}] skipped (insufficient history)")
    logger.info(f"Loaded {len(out)}/{len(universe)} tickers with ≥{MIN_BARS} bars")
    return out


# ── Calendar / rebalance helpers ──────────────────────────────────────────────
def quarter_rebalance_dates(calendar: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """First trading day of each calendar quarter present in `calendar`."""
    s = pd.Series(calendar, index=calendar)
    firsts = s.groupby(calendar.to_period("Q")).first()
    return list(firsts.values.astype("datetime64[ns]"))


def _select_topn(
    prices: Dict[str, pd.DataFrame],
    nifty: Optional[pd.Series],
    asof: pd.Timestamp,
    top_n: int,
) -> List[str]:
    """Rank by momentum_score using ONLY bars on/before `asof` (no lookahead)."""
    scores: Dict[str, float] = {}
    for t, df in prices.items():
        hist = df.loc[df.index <= asof]
        if len(hist) < 25:
            continue
        nifty_hist = (
            nifty.reindex(hist.index, method="ffill").dropna()
            if nifty is not None else None
        )
        scores[t] = momentum_score(hist, nifty_hist)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, sc in ranked if sc > -100][:top_n]


# ── Core engine: buy-and-hold quarterly rotation (NO overlay) ──────────────────
def run_rotation(
    prices: Dict[str, pd.DataFrame],
    nifty: Optional[pd.Series],
    calendar: pd.DatetimeIndex,
    top_n: int,
    regime_filter: bool,
) -> Tuple[pd.Series, List[Dict]]:
    """
    Walk quarter boundaries; each quarter hold the top-N momentum names equal-weight
    (true buy-and-hold with intra-quarter weight drift), net of rebalance costs.

    regime_filter=True → hold cash for any quarter where NIFTY closes below its
    200-day EMA on the rebalance date (Variant B).

    Returns (daily NAV series, per-quarter records).
    """
    rebal = quarter_rebalance_dates(calendar)
    nav_dates: List[pd.Timestamp] = []
    nav_vals: List[float] = []
    records: List[Dict] = []

    nifty_ema200 = (
        nifty.ewm(span=200, adjust=False).mean() if nifty is not None else None
    )

    nav = INITIAL_CAPITAL
    held: set = set()

    for qi, r in enumerate(rebal):
        seg_end = rebal[qi + 1] if qi + 1 < len(rebal) else calendar[-1]
        # Segment = [r, seg_end). Final quarter includes the last calendar day.
        if qi + 1 < len(rebal):
            seg_days = calendar[(calendar >= r) & (calendar < seg_end)]
        else:
            seg_days = calendar[(calendar >= r) & (calendar <= seg_end)]
        if len(seg_days) == 0:
            continue

        # ── Regime gate (Variant B) ───────────────────────────────────────────
        in_cash = False
        if regime_filter and nifty is not None:
            nf = float(nifty.reindex([r], method="ffill").iloc[0])
            ema = float(nifty_ema200.reindex([r], method="ffill").iloc[0])
            in_cash = (not np.isnan(nf)) and (not np.isnan(ema)) and (nf < ema)

        selected = [] if in_cash else _select_topn(prices, nifty, r, top_n)

        # ── Rebalance cost: charge on names sold and names bought ─────────────
        new_set = set(selected)
        traded = (held - new_set) | (new_set - held)
        if traded and nav > 0:
            per_pos_value = nav / max(len(new_set | held), 1)
            cost = sum(per_pos_value * PER_SIDE_RATE + BROKERAGE_FLAT for _ in traded)
            nav = max(nav - cost, 0.0)
        held = new_set

        # ── Build the daily NAV path through the quarter (buy-and-hold drift) ─
        seg_nav_start = nav
        if not selected:
            # Cash: NAV is flat across the segment.
            for d in seg_days:
                nav_dates.append(d)
                nav_vals.append(seg_nav_start)
            seg_ret = 0.0
        else:
            # Entry close per name on the rebalance day (first segment day).
            entry_day = seg_days[0]
            entry_px = {}
            valid = []
            for t in selected:
                px = prices[t]["close"].reindex([entry_day], method="ffill")
                if len(px) and not np.isnan(px.iloc[0]) and px.iloc[0] > 0:
                    entry_px[t] = float(px.iloc[0])
                    valid.append(t)
            if not valid:
                for d in seg_days:
                    nav_dates.append(d)
                    nav_vals.append(seg_nav_start)
                seg_ret = 0.0
            else:
                alloc = seg_nav_start / len(valid)   # equal cash split at entry
                for d in seg_days:
                    total = 0.0
                    for t in valid:
                        px = prices[t]["close"].reindex([d], method="ffill")
                        p = float(px.iloc[0]) if len(px) and not np.isnan(px.iloc[0]) else entry_px[t]
                        total += alloc * (p / entry_px[t])
                    nav_dates.append(d)
                    nav_vals.append(total)
                nav = nav_vals[-1]
                seg_ret = nav / seg_nav_start - 1.0

        records.append({
            "rebalance": pd.Timestamp(r).date(),
            "selected": "CASH" if not selected else ",".join(selected),
            "seg_return_pct": round(seg_ret * 100, 2),
            "nav": round(nav, 2),
        })

    nav_series = pd.Series(nav_vals, index=pd.DatetimeIndex(nav_dates))
    nav_series = nav_series[~nav_series.index.duplicated(keep="last")]
    return nav_series, records


# ── Overlay A/B diagnostic: same picks, run through simulate_position ──────────
def run_overlay_diagnostic(
    prices: Dict[str, pd.DataFrame],
    nifty: Optional[pd.Series],
    calendar: pd.DatetimeIndex,
    top_n: int,
) -> Tuple[float, List[Dict]]:
    """
    Take the SAME always-invested Variant-A selections each quarter, but instead of
    holding, run each name through the existing stop/exit overlay (simulate_position)
    with a buy-on-bar-1 signal. This isolates the cost of the stops/exits/cooldowns
    from the (broken) ML entry gate. Returns (final overlay NAV, per-quarter records).
    """
    rebal = quarter_rebalance_dates(calendar)
    nav = INITIAL_CAPITAL
    records: List[Dict] = []

    for qi, r in enumerate(rebal):
        seg_end = rebal[qi + 1] if qi + 1 < len(rebal) else calendar[-1]
        selected = _select_topn(prices, nifty, r, top_n)
        if not selected or nav <= 0:
            records.append({"rebalance": pd.Timestamp(r).date(), "seg_return_pct": 0.0})
            continue

        per_cap = nav / len(selected)
        seg_total = 0.0
        used = 0
        for t in selected:
            df = prices[t]
            seg_df = df.loc[(df.index >= r) & (df.index <= seg_end)].copy()
            if len(seg_df) < 10:
                seg_total += per_cap        # untradeable → held flat
                used += 1
                continue
            # Buy-on-bar-1 signal: simulate_position enters at the open after a signal.
            signals = pd.Series(0, index=seg_df.index)
            signals.iloc[0] = 1
            final_cap, _ = simulate_position(seg_df, signals, per_cap)
            seg_total += final_cap
            used += 1

        if used == 0:
            records.append({"rebalance": pd.Timestamp(r).date(), "seg_return_pct": 0.0})
            continue

        seg_ret = seg_total / nav - 1.0
        nav = seg_total
        records.append({"rebalance": pd.Timestamp(r).date(),
                        "seg_return_pct": round(seg_ret * 100, 2),
                        "nav": round(nav, 2)})

    return nav, records


# ── Metrics ─────────────────────────────────────────────────────────────────────
def benchmark_nav(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Normalise a close series to a NAV starting at INITIAL_CAPITAL over `calendar`."""
    s = series.reindex(calendar, method="ffill").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s / float(s.iloc[0]) * INITIAL_CAPITAL


def equal_weight_universe_nav(
    prices: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> pd.Series:
    """Buy every universe name equal-weight on day 1, hold to the end (drift)."""
    cols = {}
    for t, df in prices.items():
        s = df["close"].reindex(calendar, method="ffill")
        if s.notna().sum() > 1:
            first_valid = s.first_valid_index()
            cols[t] = s / float(s.loc[first_valid])
    if not cols:
        return pd.Series(dtype=float)
    norm = pd.DataFrame(cols).ffill().dropna(how="all")
    nav = norm.mean(axis=1)
    return nav / float(nav.iloc[0]) * INITIAL_CAPITAL


def metrics(nav: pd.Series) -> Dict:
    """CAGR (calendar-day based), annualised Sharpe, max drawdown, total return."""
    if nav is None or len(nav) < 2:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0}
    nav = nav.dropna()
    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1.0
    daily = nav.pct_change().dropna()
    sharpe = (
        float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))
        if len(daily) > 5 and daily.std() > 1e-12 else 0.0
    )
    max_dd = float((nav / nav.cummax() - 1.0).min())
    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────
def build_report(
    start: str, end: str, top_n: int, universe: List[str], n_loaded: int,
    m_a: Dict, m_b: Dict, m_ew: Dict, m_nifty: Dict, m_overlay: Dict,
    beats_a: float, beats_b: float,
    recs_a: List[Dict],
) -> str:
    def row(name, m):
        return (f"| {name:<34} | {m['total_return_pct']:>9.2f} | {m['cagr_pct']:>8.2f} | "
                f"{m['sharpe']:>7.3f} | {m['max_dd_pct']:>8.2f} |")

    alpha_a = round(m_a["cagr_pct"] - m_nifty["cagr_pct"], 2)
    alpha_b = round(m_b["cagr_pct"] - m_nifty["cagr_pct"], 2)
    overlay_cost = round(m_a["cagr_pct"] - m_overlay["cagr_pct"], 2)

    lines = [
        "# MARK5 Factor-Tilt Buy-and-Hold Baseline — Honest Validation",
        "",
        f"**Period:** {start} → {end}  |  **Top-N:** {top_n}  |  "
        f"**Universe:** {n_loaded}/{len(universe)} loaded",
        f"**Costs:** {PER_SIDE_RATE*100:.2f}%/side + ₹{BROKERAGE_FLAT:.0f}/trade  "
        f"(reused from portfolio_backtest)",
        "",
        "## Summary — net of costs",
        "",
        "| Strategy / Benchmark               | Total % |   CAGR % | Sharpe |  MaxDD % |",
        "|------------------------------------|---------|----------|--------|----------|",
        row("Variant A — always-invested", m_a),
        row("Variant B — regime-filtered (cash)", m_b),
        row("Overlay A/B — A picks + stops/exits", m_overlay),
        row("Equal-weight universe B&H", m_ew),
        row("NIFTY 50 buy-and-hold", m_nifty),
        "",
        "## Verdict vs the motto (beat buy-and-hold, maximise profit)",
        "",
        f"- **Variant A alpha vs NIFTY CAGR:** {alpha_a:+.2f}%/yr  "
        f"({'BEATS' if alpha_a > 0 else 'TRAILS'} the index)  |  "
        f"beats NIFTY in **{beats_a:.0f}%** of quarters",
        f"- **Variant B alpha vs NIFTY CAGR:** {alpha_b:+.2f}%/yr  "
        f"({'BEATS' if alpha_b > 0 else 'TRAILS'} the index)  |  "
        f"beats NIFTY in **{beats_b:.0f}%** of quarters",
        f"- **Selection alpha check:** Variant A CAGR {m_a['cagr_pct']:+.2f}% vs "
        f"equal-weight-universe {m_ew['cagr_pct']:+.2f}%  → "
        f"{'real selection edge' if m_a['cagr_pct'] > m_ew['cagr_pct'] else 'no edge beyond being in midcaps'}",
        f"- **Cost of the overlay (the value destroyer):** holding earns "
        f"**{overlay_cost:+.2f}% CAGR** more than running the same picks through the "
        f"stops/exits overlay.",
        "",
        "## Variant A — per-quarter log",
        "",
        "| Rebalance  | Seg % | NAV | Selected |",
        "|------------|-------|-----|----------|",
    ]
    for rec in recs_a:
        lines.append(f"| {rec['rebalance']} | {rec['seg_return_pct']:>+6.2f} | "
                     f"{rec.get('nav', 0):,.0f} | {rec['selected']} |")
    lines.append("")
    lines.append("> Walk-forward by construction: every quarter's selection uses only "
                 "price history on/before its rebalance date. Returns are net of "
                 "round-trip transaction costs.")
    lines.append("")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MARK5 Factor-Tilt Buy-and-Hold Baseline")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--universe-file", default=None,
                    help="JSON with an 'active_universe' list (or a bare list).")
    ap.add_argument("--tickers", nargs="+", default=None, help="Explicit ticker override.")
    ap.add_argument("--variant", choices=["A", "B", "both"], default="both")
    ap.add_argument("--out", default="reports/factor_tilt_baseline.md")
    args = ap.parse_args()

    universe = load_universe(args)
    logger.info(f"Universe: {len(universe)} tickers | {args.start}→{args.end} | top-{args.top_n}")

    prices = load_prices(universe, args.start, args.end)
    if len(prices) < args.top_n:
        logger.error(
            f"Only {len(prices)} tickers loaded — need ≥ top-N ({args.top_n}). Aborting.\n"
            "  Cause: no OHLCV could be fetched. This happens when the data source\n"
            "  (Yahoo Finance via yfinance) is unreachable — e.g. a restricted network\n"
            "  allowlist, or an empty data/cache/nse/ parquet cache.\n"
            "  Fix: run where the data source is reachable (the cache fills on first run),\n"
            "  or pre-populate data/cache/nse/ with cached parquet files."
        )
        sys.exit(1)

    nifty = fetch_nifty50_index(args.start, args.end)

    # Master trading calendar: NIFTY dates if available, else the longest ticker index.
    if nifty is not None and len(nifty) > 50:
        calendar = nifty.index
    else:
        calendar = max((df.index for df in prices.values()), key=len)
    calendar = calendar[(calendar >= pd.Timestamp(args.start)) & (calendar <= pd.Timestamp(args.end))]
    calendar = pd.DatetimeIndex(sorted(set(calendar)))

    # ── Variant A (always-invested) ────────────────────────────────────────────
    nav_a, recs_a = run_rotation(prices, nifty, calendar, args.top_n, regime_filter=False)
    m_a = metrics(nav_a)

    # ── Variant B (regime-filtered) ─────────────────────────────────────────────
    if args.variant in ("B", "both"):
        nav_b, recs_b = run_rotation(prices, nifty, calendar, args.top_n, regime_filter=True)
        m_b = metrics(nav_b)
    else:
        m_b = metrics(nav_a)

    # ── Benchmarks ──────────────────────────────────────────────────────────────
    m_nifty = metrics(benchmark_nav(nifty, calendar)) if nifty is not None else \
        {"total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0}
    m_ew = metrics(equal_weight_universe_nav(prices, calendar))

    # ── Overlay A/B diagnostic ──────────────────────────────────────────────────
    overlay_final, _ = run_overlay_diagnostic(prices, nifty, calendar, args.top_n)
    # Reconstruct an approximate overlay NAV path for metric comparability:
    # we only need CAGR/total, so build a 2-point bound is insufficient; instead
    # derive CAGR directly from start/end over the same span.
    overlay_total = overlay_final / INITIAL_CAPITAL - 1.0
    span_days = max((calendar[-1] - calendar[0]).days, 1)
    overlay_cagr = (overlay_final / INITIAL_CAPITAL) ** (365.25 / span_days) - 1.0
    m_overlay = {
        "total_return_pct": round(overlay_total * 100, 2),
        "cagr_pct": round(overlay_cagr * 100, 2),
        "sharpe": 0.0,            # segment-level; daily NAV not tracked for the overlay
        "max_dd_pct": 0.0,
    }

    # ── Quarter win-rates vs NIFTY ──────────────────────────────────────────────
    def quarter_beats(recs: List[Dict]) -> float:
        if nifty is None or not recs:
            return 0.0
        wins = tot = 0
        rebal = quarter_rebalance_dates(calendar)
        for qi, rec in enumerate(recs):
            r = pd.Timestamp(rebal[qi])
            seg_end = pd.Timestamp(rebal[qi + 1]) if qi + 1 < len(rebal) else calendar[-1]
            nf = nifty.reindex(calendar[(calendar >= r) & (calendar <= seg_end)], method="ffill").dropna()
            if len(nf) < 2:
                continue
            nifty_ret = float(nf.iloc[-1] / nf.iloc[0] - 1) * 100
            tot += 1
            if rec["seg_return_pct"] > nifty_ret:
                wins += 1
        return 100.0 * wins / tot if tot else 0.0

    beats_a = quarter_beats(recs_a)
    beats_b = quarter_beats(recs_b) if args.variant in ("B", "both") else beats_a

    report = build_report(
        args.start, args.end, args.top_n, universe, len(prices),
        m_a, m_b, m_ew, m_nifty, m_overlay, beats_a, beats_b, recs_a,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(report)

    print("\n" + "=" * 78)
    print(report)
    print("=" * 78)
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
