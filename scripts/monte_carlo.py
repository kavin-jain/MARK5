"""
MARK5 — Monte Carlo Simulation
================================
Three complementary simulation methods to stress-test the momentum portfolio:

  1. Bootstrap Daily Returns
     Resample actual daily returns (with replacement) to build 1,000+ equity paths.
     Shows: distribution of CAGR, MaxDD, Sharpe, Calmar across possible outcomes.
     Answers: "If the same market regime produced differently-ordered days, where do we end up?"

  2. Trade Sequence Shuffle
     Shuffle the order of actual trades, replay the portfolio 1,000+ times.
     Answers: "How much does lucky/unlucky sequencing of wins vs losses affect the result?"
     Key risk: if big losses cluster at the start, the circuit breaker fires early.

  3. Trade Return Bootstrap
     Sample 42 trades with replacement from the actual 42 trades, replay portfolio.
     Answers: "If we had drawn a different mix of trades from the same distribution, what
     range of outcomes would we see?" Reveals how HAL-dominated the upside is.

Usage:
  python3 scripts/monte_carlo.py                        # default 1000 sims
  python3 scripts/monte_carlo.py --n-sims 5000          # more sims
  python3 scripts/monte_carlo.py --seed 42              # reproducible
  python3 scripts/monte_carlo.py --results-file reports/momentum_portfolio_results.json

PAPER MODE ONLY — simulation of a paper trading strategy. Never used for live trading.
"""
import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DEFAULT = os.path.join(_ROOT, "reports", "momentum_portfolio_results.json")
REPORTS_DIR     = os.path.join(_ROOT, "reports")

# Portfolio constants (must match momentum_portfolio.py)
ALLOC_PER_TRADE  = 0.25   # 25% base allocation per position
COST_ROUND_TRIP  = 0.0029 # 0.29% round-trip transaction cost
LTCG_RATE        = 0.125  # 12.5% for holds > 365 days
STCG_RATE        = 0.200  # 20% for holds ≤ 365 days
LTCG_EXEMPT_PA   = 125_000.0  # ₹1.25L annual exemption


# ─── Core Monte Carlo functions (importable by tests) ────────────────────────

def load_equity_curve(results_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load equity curve and trade list from momentum_portfolio results JSON.

    Returns:
        dates      : np.ndarray of date strings (shape N,)
        equity_vals: np.ndarray of equity floats  (shape N,)
        meta       : dict with strategy metadata (initial_capital, years, etc.)
    """
    with open(results_path) as f:
        data = json.load(f)

    if "equity_curve" not in data:
        raise ValueError(
            "equity_curve not found in results JSON. "
            "Re-run scripts/momentum_portfolio.py to regenerate the file."
        )

    ec = data["equity_curve"]  # list of [date_str, equity_float]
    dates  = np.array([row[0] for row in ec], dtype="U16")
    equity = np.array([row[1] for row in ec], dtype=float)

    meta = {
        "initial_capital":   float(data["initial_capital"]),
        "final_equity_net":  float(data["final_equity_net"]),
        "net_cagr_pct":      float(data["net_cagr_pct"]),
        "max_drawdown_pct":  float(data["max_drawdown_pct"]),
        "sharpe":            float(data["sharpe"]),
        "years":             float(data["years"]),
        "n_trades":          int(data["n_trades"]),
        "win_rate_pct":      float(data["win_rate_pct"]),
        "annual_returns":    data["annual_returns"],
        "trades":            data["trades"],
    }
    return dates, equity, meta


def daily_log_returns(equity: np.ndarray) -> np.ndarray:
    """Compute daily log-returns from equity curve. Returns array of length N-1."""
    return np.log(equity[1:] / equity[:-1])


def compute_max_drawdown(equity_path: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown for a single equity path.
    Returns a negative float, e.g. -0.21 means -21%.
    """
    if len(equity_path) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_path)
    dd   = (equity_path - peak) / peak
    return float(np.min(dd))


def compute_sharpe(log_returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    if len(log_returns) < 2:
        return 0.0
    mu    = float(np.mean(log_returns)) * trading_days_per_year
    sigma = float(np.std(log_returns, ddof=1)) * np.sqrt(trading_days_per_year)
    return mu / sigma if sigma > 0 else 0.0


def compute_cagr(initial: float, final: float, years: float) -> float:
    """CAGR as a decimal. Returns 0.0 on bad inputs."""
    if initial <= 0 or final <= 0 or years <= 0:
        return 0.0
    return float((final / initial) ** (1.0 / years) - 1.0)


def bootstrap_daily_returns(
    log_returns: np.ndarray,
    initial_capital: float,
    n_simulations: int = 1_000,
    n_days: Optional[int] = None,
    years: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Resample daily log-returns with replacement to build N equity paths.

    Args:
        log_returns:      Daily log-returns from actual backtest (shape D,)
        initial_capital:  Starting equity value
        n_simulations:    Number of Monte Carlo paths
        n_days:           Days per path (defaults to len(log_returns))
        years:            Years corresponding to n_days (for CAGR calc)
        seed:             RNG seed for reproducibility

    Returns:
        equity_paths: ndarray of shape (n_simulations, n_days)
                      Each row is one simulated equity curve.
    """
    rng    = np.random.default_rng(seed)
    n_days = n_days or len(log_returns)

    # Resample indices → sampled log-returns → cumulative product
    idx     = rng.integers(0, len(log_returns), size=(n_simulations, n_days))
    sampled = log_returns[idx]                     # (n_sims, n_days)
    growth  = np.exp(np.cumsum(sampled, axis=1))   # cumulative log-return → multiplier
    paths   = initial_capital * growth             # (n_sims, n_days)
    return paths


def trade_sequence_shuffle(
    trade_pnl_pcts: np.ndarray,
    trade_hold_days: np.ndarray,
    initial_capital: float,
    alloc_per_trade: float = ALLOC_PER_TRADE,
    n_simulations: int = 1_000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the order of actual trades N times to measure sequence risk.

    Uses a simplified sequential model: each trade runs one-at-a-time, and
    the portfolio allocates `alloc_per_trade` of current equity. This is an
    approximation (the real portfolio has overlapping positions), but it
    captures whether clustering of losses at the start triggers the circuit
    breaker prematurely.

    Returns:
        final_equities : ndarray shape (n_simulations,) — final equity per sim
        total_returns  : ndarray shape (n_simulations,) — total return as decimal
    """
    rng      = np.random.default_rng(seed)
    n_trades = len(trade_pnl_pcts)

    final_equities = np.empty(n_simulations)

    for i in range(n_simulations):
        order  = rng.permutation(n_trades)
        pnls   = trade_pnl_pcts[order] / 100.0   # fraction, e.g. 0.42
        equity = initial_capital

        for pnl_frac in pnls:
            # Allocate `alloc_per_trade` of current equity to this trade.
            # Net: subtract cost from the gross PnL fraction.
            gross_pnl = alloc_per_trade * equity * pnl_frac
            cost      = alloc_per_trade * equity * COST_ROUND_TRIP
            equity   += gross_pnl - cost

        final_equities[i] = equity

    total_returns = final_equities / initial_capital - 1.0
    return final_equities, total_returns


def trade_return_bootstrap(
    trade_pnl_pcts: np.ndarray,
    trade_hold_days: np.ndarray,
    initial_capital: float,
    n_trades_per_sim: Optional[int] = None,
    alloc_per_trade: float = ALLOC_PER_TRADE,
    years: float = 4.4,
    n_simulations: int = 1_000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample trades with replacement from the actual trade distribution.

    Answers: "What if we had drawn a different set of trades from the same
    underlying distribution?" This reveals how concentrated the upside is
    (e.g. how much depends on the single HAL +195% trade).

    Returns:
        final_equities : ndarray shape (n_simulations,)
        cagrs          : ndarray shape (n_simulations,) — annualised returns
    """
    rng              = np.random.default_rng(seed)
    n_actual         = len(trade_pnl_pcts)
    n_trades_per_sim = n_trades_per_sim or n_actual

    final_equities = np.empty(n_simulations)

    for i in range(n_simulations):
        idx    = rng.integers(0, n_actual, size=n_trades_per_sim)
        pnls   = trade_pnl_pcts[idx] / 100.0
        equity = initial_capital

        for pnl_frac in pnls:
            gross_pnl = alloc_per_trade * equity * pnl_frac
            cost      = alloc_per_trade * equity * COST_ROUND_TRIP
            equity   += gross_pnl - cost

        final_equities[i] = equity

    cagrs = (final_equities / initial_capital) ** (1.0 / years) - 1.0
    return final_equities, cagrs


def path_statistics(
    equity_paths: np.ndarray,
    initial_capital: float,
    years: float,
    trading_days_per_year: int = 252,
) -> Dict:
    """
    Compute CAGR, MaxDD, Sharpe, Calmar distributions across all paths.

    Args:
        equity_paths: ndarray (n_simulations, n_days)
        initial_capital: starting equity
        years: duration for CAGR calculation

    Returns:
        dict with arrays 'cagr', 'max_dd', 'sharpe', 'calmar', each shape (n_sims,)
    """
    n_sims, n_days = equity_paths.shape

    # CAGR from final value
    final   = equity_paths[:, -1]
    cagrs   = (final / initial_capital) ** (1.0 / years) - 1.0

    # MaxDD per path — vectorised
    peak    = np.maximum.accumulate(equity_paths, axis=1)
    dd_all  = (equity_paths - peak) / peak
    max_dds = dd_all.min(axis=1)                    # most negative, one per sim

    # Sharpe per path
    log_rets     = np.diff(np.log(equity_paths), axis=1)  # (n_sims, n_days-1)
    mu_ann       = log_rets.mean(axis=1) * trading_days_per_year
    sigma_ann    = log_rets.std(axis=1, ddof=1) * np.sqrt(trading_days_per_year)
    sharpes      = np.where(sigma_ann > 0, mu_ann / sigma_ann, 0.0)

    # Calmar = CAGR / |MaxDD|
    with np.errstate(divide="ignore", invalid="ignore"):
        calmars = np.where(max_dds < 0, cagrs / np.abs(max_dds), 0.0)

    return {
        "cagr":    cagrs,
        "max_dd":  max_dds,
        "sharpe":  sharpes,
        "calmar":  calmars,
        "final":   final,
    }


def percentile_table(values: np.ndarray) -> Dict[str, float]:
    """Return p5, p25, p50, p75, p95 percentiles."""
    ps = np.percentile(values, [5, 25, 50, 75, 95])
    return {"p5": ps[0], "p25": ps[1], "p50": ps[2], "p75": ps[3], "p95": ps[4]}


# ─── Report generation ───────────────────────────────────────────────────────

def _pct(v: float, decimals: int = 1) -> str:
    return f"{v * 100:+.{decimals}f}%"


def print_mc_report(
    meta: dict,
    bootstrap_stats: Dict,
    shuffle_returns: np.ndarray,
    bootstrap_trade_cagrs: np.ndarray,
    n_simulations: int,
) -> dict:
    """Print a formatted Monte Carlo report and return the raw numbers."""

    orig_cagr  = meta["net_cagr_pct"] / 100
    orig_dd    = meta["max_drawdown_pct"] / 100
    orig_sharpe = meta["sharpe"]
    orig_calmar = orig_cagr / abs(orig_dd) if orig_dd != 0 else 0.0

    cagrs    = bootstrap_stats["cagr"]
    max_dds  = bootstrap_stats["max_dd"]
    sharpes  = bootstrap_stats["sharpe"]
    calmars  = bootstrap_stats["calmar"]

    # Probability thresholds
    p_pos   = float(np.mean(cagrs > 0.00))
    p_10    = float(np.mean(cagrs > 0.10))
    p_15    = float(np.mean(cagrs > 0.15))
    p_20    = float(np.mean(cagrs > 0.20))

    # VaR / CVaR
    var_95  = float(np.percentile(cagrs, 5))   # annual return at 5th percentile
    cvar_95 = float(np.mean(cagrs[cagrs <= var_95]))

    W = 74
    print("\n" + "═" * W)
    print("  MARK5 MOMENTUM PORTFOLIO — MONTE CARLO STRESS TEST")
    print("═" * W)
    print(f"  Simulations : {n_simulations:,} paths (bootstrap daily returns)")
    print(f"  Period      : {meta['years']:.1f} years ({len(meta['trades'])} trades)")
    print()

    print("  ┌─ ORIGINAL BACKTEST RESULT ──────────────────────────────────┐")
    print(f"  │  CAGR (net): {_pct(orig_cagr):>8}   "
          f"MaxDD: {_pct(orig_dd):>8}   Sharpe: {orig_sharpe:.3f}   "
          f"Calmar: {orig_calmar:.2f}  │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    print("  Bootstrap distribution (resampled daily returns):")
    print(f"  {'Metric':<18}  {'p5':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}")
    print("  " + "─" * 60)

    def row(label, arr, fmt="pct"):
        pt = percentile_table(arr)
        if fmt == "pct":
            vals = [_pct(pt[k]) for k in ("p5","p25","p50","p75","p95")]
        else:
            vals = [f"{pt[k]:+.2f}" for k in ("p5","p25","p50","p75","p95")]
        print(f"  {label:<18}  {'  '.join(f'{v:>8}' for v in vals)}")

    row("CAGR (annual)",  cagrs)
    row("Max Drawdown",   max_dds)
    row("Sharpe",         sharpes, fmt="float")
    row("Calmar",         calmars, fmt="float")
    print()

    print("  Probability of exceeding CAGR threshold:")
    print(f"    P(CAGR > 0%) :   {p_pos * 100:.1f}%")
    print(f"    P(CAGR > 10%):   {p_10 * 100:.1f}%")
    print(f"    P(CAGR > 15%):   {p_15 * 100:.1f}%")
    print(f"    P(CAGR > 20%):   {p_20 * 100:.1f}%")
    print()

    print("  Risk metrics (annual, 95% confidence):")
    print(f"    VaR  (worst 5% of outcomes)     : {_pct(var_95)}")
    print(f"    CVaR (avg of worst 5% outcomes) : {_pct(cvar_95)}")
    print()

    # Trade shuffle analysis
    p50_sh = float(np.percentile(shuffle_returns, 50))
    p5_sh  = float(np.percentile(shuffle_returns,  5))
    p95_sh = float(np.percentile(shuffle_returns, 95))
    years  = meta["years"]
    cagr_p50 = (1 + p50_sh) ** (1 / years) - 1
    cagr_p5  = (1 + p5_sh)  ** (1 / years) - 1
    cagr_p95 = (1 + p95_sh) ** (1 / years) - 1

    print("  Trade Sequence Shuffle (same trades, different order):")
    print(f"    Worst  5% CAGR: {_pct(cagr_p5)}")
    print(f"    Median    CAGR: {_pct(cagr_p50)}")
    print(f"    Best  95% CAGR: {_pct(cagr_p95)}")
    spread = cagr_p95 - cagr_p5
    print(f"    Sequence risk spread: {_pct(spread)} (p5→p95 range)")
    print()

    # Trade bootstrap analysis
    tb_p5  = float(np.percentile(bootstrap_trade_cagrs,  5))
    tb_p50 = float(np.percentile(bootstrap_trade_cagrs, 50))
    tb_p95 = float(np.percentile(bootstrap_trade_cagrs, 95))
    pct_positive = float(np.mean(bootstrap_trade_cagrs > 0.0))
    pct_10       = float(np.mean(bootstrap_trade_cagrs > 0.10))

    print("  Trade Return Bootstrap (sampling 42 trades with replacement):")
    print(f"    p5%  CAGR: {_pct(tb_p5)}")
    print(f"    p50% CAGR: {_pct(tb_p50)}")
    print(f"    p95% CAGR: {_pct(tb_p95)}")
    print(f"    P(CAGR > 0%)  : {pct_positive * 100:.1f}%")
    print(f"    P(CAGR > 10%) : {pct_10 * 100:.1f}%")
    print()

    # HAL concentration analysis
    trades     = meta["trades"]
    hal_trades = [t for t in trades if t["ticker"] == "HAL"]
    hal_pnl    = sum(t["net_pnl"] for t in hal_trades)
    total_pnl  = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    hal_pct    = hal_pnl / total_pnl * 100 if total_pnl > 0 else 0

    top1_pnl   = max(t["net_pnl"] for t in trades)
    top1_trade = next(t for t in trades if t["net_pnl"] == top1_pnl)
    top1_pct   = top1_pnl / total_pnl * 100 if total_pnl > 0 else 0

    print("  Concentration analysis:")
    print(f"    HAL total PnL:              ₹{hal_pnl/1e5:.1f}L  ({hal_pct:.0f}% of gross profit)")
    print(f"    Single best trade ({top1_trade['ticker']}): "
          f"₹{top1_pnl/1e5:.1f}L  ({top1_pct:.0f}% of gross profit)")
    print(f"    Total gross profit:         ₹{total_pnl/1e5:.1f}L")
    print()

    # Verdict
    print("  Verdict:")
    if p_10 > 0.50:
        verdict_cagr = f"MORE LIKELY than not to exceed 10% CAGR (P={p_10*100:.0f}%)"
    else:
        verdict_cagr = f"LESS LIKELY than not to exceed 10% CAGR (P={p_10*100:.0f}%)"

    spread_pct = spread * 100
    if spread_pct < 15:
        verdict_seq = "LOW sequence risk (order of trades barely matters)"
    elif spread_pct < 25:
        verdict_seq = "MODERATE sequence risk (early losses hurt noticeably)"
    else:
        verdict_seq = "HIGH sequence risk (bad early runs significantly affect outcomes)"

    print(f"    → {verdict_cagr}")
    print(f"    → {verdict_seq}")

    hal_frac = hal_pnl / total_pnl if total_pnl > 0 else 0
    if hal_frac > 0.50:
        print(f"    → ⚠️  HAL drives {hal_frac*100:.0f}% of gross profit — upside is HAL-concentrated")
    else:
        print(f"    → ✅ No single stock exceeds 50% of gross profit — reasonably diversified")

    print("═" * W + "\n")

    return {
        "n_simulations": n_simulations,
        "original": {
            "net_cagr":   round(orig_cagr * 100, 2),
            "max_dd":     round(orig_dd * 100, 2),
            "sharpe":     orig_sharpe,
            "calmar":     round(orig_calmar, 3),
        },
        "bootstrap_daily": {
            "cagr":    {k: round(v * 100, 2) for k, v in percentile_table(cagrs).items()},
            "max_dd":  {k: round(v * 100, 2) for k, v in percentile_table(max_dds).items()},
            "sharpe":  {k: round(v, 3) for k, v in percentile_table(sharpes).items()},
            "calmar":  {k: round(v, 3) for k, v in percentile_table(calmars).items()},
            "p_above_0pct":  round(p_pos * 100, 1),
            "p_above_10pct": round(p_10 * 100, 1),
            "p_above_15pct": round(p_15 * 100, 1),
            "p_above_20pct": round(p_20 * 100, 1),
            "var_95":  round(var_95 * 100, 2),
            "cvar_95": round(cvar_95 * 100, 2),
        },
        "trade_shuffle": {
            "cagr_p5":  round(cagr_p5 * 100, 2),
            "cagr_p50": round(cagr_p50 * 100, 2),
            "cagr_p95": round(cagr_p95 * 100, 2),
            "spread":   round(spread * 100, 2),
        },
        "trade_bootstrap": {
            "cagr_p5":  round(tb_p5 * 100, 2),
            "cagr_p50": round(tb_p50 * 100, 2),
            "cagr_p95": round(tb_p95 * 100, 2),
            "p_positive": round(pct_positive * 100, 1),
            "p_above_10": round(pct_10 * 100, 1),
        },
        "concentration": {
            "hal_pct_of_gross_profit":   round(hal_pct, 1),
            "top1_pct_of_gross_profit":  round(top1_pct, 1),
            "total_gross_profit_lakhs":  round(total_pnl / 1e5, 1),
        },
    }


# ─── Main entry point ────────────────────────────────────────────────────────

def run_monte_carlo(
    results_path: str = RESULTS_DEFAULT,
    n_simulations: int = 1_000,
    seed: Optional[int] = None,
    save: bool = True,
) -> dict:
    """
    Run all three Monte Carlo methods against the momentum portfolio results.

    Args:
        results_path: Path to momentum_portfolio_results.json
        n_simulations: Number of simulation paths
        seed: RNG seed for reproducibility (None = random)
        save: Whether to save results JSON to reports/

    Returns:
        dict with all simulation statistics
    """
    print(f"\nLoading backtest results from: {results_path}")
    dates, equity, meta = load_equity_curve(results_path)
    log_rets = daily_log_returns(equity)

    initial = meta["initial_capital"]
    years   = meta["years"]

    trades      = meta["trades"]
    pnl_pcts    = np.array([t["pnl_pct"]  for t in trades], dtype=float)
    hold_days   = np.array([t["hold_days"] for t in trades], dtype=float)

    print(f"  {len(equity)} days · {len(trades)} trades · {years:.1f} years")
    print(f"  Running {n_simulations:,} simulations…", end="", flush=True)

    # ── Method 1: Bootstrap daily returns ──────────────────────────────────
    paths = bootstrap_daily_returns(
        log_rets, initial,
        n_simulations=n_simulations,
        years=years,
        seed=seed,
    )
    bstats = path_statistics(paths, initial, years)
    print(".", end="", flush=True)

    # ── Method 2: Trade sequence shuffle ───────────────────────────────────
    _, shuffle_returns = trade_sequence_shuffle(
        pnl_pcts, hold_days, initial,
        n_simulations=n_simulations,
        seed=seed,
    )
    print(".", end="", flush=True)

    # ── Method 3: Trade return bootstrap ───────────────────────────────────
    _, tb_cagrs = trade_return_bootstrap(
        pnl_pcts, hold_days, initial,
        years=years,
        n_simulations=n_simulations,
        seed=seed,
    )
    print(" done\n")

    # ── Report ─────────────────────────────────────────────────────────────
    results = print_mc_report(meta, bstats, shuffle_returns, tb_cagrs, n_simulations)

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        out = os.path.join(REPORTS_DIR, "monte_carlo_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved → {out}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARK5 Monte Carlo Simulation")
    parser.add_argument("--n-sims",       type=int, default=1_000,
                        help="Number of simulations (default: 1000)")
    parser.add_argument("--seed",         type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--results-file", type=str, default=RESULTS_DEFAULT,
                        help="Path to momentum_portfolio_results.json")
    parser.add_argument("--no-save",      action="store_true",
                        help="Skip saving results JSON")
    args = parser.parse_args()

    run_monte_carlo(
        results_path=args.results_file,
        n_simulations=args.n_sims,
        seed=args.seed,
        save=not args.no_save,
    )
