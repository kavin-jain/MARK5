"""
MARK6 — Overfitting statistics (Bailey & López de Prado)
========================================================
Rigorous answers to "is the backtest's Sharpe real or luck given how many
strategies we tried?":

  - probabilistic_sharpe_ratio (PSR): P(true SR > benchmark SR), adjusting for
    sample length, skewness and (fat-tailed) kurtosis of the returns.
  - deflated_sharpe_ratio (DSR): PSR with the benchmark set to the EXPECTED MAXIMUM
    Sharpe achievable by pure luck across N trials -> deflates for multiple testing.
  - pbo_cscv: Probability of Backtest Overfitting via Combinatorially-Symmetric
    Cross-Validation — the fraction of train/test splits where the in-sample-best
    strategy underperforms the median out-of-sample.

All operate on per-period (e.g. daily) returns; do NOT pre-annualise the Sharpe.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats as _ss

EULER = 0.5772156649015329


def _sharpe(ret: np.ndarray) -> float:
    sd = ret.std(ddof=1)
    return ret.mean() / sd if sd > 0 else 0.0


def probabilistic_sharpe_ratio(ret: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark). ret = per-period returns."""
    ret = np.asarray(ret, float)
    ret = ret[np.isfinite(ret)]
    n = len(ret)
    if n < 10:
        return float("nan")
    sr = _sharpe(ret)
    g3 = _ss.skew(ret)
    g4 = _ss.kurtosis(ret, fisher=False)   # non-excess (normal = 3)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(_ss.norm.cdf(z))


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """Expected max per-period Sharpe under the null across N independent trials."""
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    a = _ss.norm.ppf(1 - 1.0 / n_trials)
    b = _ss.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return sr_std * ((1 - EULER) * a + EULER * b)


def deflated_sharpe_ratio(ret: np.ndarray, all_trial_sharpes: list[float]) -> dict:
    """DSR for the chosen strategy given the Sharpes of ALL trials attempted."""
    ret = np.asarray(ret, float)
    sr_std = float(np.std(all_trial_sharpes, ddof=1)) if len(all_trial_sharpes) > 1 else 0.0
    n_trials = len(all_trial_sharpes)
    sr_star = expected_max_sharpe(sr_std, n_trials)
    return {
        "n_trials": n_trials,
        "observed_sharpe_daily": _sharpe(ret[np.isfinite(ret)]),
        "psr_vs_zero": probabilistic_sharpe_ratio(ret, 0.0),
        "expected_max_sharpe_luck": sr_star,
        "deflated_sharpe": probabilistic_sharpe_ratio(ret, sr_star),
    }


def pbo_cscv(returns_matrix: np.ndarray, n_splits: int = 12) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    returns_matrix: shape (T observations, N strategies) of per-period returns.
    Splits T into n_splits groups; over all C(n_splits, n_splits/2) train/test
    partitions, finds the in-sample-best strategy and records its out-of-sample
    rank. PBO = fraction where that rank is below the OOS median (logit <= 0).
    """
    M = np.asarray(returns_matrix, float)
    T, N = M.shape
    if N < 2 or n_splits % 2 or T < n_splits * 3:
        return {"pbo": float("nan"), "n_combos": 0, "n_strategies": N}
    rows = np.array_split(np.arange(T), n_splits)
    groups = list(range(n_splits))
    logits, n_below = [], 0
    combos = list(combinations(groups, n_splits // 2))
    for tr in combos:
        te = [g for g in groups if g not in tr]
        tr_idx = np.concatenate([rows[g] for g in tr])
        te_idx = np.concatenate([rows[g] for g in te])
        sr_is = np.array([_sharpe(M[tr_idx, j]) for j in range(N)])
        sr_oos = np.array([_sharpe(M[te_idx, j]) for j in range(N)])
        best = int(np.argmax(sr_is))
        # OOS relative rank of the IS-best (1 = best OOS, ->0 = worst)
        rank = (_ss.rankdata(sr_oos)[best]) / (N + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        lam = math.log(rank / (1 - rank))
        logits.append(lam)
        if lam <= 0:
            n_below += 1
    return {
        "pbo": n_below / len(combos),
        "n_combos": len(combos),
        "n_strategies": N,
        "median_logit": float(np.median(logits)),
    }
