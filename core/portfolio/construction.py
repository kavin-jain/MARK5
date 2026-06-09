"""
MARK6 — Portfolio Construction
==============================
Turns point-in-time factor scores into target weights, with risk controls that
were chosen to *preserve* return (the session proved that return-killing overlays
like stop-losses and regime-to-cash must be avoided).

Construction modes:
  - "equal_weight" : equal-weight the eligible universe (the honest smart-beta
                     core; the bar everything must beat net of tax).
  - "factor_tilt"  : hold top-N by multi-factor composite, weighted by a blend of
                     equal / inverse-vol / composite tilt. Always fully invested.

Risk controls (all weight-space, never timing):
  - inverse-vol weighting   -> directly attacks the ~40% drawdown problem
  - max single-name weight  -> idiosyncratic cap
  - max sector weight       -> concentration cap (optional, needs sector map)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConstructionConfig:
    mode: str = "factor_tilt"            # "equal_weight" | "factor_tilt"
    n_hold: int = 20                      # names held in factor_tilt mode
    factor_weights: dict = field(default_factory=lambda: {
        "momentum": 0.30, "low_vol": 0.30, "trend": 0.20, "stability": 0.20})
    base_weighting: str = "inverse_vol"  # "equal" | "inverse_vol"
    tilt_strength: float = 0.50           # 0 = pure base weighting; >0 tilts to score
    max_weight: float = 0.08              # cap per name
    max_sector_weight: float = 0.30       # cap per sector (if sector map provided)
    buffer_mult: float = 2.0              # hold a name until it leaves top n_hold*mult


def _cap_weights(w: pd.Series, max_weight: float) -> pd.Series:
    """Cap weights at max_weight via water-filling (converges; never re-violates).

    If the cap is infeasible (n_names * max_weight < 1), returns equal weight —
    the closest attainable allocation. In real configs max_weight >= 1/n_hold so
    this branch never fires.
    """
    w = w[w > 0].astype(float)
    if w.empty:
        return w
    w = w / w.sum()
    n = len(w)
    if n * max_weight <= 1.0 + 1e-12:
        return pd.Series(1.0 / n, index=w.index)
    capped: dict = {}
    free = list(w.index)
    budget = 1.0
    for _ in range(200):
        fsum = w[free].sum()
        if fsum <= 0:
            break
        fw = w[free] / fsum * budget
        over = [f for f in free if fw[f] > max_weight + 1e-15]
        if not over:
            for f in free:
                capped[f] = float(fw[f])
            break
        for f in over:
            capped[f] = max_weight
        budget -= max_weight * len(over)
        free = [f for f in free if f not in over]
        if not free:
            break
    out = pd.Series(capped).reindex(w.index).fillna(0.0)
    return out / out.sum()


class PortfolioConstructor:
    def __init__(self, config: ConstructionConfig, sector_map: dict | None = None):
        self.cfg = config
        self.sector_map = sector_map or {}

    def select(self, composite: pd.Series, currently_held: list[str]) -> list[str]:
        """Buffered selection: keep held names still in the top n_hold*buffer_mult,
        fill the rest with the highest-scoring new names. Cuts turnover -> LTCG."""
        cfg = self.cfg
        ranked = composite.sort_values(ascending=False)
        if cfg.mode == "equal_weight":
            return list(ranked.index)
        rank_of = {t: i for i, t in enumerate(ranked.index)}
        exit_rank = int(cfg.n_hold * cfg.buffer_mult)
        keep = [t for t in currently_held if rank_of.get(t, 10**9) < exit_rank][:cfg.n_hold]
        adds = [t for t in ranked.index if t not in keep][:max(0, cfg.n_hold - len(keep))]
        return (keep + adds)[:cfg.n_hold]

    def target_weights(self, composite: pd.Series, recent_vol: pd.Series,
                       currently_held: list[str]) -> pd.Series:
        """Compute target weights for the selected holding set."""
        cfg = self.cfg
        picks = self.select(composite, currently_held)
        if not picks:
            return pd.Series(dtype=float)

        # base weighting
        if cfg.base_weighting == "inverse_vol" and recent_vol is not None:
            iv = (1.0 / recent_vol.reindex(picks).clip(lower=1e-3)).fillna(0.0)
            base = iv / iv.sum() if iv.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
        else:
            base = pd.Series(1.0 / len(picks), index=picks)

        # multiplicative score tilt (kept mild; tilt_strength bounded)
        if cfg.mode == "factor_tilt" and cfg.tilt_strength > 0:
            z = composite.reindex(picks).fillna(0.0)
            mult = (1.0 + cfg.tilt_strength * z).clip(lower=0.1)
            w = base * mult
        else:
            w = base
        w = w / w.sum()

        # sector cap (optional)
        if self.sector_map and cfg.max_sector_weight < 1.0:
            w = self._apply_sector_cap(w)
        # name cap
        return _cap_weights(w, cfg.max_weight)

    def _apply_sector_cap(self, w: pd.Series) -> pd.Series:
        """Scale over-cap sectors down to the cap and redistribute the freed
        weight to NON-capped sector members. Capped sectors are frozen so they
        can't creep back over -> guaranteed convergence (no oscillation)."""
        cap = self.cfg.max_sector_weight
        w = w.astype(float).copy()
        sectors_present = {self.sector_map.get(t, t) for t in w.index}
        if len(sectors_present) * cap <= 1.0 + 1e-9:
            return w / w.sum()        # cap infeasible for this many sectors -> best effort
        frozen: set = set()
        for _ in range(100):
            sect: dict = {}
            for t, wt in w.items():
                s = self.sector_map.get(t, t)
                sect[s] = sect.get(s, 0.0) + wt
            over = {s: v for s, v in sect.items() if v > cap + 1e-12 and s not in frozen}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            for s, v in over.items():
                members = [t for t in w.index if self.sector_map.get(t, t) == s]
                w[members] *= cap / v
                frozen.add(s)
            under = [t for t in w.index if self.sector_map.get(t, t) not in frozen]
            usum = w[under].sum()
            if not under or usum <= 0:
                break
            w[under] += excess * w[under] / usum
        return w / w.sum()
