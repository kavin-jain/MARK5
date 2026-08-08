# Deployment recommendation — scheduled rebalance ~2027-01-24

Per Operating Mandate §6, strategy changes land at the **scheduled rebalance**, not
when they finish testing. Everything below is measured and staged; **nothing here
is live.** The live paper book continues on its current config until January.

Each item cites the experiment that justifies it. Items that FAILED are listed too,
so a future reader can see what was rejected and why rather than re-proposing it.

---

## DEPLOY

### 1. `n_hold` 20 → 60  *(P2.1, 19-year evidence)*

| n_hold | 12 | **20 (now)** | 40 | **60** | 80 | 100 |
|---|---|---|---|---|---|---|
| IR | 0.360 | **0.365** | 0.379 | **0.433** | 0.428 | 0.415 |

IR peaks at 60 exactly where pre-registered. **Falsifies P5's "concentrate to 12"**,
which was a 2016-2026 window artifact. Costs ~0.5pp of CAGR and buys +0.069 IR —
the only metric that measures skill.

`max_weight` must move with it: use `max(0.08, 1.5/n_hold)`.

### 2. Add a long-duration gilt sleeve, ~25%  *(P1.1, P1.1b)*

Target allocation **35 equity / 20 gold / 20 US / 25 gilt**, annual sleeve rebalance.
Instrument: **LTGILTBEES** (the only domestic long-gilt series with usable history).

On real data, 2018-2026 (COVID + the 2022 rate shock):

| Book | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| No bonds (current) | +23.11% | 1.60 | −21.68% | 1.07 |
| **+25% LTGILTBEES** | +19.54% | **1.77** | **−16.39%** | **1.19** |

**This is insurance with a ~1.2pp/yr premium, not free alpha.** It loses in
inflationary rate shocks — it cost 5.3pp of CAGR through 2022-26. Deploy it for the
drawdown, which is the binding constraint on how much can be sized, not for return.

### 3. Keep `deliv_chg` at 10%  *(P3.2)*

Removing it drops IR 0.580 → 0.444. It contributes **+0.136 IR**, twice what the
breadth change buys. Its status should be upgraded from PROVISIONAL to KEEP in the
research log: residual IC +0.0444 after the momentum composite is projected out,
and it is *negatively* correlated with that composite.

### 4. The 1/N ensemble  *(P5.2, approved 2026-08-08)*

Removes config selection entirely, so PBO stops applying. Priced at −1.92pp CAGR
for a better drawdown (−20.95% vs −22.16%). Land it together with the above so the
book is reconstituted once, not twice.

---

## DO NOT DEPLOY — tested and failed

| Item | Verdict | Why |
|---|---|---|
| **Sector-neutral ranking** (P2.2) | ❌ FALSIFIED 0/6 | Cut IR roughly in half at every size. Sector rotation is part of what momentum captures in India; neutralising it removes signal, not noise. |
| **LTCG exit deferral** (P4.1) | ❌ FALSIFIED | Worked as designed — tax −17%, long-term winners 36%→47% — and still lost 0.23pp net. The saving is smaller than the cost of holding the deranked name to collect it. |
| **Silver sleeve** (P1.3) | ❌ FALSIFIED | 79% correlated with gold. A gold proxy, not a sleeve. |

---

## Expected effect on the composite grade

| Category | Now | After | Source |
|---|---|---|---|
| Edge (IR) | 4.0 | 4.6 | P2.1 |
| Risk-adjusted | 7.5 | 8.5 | P1.1b |
| Drawdown | 3.0 | 6.5 | P1.1b |
| Robustness | 4.5 | 6.0 | P5.1/P5.2 |
| **Composite (with 3y live)** | 5.90 | **~7.2** | |

**Not 8.0.** The remaining gap is IC, and the only credible route to it is now
known: ~10 orthogonal non-price signals at IC ≈ 0.04 each, combining as
`√(ΣIC²)`. `deliv_chg` is the first proven one.

---

## Pre-deployment checklist

- [ ] Re-run the full pipeline on `data/pit_cache_2007` with the new config
- [ ] Confirm published artifacts still agree (`TestPublishedArtifactsAgree`)
- [ ] Verify LTGILTBEES liquidity supports the intended position size
- [ ] Update `run_mark6.py` — and fix its stale comment crediting `n_hold=12`
- [ ] Record the reconstitution in `rebalance_events` with `off_cadence: false`
