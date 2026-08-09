# Deployment recommendation — scheduled rebalance ~2027-01-24

Per Operating Mandate §6, strategy changes land at the **scheduled rebalance**, not
when they finish testing. Everything below is measured and staged; **nothing here
is live.** The live paper book continues on its current config until January.

Each item cites the experiment that justifies it. Items that FAILED are listed too,
so a future reader can see what was rejected and why rather than re-proposing it.

---

## DEPLOY

### 1. ~~`n_hold` 20 → 60~~ — **DECLINED 2026-08-09. Stay at 20.**

| n_hold | 12 | **20 (deployed)** | 40 | 60 | 80 |
|---|---|---|---|---|---|
| IR | 0.360 | **0.365** | 0.379 | 0.433 | 0.428 |
| **t-stat** | 1.57 | **1.59** | 1.66 | **1.90** | 1.87 |
| Net CAGR | 12.64% | 12.05% | 11.40% | 11.49% | 11.07% |
| Tracking error | 14.2pp | 12.0pp | 9.7pp | 8.4pp | 7.6pp |
| Turnover/yr | 316% | 281% | 239% | 198% | 178% |
| ₹/position at 25% equity | ₹10,831 | ₹6,499 | ₹3,249 | **₹2,166** | ₹1,625 |

IR does peak near 60, and the original entry above was right that IR is the metric
that measures skill. Two things kill it anyway:

**It is not statistically distinguishable.** The t-statistic tops out at **1.90** —
below this project's own 3.0 bar (§5, Harvey/Liu/Zhu) and below even the conventional
2.0. The gap between IR 0.365 and 0.433 is inside the noise. Changing a live config on
that is the precise pattern §4 says killed 22 previous ideas.

**It is not implementable at this book's size, and failing quietly is the worst part.**
25% equity across 60 names is ~₹2,166 a position at the current NAV. Six of today's 22
holdings cost more than that for a *single share*. The allocator drops what it cannot
afford, and affordability tracks SHARE PRICE — which is arbitrary, since a company can
split 1:100 and change nothing about itself. The book would acquire a systematic tilt
toward low-priced shares: a bet nobody chose, nobody tested, and that appears nowhere
in the config to be disclosed.

**Why the evidence could not have caught this.** The breadth sweep runs in NAV units —
scale-free, fractional shares implicit, a 1.6% weight is always exactly 1.6%. Whole
shares exist only in the live book. So the IR finding is real *and* silent on whether
n_hold=60 is buyable at ₹5 lakh. It becomes viable somewhere above ₹25 lakh.

In the Fundamental Law: n_hold=60 buys **breadth**, and unfillable names make
**transfer** pay for it. At this capital the trade is not worth making.

*`max_weight` stays 0.08, unchanged.*

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

### 4. Equal weights, FOUR sleeves — 25% each  *(A2; owner decision 2026-08-09)*

**DECIDED: equity 25 / gold 25 / US equity 25 / long gilt 25.**

| | CAGR | Sharpe | Vol | MaxDD | Calmar |
|---|---|---|---|---|---|
| **4-sleeve equal weight** | **+16.16%** | **1.43** | 11.2% | **−17.09%** | 0.95 |

Rationale: A2 showed naive equal weights beat the searched 30/30/10/30 allocation
on 5-year worst (+11.88% vs +10.62%) and median (+17.46% vs +16.01%). The search
bought 1.9pp of drawdown for 1.45pp of return. **Nothing fitted means nothing to
overfit** — which is the point of the whole exercise.

### 4b. SHELVED — the fifth sleeve (US short-duration treasuries)

**Tested, cleared, and deliberately NOT deployed. Revisit later.**

R2 plus the FX-frozen control test showed the fifth sleeve is genuine
diversification, not a currency artefact: at identical 80% USD exposure it is
worth 4.60pp of drawdown, and the advantage GROWS when FX is frozen (+7.19pp vs
+5.17pp live). MaxDD −17.09% → −12.79%, Calmar 0.95 → 1.09.

**Why it is shelved anyway: SHY is a US instrument.** Owning it from India needs
the RBI Liberalised Remittance Scheme — legal and routine, but it adds paperwork,
annual limits, and an operational dependency that conflicts with the owner's
stated goal of a hands-off system. A domestic short-duration debt fund is NOT a
substitute: it carries no dollar exposure, and the FX test showed the dollar
exposure is where the benefit comes from.

**Open question for later:** find a fifth sleeve that is genuinely uncorrelated,
USD-denominated, and buyable from India without LRS friction — or decide the LRS
route is acceptable. The evidence for adding it is already in hand; only the
plumbing is missing.

<!-- superseded detail retained for the record -->
**The five-sleeve evidence.** Two findings drove it:

- **A2:** naive equal weights beat the searched allocation on 5-year worst
  (+11.88% vs +10.62%) and median (+17.46% vs +16.01%). The search bought 1.9pp of
  drawdown for 1.45pp of return. Nothing fitted means nothing to overfit.
- **R2:** adding short-duration treasuries (SHY) as a fifth equal sleeve cuts
  MaxDD −17.09% → −13.09% and lifts Calmar 0.95 → 1.09.

**FX-frozen control test (the one that mattered).** SHY is a USD asset, so the
5-sleeve book is 80% USD against the 4-sleeve's 75% — the improvement could have
been more short-INR exposure rather than diversification. Tested against a
4-sleeve control re-weighted to the *same* 80% USD:

| Book | USD% | MaxDD live | MaxDD frozen |
|---|---|---|---|
| 4-sleeve | 75% | −17.96% | −31.91% |
| 4-sleeve CONTROL | 80% | −15.75% | −29.32% |
| **5-sleeve** | 80% | **−12.79%** | **−24.72%** |

At identical currency exposure the fifth sleeve is still worth **4.60pp** of
drawdown, and the advantage GROWS when FX is frozen (+7.19pp vs +5.17pp). Real
diversification, not a currency artefact.

**Standing caveat (A1).** Freezing FX costs the whole book ~3.7pp of CAGR and
roughly doubles its drawdown. The five-sleeve structure is genuine, and the book is
*also* structurally short the rupee. Both are true and both must be disclosed.

### 5. The 1/N ensemble  *(P5.2, approved 2026-08-08)*

Removes config selection entirely, so PBO stops applying. Priced at −1.92pp CAGR
for a better drawdown (−20.95% vs −22.16%). Land it together with the above so the
book is reconstituted once, not twice.

---

## DO NOT DEPLOY — tested and failed

| Item | Verdict | Why |
|---|---|---|
| **Rebalance tolerance band** (R1) | ❌ FALSIFIED | Every band worse on CAGR *and* drawdown. Bands trigger because weights drifted, which happens most in volatile markets, so they rebalance repeatedly into a falling asset where the calendar does it once. |
| **Commodities sleeve** (R2) | ❌ FALSIFIED | CAGR −1.5pp, Sharpe 1.43→1.31, MaxDD −17.1%→−21.7%. |
| **Developed / emerging ex-US sleeves** (R2) | ❌ REJECTED PRE-TEST | 81% and 78% correlated with the existing US sleeve — the F7 silver mistake in a different costume. |
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

## What lands in January, final

| Change | Status |
|---|---|
| Four equal sleeves — equity 25 / gold 25 / US 25 / long gilt 25 | ✅ deploy |
| `LTGILTBEES` as the gilt instrument | ✅ verified 2026-08-09 — ₹30.07/share, ₹3.78cr/day median turnover. Cheap per share, so whole-share rounding is negligible here |
| Keep `deliv_chg` at 10% | ✅ deploy |
| 1/N ensemble | ✅ deploy |
| `n_hold` 20 → 60 | ❌ **declined** — see §1 |
| Fifth sleeve (US short-duration) | ⏸ shelved — see §4b |

Nothing above requires the owner to be present. The rebalance runs itself on the
scheduled date and reports the fills; that automation landed 2026-08-09.

---

## Pre-deployment checklist

- [ ] Re-run the full pipeline on `data/pit_cache_2007` with the new config
- [ ] Confirm published artifacts still agree (`TestPublishedArtifactsAgree`)
- [ ] Verify LTGILTBEES liquidity supports the intended position size
- [ ] Update `run_mark6.py` — and fix its stale comment crediting `n_hold=12`
- [ ] Record the reconstitution in `rebalance_events` with `off_cadence: false`
