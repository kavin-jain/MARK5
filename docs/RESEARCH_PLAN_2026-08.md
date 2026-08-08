# Research Plan — the road from 4.5/10 to 8/10

**Pre-registered 2026-08-08, before any experiment was run.** Written per the
Operating Mandate §1.3: every hypothesis carries a falsification condition fixed
in advance, so a null result cannot be reinterpreted into a success afterwards.

Results are appended to each entry as they land — **including failures**. If an
entry below has no result, it has not been run. If it has a FALSIFIED result, it
stays in this document; it is not deleted.

---

## The arithmetic being targeted

Composite grade uses the weights in the 2026-08-08 investment-committee review:
live 20% · edge 20% · risk-adjusted 15% · drawdown 15% · robustness 15% ·
process 10% · capacity 5%.

| Stage | Composite |
|---|---|
| Today | **4.50** |
| + 3 years live track record (running, no action available) | 5.90 |
| + bond sleeve (P1) | **7.10** |
| + breadth/transfer work succeeds (P2) | ~~7.45~~ **7.22** *(measured, not projected)* |
| 8.00 requires IR ≈ 0.85 | 8.00 |

**REVISED 2026-08-08 after Phase 2.** Breadth delivered +0.12 composite, not +0.35.
Sector-neutralisation delivered nothing. The honest projection with three years of
live data and the bond sleeve is **~7.2, not 7.45**. Reaching 8.0 now requires IC
itself to rise, which is the one thing 22 attempts have failed to move. P3.1 is
the remaining shot and its prior should be revised DOWN, not up.

Scoring frame is the full Fundamental Law (Clarke, de Silva & Thorley 2002):

```
IR = IC × √BR × TC
0.365 = 0.105 × √40 × 0.55      ← today (IC back-solved; matches the
0.85  = 0.105 × √134 × 0.70       knowledge base's independent 0.05-0.10)
```

**Every experiment below is ranked by its effect on IR or on the composite, never
on CAGR.** CAGR carries a ±9pp standard error on this sample and cannot rank
anything.

---

## PHASE 1 — Portfolio construction

Highest expected value. These are engineering, not alpha discovery: the mechanism
is understood in advance and the hit rate on this class of change has been far
better than on signal research.

### P1.1 — Long-duration bond sleeve *(evidence already collected 2026-08-08)*

**Hypothesis.** Adding a long-duration sovereign bond sleeve raises product Sharpe
and materially cuts max drawdown, because duration is convex to the rate cuts that
accompany deflationary equity crashes — the one behaviour gold does not reliably
provide.

**Falsification.** Sharpe does not improve full-period, OR the drawdown improvement
does not survive outside 2008.

**Evidence so far** (19.5y, proxy sleeves in INR, gross of tax):

| Book | CAGR | Sharpe | Vol | MaxDD | Calmar |
|---|---|---|---|---|---|
| Deployed 50/25/25 | +17.27% | 1.25 | 13.7% | −41.80% | 0.41 |
| +20% bonds | +16.31% | 1.41 | 11.4% | −27.33% | 0.60 |
| 35/20/20/25 "defensive" | +16.03% | **1.43** | 11.1% | **−22.66%** | **0.71** |

Correlation to equity: **−12% full period, −20% during 2007-09** — the hedge
strengthens exactly when needed. Gold only moves −2% → −11%.

**Regime split — the honest caveat:**

| Regime | Deployed | +25% bond |
|---|---|---|
| GFC 2007-12 | 9.44% / 0.73 / −41.8% | **13.17% / 1.21 / −22.7%** ✅ |
| Bond bull 2013-21 | 20.98% / 1.58 / −23.3% | 18.37% / **1.70** / **−16.1%** ✅ |
| Bond bear 2022-26 | **20.53% / 1.35** / −19.8% | 15.21% / 1.24 / −17.4% ❌ |

**Verdict: NOT falsified, but it is insurance with a premium (~1.2pp/yr CAGR), not
free alpha.** It loses in inflationary rate shocks. 2 of 3 regimes improve Sharpe.

**Still required before deployment (P1.1b):** the numbers above use TLT×USDINR as a
proxy. Re-run on instruments an Indian retail investor can actually buy —
LTGILTBEES (2018+), Bharat Bond ETFs (2019+) — and quantify the substitution error
against the proxy over the overlapping window. Indian and US curves are different
instruments and this is an unquantified risk.

### P1.2 — Allocation optimisation

**Hypothesis.** A weighting exists with MaxDD better than −25% and CAGR ≥ 15%.

**Falsification.** No allocation satisfies both.

**Evidence so far.** 33 allocations satisfy the constraint. Best by Sharpe:
eq 40 / gold 20 / US 15 / bond 25 → CAGR +15.80%, Sharpe 1.39, MaxDD −24.73%.
**Not falsified.** Final weights to be set after P1.1b.

### P1.3 — Silver as a 5th sleeve

**Result: FALSIFIED.** Silver correlates **79%** with gold full-period. It is a gold
proxy, not an independent sleeve. This confirms half of the earlier F7 kill —
though F7's stated reason (no pre-existence) was wrong, since silver has price
history to 2006. Right verdict, wrong reason.

---

## PHASE 2 — Breadth and transfer coefficient (the IR lever)

The formula says both terms move with one change: hold more names. This phase is
worth ~+0.35 composite and has never been tested on the 19-year sample.

### P2.1 — Breadth sweep

**Pre-registered hypothesis.** IR rises monotonically with `n_hold` up to roughly
60–80 names, then flattens as the marginal name adds more correlation than
information.

**Falsification.** IR is flat or falling in `n_hold` across the range. That would
mean breadth is already saturated and P5's concentration finding was right.

**Method.** `n_hold ∈ {12, 20, 40, 60, 80, 100}` over 2007-2026 on
`data/pit_cache_2007`. **Scored on IR versus equal-weight of the same universe**,
with CAGR and MaxDD reported alongside but not used to rank.

**Why this is worth doing despite P5.** P5 found n_hold 20→12 won 8/8 walk-forward
windows — the opposite direction. But P5 was selected in-sample on the 2016-2026
window, and PBO plus a negative IS/OOS rank correlation say selections made that
way do not generalise here.

**Cost acknowledged.** This is a grid search, and Mandate §5 states searching is not
free: each variant raises the multiple-testing penalty in DSR. Registering the
hypothesis in advance is the mitigation; the sweep is 6 points, not a fishing net.

**RESULT 2026-08-08 — SUPPORTED, but the gain is ~25% of what theory predicted.**

| n_hold | IR | t | CAGR | effN |
|---|---|---|---|---|
| 12 | 0.360 | 1.57 | +12.64% | 11.3 |
| 20 *(deployed)* | 0.365 | 1.59 | +12.05% | 18.6 |
| 40 | 0.379 | 1.66 | +11.40% | 36.5 |
| **60** | **0.433** | **1.90** | +11.49% | 54.0 |
| 80 | 0.428 | 1.87 | +11.07% | 70.1 |
| 100 | 0.415 | 1.81 | +10.79% | 84.2 |

IR peaks at n_hold=60 exactly where pre-registered (60-80), then declines.
**P5's concentrate-to-12 is falsified on 19 years — 12 is the WORST setting.**
That finding was a 2016-2026 window artifact, as PBO predicted.

But Grinold predicts IR x sqrt(3) = 0.63 for 20->60 holdings. Realised 0.433 —
about a quarter of the theoretical gain. `effN` shows why it is NOT a weighting
leak: 54 of 60 bets are effectively independent by weight. The bets are not
independent by *return*. Sixty Indian midcaps still ride one market factor, so
nominal breadth converts to real breadth at roughly 1:4 here.

t=1.90, p=0.029 at n=60 clears the conventional 95% bar for the first time, but
not the Harvey-Liu-Zhu t>3 hurdle for a new factor.

**Composite effect: edge 4 -> 4.6, worth +0.12, not the +0.35 projected.**

### P2.2 — Sector-neutral ranking

**Hypothesis.** Ranking within sector rather than across the whole market raises IR
at every `n_hold`, because it strips the common sector factor and converts nominal
breadth into independent breadth.

**Falsification.** IR does not improve, or improves only at low `n_hold`.

**Note.** The repo has a sector *cap* (30%) but no sector *neutralisation*. These
are different things and only the cap has been tested.

**KNOWN LIMITATION, recorded before the result.** `config/sector_map.json` holds
500 tickers across 19 sectors with >=5 members, but it was built from today's
listed universe and covers **zero of the 258 names that delisted inside the
window**. It carries the same survivorship shape the price cache was rebuilt to
remove. Unmapped names are therefore pooled and neutralised as one group so that
every score stays on a comparable scale — mixing within-sector z-scores against
raw scores would rank two different units, biased exactly along the survivorship
axis. Pooling fixes the scale; it does not recover the missing sector labels. **Any
P2.2 result is a LOWER BOUND on what full sector coverage would deliver, and must
be reported as such.** A first version of the implementation skipped the unmapped
pool entirely; that run was discarded before its numbers were read.

**RESULT 2026-08-08 — FALSIFIED, 0 of 6.** Sector-neutral ranking did not merely
fail to help; it HURT at every size, roughly halving IR:

| n_hold | raw IR | neutral IR | delta |
|---|---|---|---|
| 12 | 0.360 | 0.147 | −0.213 |
| 20 | 0.365 | 0.114 | −0.251 |
| 40 | 0.379 | 0.215 | −0.163 |
| 60 | 0.433 | 0.200 | −0.234 |
| 80 | 0.428 | 0.289 | −0.139 |
| 100 | 0.415 | 0.254 | −0.161 |

**Interpretation.** Forcing the book to buy the best name in a WEAK sector destroys
more information than sector-clustering costs. The concentration is not a bug
diluting breadth — **sector rotation is part of what momentum captures in India**,
and neutralising it removes signal rather than noise.

The survivorship hole above makes this a lower bound, so the true effect may be
less negative. It does not plausibly flip from −0.2 to positive.

**This was the author's high-confidence hypothesis and it was wrong. Recorded in
full rather than reframed.** Do not re-test sector neutralisation without new
evidence about WHY it should work here.

### P2.3 — Direct measurement of the transfer coefficient

**Hypothesis.** Current TC is below 0.6, and the largest single leak is
`base_weighting="inverse_vol"`, which sizes positions by volatility rather than by
conviction.

**Falsification.** Measured TC is already above 0.7, leaving no headroom.

**Method.** TC = cross-sectional correlation between realised active weights and
the active weights the composite score implies. Then sweep the leaks:
`tilt_strength`, `max_weight`, `buffer_mult`, sector cap.

---

## PHASE 3 — Signal quality (IC)

Low hit rate historically — 22 attempts, all failed. But the Group B failures share
a methodological defect, so the class deserves exactly one properly-run test.

### P3.1 — Orthogonalised signal re-test

**Hypothesis.** At least one signal killed in Group B (ownership flow, fundamental
quality, FIP, candlestick) carries material IC **orthogonal to momentum**, and was
killed because it was judged on raw IC while being correlated with a factor already
in the composite.

**Falsification.** Residual IC after regressing out the existing composite is below
0.03 for every candidate.

**Why this is not re-litigating a KILL.** Mandate §4 permits it explicitly: the
Group B verdicts were produced by a method that could not distinguish "no
information" from "no *incremental* information". This tests the distinction.

### P3.2 — Remove `deliv_chg`

**Hypothesis.** Dropping the provisional delivery factor costs less than 0.3pp of
IR and reduces the trial count that DSR penalises.

**Falsification.** IR falls materially without it.

---

## PHASE 4 — Tax

### P4.1 — LTCG-aware exit deferral

**Hypothesis.** Deferring the sale of profitable positions past the 365-day boundary
(unless badly deranked) raises net return by ~0.3–0.7pp/yr at no risk cost.

**Falsification.** Net return does not improve, because holding deranked names
longer costs more than the tax saved.

**Evidence motivating it.** 306 winning sells sit in the 6–10 month bucket carrying
₹20,03,544 of gains taxed at 20% rather than 12.5%. The 10–12 month bucket is
**completely empty** — the 126-bar cadence lands the first exit at ~182 days, in the
worst possible tax zone. Direct saving if deferred: **₹1,51,800**.

**Not a re-run of K3/K18/K20.** Those changed rebalance *frequency* or harvested
losses. This changes only the *exit condition for profitable lots*, leaving cadence
untouched.

---

## PHASE 5 — Measurement corrections

No strategy change; these fix statistics that currently describe the wrong object.

### P5.1 — Grade the product, not the sleeve

DSR is computed on the equity sleeve (Sharpe 0.63 over 19y) while the deployed book
is the multi-asset blend (Sharpe 1.43). Mandate §3 requires grading the product.
Trial dispersion must be recomputed at product level too: with half the book fixed
and passive, config search can only move half the risk, so the luck ceiling is
genuinely lower. **This is a correction, not a flattering re-basing — state the
reasoning in the output so a reader can check it.**

### P5.2 — Deploy the 1/N ensemble

Approved by the user for the scheduled rebalance (~2027-01-24). Removes config
selection entirely, so PBO stops applying. Priced at −1.92pp CAGR for a better
drawdown.

---

## Execution order

1. **P2.1 + P2.2 + P2.3** — the IR lever; largest uncertainty, so resolve first
2. **P1.1b + P1.2** — bond sleeve on real instruments, then final weights
3. **P4.1** — tax deferral
4. **P3.1 + P3.2** — signal work
5. **P5.1 + P5.2** — measurement, then deploy at the scheduled rebalance

Phase 2 runs first because it is the only item whose outcome is genuinely unknown
and on which the 8/10 target depends. If it fails, 8/10 is not reachable by this
route and the plan must be revised rather than pushed.

---

## Log

| Date | Item | Result |
|---|---|---|
| 2026-08-08 | P1.1 | NOT falsified — Sharpe 1.25→1.43, MaxDD −41.8%→−22.7%; fails in the 2022-26 rate shock |
| 2026-08-08 | P1.2 | NOT falsified — 33 feasible allocations; best eq40/gold20/us15/bond25 |
| 2026-08-08 | P1.3 | **FALSIFIED** — silver 79% correlated with gold, redundant |
| 2026-08-08 | P2.1 | SUPPORTED — IR 0.365→0.433 at n_hold=60; P5's n=12 falsified; gain only ~25% of theory |
| 2026-08-08 | P2.2 | **FALSIFIED 0/6** — sector-neutral cut IR roughly in half at every size |
