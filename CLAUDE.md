# MARK5 / MARK6 — Operating Mandate

This file governs all work in this repository. `docs/RESEARCH_LOG.md` cites it as
the authority for how research is conducted here; this is that authority.

---

## 0. Prime directive

**The deliverable is the truth about the system, not a flattering number.**

Every defect found in this repo's history was a measurement error that made
results look better: exit tax inside the return series, survivorship in the
universe, a benchmark scored on a shorter window than the strategy, corporate
actions missing from two warmup years. None was a strategy failure. All were
self-deception with a plausible cover story.

A change that lowers the headline and raises its truth is a **win**. Ship it and
say so plainly.

---

## 1. Before any research

1. **Read `docs/RESEARCH_LOG.md` first.** It holds 22 falsified approaches. Re-running
   a KILL wastes the user's money and your credibility.
2. **Check the failure taxonomy in §4.** If the idea belongs to Group A, it is dead
   on arrival — say so and stop, do not "test it once to be sure".
3. **State the hypothesis and the falsification condition BEFORE running anything.**
   Write down what result would make you abandon the idea. If no such result
   exists, it is not a hypothesis.
4. **Add a graded entry to the log after every test**, including — especially —
   failures. `[H]` high confidence, `[M]` medium, `[L]` low.

---

## 2. The scoring frame

Score on the **Fundamental Law of Active Management** (Grinold; Clarke, de Silva &
Thorley 2002):

```
IR = IC × √BR × TC
```

- **IC** — information coefficient; skill per bet. Currently ≈ 0.10. Hard to move.
- **BR** — breadth; independent bets per year = holdings × rebalances. Currently ≈ 40.
- **TC** — transfer coefficient; how much of the signal survives constraints.
  Long-only caps it near 0.5–0.6; weight caps, sector caps, the ranking buffer and
  inverse-vol sizing all leak further.

**Consequences that must shape every proposal:**

- **Rank ideas by their effect on IR, never on CAGR.** CAGR is a point estimate on
  one path with a ±9pp error bar; IR is the estimate of skill.
- **Breadth and transfer are engineering. IC is research.** 22 attempts to raise IC
  have failed. Portfolio construction has a far better hit rate — prefer it.
- **Breadth only counts if bets are independent.** Correlated names inflate nominal
  BR without raising real BR. Sector-neutralisation is what converts one into the
  other.

---

## 3. Measurement standards (non-negotiable)

| Rule | Why |
|---|---|
| Point-in-time universe, delisted names held to their last print | Survivorship inflates ~2–5pp/yr |
| Net of Indian tax: STCG 20%, LTCG 12.5%, FY loss netting, FIFO lots | Gross results are fiction here |
| Costs set **above** real broker rates | Optimism must never enter through costs |
| Terminal liquidation tax excluded from the return series | It is a cost, not a market return (`metrics_after_exit_tax`) |
| Benchmark must span the **same window** as the strategy | A benchmark that misses 2008 is not a benchmark |
| Any cached series must be checked for coverage before use | BUG2, BUG3, and the CA-cache bug were all this |
| Walk-forward across rolling windows, not one full-sample number | Full-sample fit is not evidence |
| Report DSR **and** PBO, calibrated (see §5) | Raw values are uninterpretable |

**Grade the product, not a sleeve.** The deployed book is the multi-asset blend.
Statistics computed on the equity sleeve alone describe something nobody owns.

---

## 4. Failure taxonomy — why 22 things died

Three mechanisms explain 19 of them. Classify every new idea before testing.

**Group A — reduces exposure after a loss. DEAD ON ARRIVAL.**
K2 regime overlays · K4 stops · K5 circuit breakers · K14 vol-targeting ·
K21 fast rebalance · K22 asymmetric exits.

> Indian equity has positive drift and V-shaped recoveries, so forward expected
> return is *highest* precisely when these rules cut exposure. They are structurally
> guaranteed to sell low and rebuy high. **Never test this family again.** If risk
> must come down, take it out of the allocation, not out of timing.

**Group B — weak signal correlated with what we already own.**
K1 ML · K6 multibagger · K7 ownership · K9 candlestick/foundation · K11 promoter
level · K12 promoter Δ · K15 quality tilt · K19 FIP.

> All had IC ≈ 0.02–0.07. But each was judged on **raw** IC, never on IC
> *orthogonal to momentum*. A weak signal correlated with a strong one adds no
> information and dilutes the strong one. **The method was wrong, so some of these
> verdicts may be wrong.** Any future signal must be tested on its residual after
> the existing composite is regressed out. Raw IC is not evidence.

**Group C — hard cost wall.**
K3 · K13 leverage · K16 naive risk parity · K18 TLH · K20 sleeve frequency.

> Indian financing ≈ the asset's own return, and STCG at 20% eats turnover. These
> are real constraints, not failures of imagination. Do not route around them with
> cleverness; design within them.

---

## 5. Statistics: calibrate before interpreting

- **PBO has a null of ~50%, not 0%.** When candidate configs are statistically
  indistinguishable, ranking them is ranking noise and PBO goes to ~50% with no
  overfitting present. `scripts/pbo_calibration.py` establishes the bands:
  real edge ≈ 1%, indistinguishable ≈ 50% (28–78%), true overfitting ≈ 99%.
  **Never "improve" a PBO inside the null band — there is nothing there to fix.**
- **A Sharpe carries a standard error.** `SE ≈ √((1+SR²/2)/T)` — ±0.39 at 10 years.
  Two configs differing by less than ~1.0 in Sharpe are the same config.
- **A CAGR carries a standard error.** `SE = vol/√years`. Publish the interval or do
  not publish the point.
- **The t-stat hurdle for a new factor is ~3.0, not 2.0** (Harvey, Liu & Zhu 2016),
  because the profession has already tried thousands. At t < 2 you have nothing.
- **Every variant tested raises the multiple-testing penalty in DSR.** Searching is
  not free. Pre-register; do not grid-search for a headline.

---

## 6. Deployment discipline

- The live paper book is an **append-only integrity record**. Never rewrite it,
  never stamp a mid-session price into it, never rebalance off-cadence. Off-cadence
  changes must be disclosed in the published feed.
- Strategy changes land at the **scheduled rebalance**, not when they are finished.
- Config selection does not generalise here (PBO in the null band, IS/OOS rank
  correlation negative). Prefer a **fixed, economically-motivated config or the 1/N
  ensemble** over any in-sample winner.
- Public artifacts must agree with each other. If the page says one thing and a
  report it links says another, the page has lost the only thing it is selling.
  `TestPublishedArtifactsAgree` enforces this — keep it passing.

---

## 7. Engineering

- Tests are the contract. Never weaken a test to make a change pass; if a test
  fails, first assume the test is right.
- Fix at the root: one guard in the shared function, not a guard in every caller.
- Never trust a fetched series without checking length **and** date coverage.
- Long-running research runs in the background with the window and universe
  recorded **in the output**, never hardcoded in prose.
- Never commit secrets, keys, or `.env`. Never commit the data caches.

---

## 8. Reporting to the user

The user is not a trader. They are the owner and need to defend this system to
other people.

- **Lead with the conclusion**, then the evidence.
- **Define every term on first use.** IR, drawdown, PBO — assume nothing.
- **Volunteer the weakness before it is asked for.** The −41.8% drawdown and the
  regime-dependence are part of the honest answer, not footnotes.
- **Never state a return without its error bar or its worst case.**
- Distinguish **paper gains from realised gains** every time money is discussed.
- If a previous statement turns out to be wrong, correct it in one plain sentence
  and continue. No hedging, no ceremony.
- **Never give personalised investment advice.** Report what the evidence supports
  and what it does not; the allocation decision is the user's alone.
