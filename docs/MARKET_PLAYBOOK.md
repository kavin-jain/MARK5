# MARK5 Market Playbook — How Money Is Actually Made (Teachable)

> The operational, teachable layer of the second brain: how returns really work, how
> hedge funds make money, the strategy taxonomy, how to analyse a market, and the honest
> truth about *when to buy and sell*. Written so the system can **explain** it to a user.
> Companion to `docs/KNOWLEDGE_BASE.md` (theory) and `docs/RESEARCH_LOG.md` (our results).
>
> **Rule of this doc:** teach only what survives evidence. Where the popular answer is
> wrong, say so. Tags: ✅ we do it · 🔭 frontier · ❌ myth/doesn't survive · ⚠️ caveat.
> Curated 2026-06-07.

---

## 1. Where market returns actually come from (the decomposition)

Every rupee of return breaks into four buckets. Teach this first — it dissolves most
confusion:

```
Total return  =  BETA (market)  +  FACTOR premia  +  ALPHA (skill)  −  COSTS (tax+fees+slippage)
```

- **Beta** — just being in the market. The largest, cheapest bucket. Historically ~10–12%
  gross for Indian equity. You get it for free by holding. **[H]**
- **Factor premia** — systematic tilts that paid historically: value, momentum, low-vol,
  quality, size. Real but *lumpy and regime-dependent*; a few points/yr. **[H]**
- **Alpha** — genuine skill beyond the above. **Tiny and rare.** Hedge-fund studies:
  pre-fee return ≈ **beta 5.9% + alpha 3.0% + fees 3.7%**; and ~**2.3pp of that "alpha" is
  just the low-beta anomaly**, not skill. The median hedge fund hasn't beaten cash by 200bps
  long-term. **[H]**
- **Costs** — tax (LTCG 12.5% / STCG 20%), brokerage, STT, slippage. The only bucket you
  *fully control*. Minimising it is the most reliable "alpha" there is. **[H]**

**Teachable takeaway:** most "beating the market" stories are repackaged **beta + factor +
leverage**, sold as skill. Real alpha is ~3% and fragile. So the rational retail goal is:
**capture beta cheaply, tilt toward proven factors, crush costs, and don't pay for fake
alpha (including your own overtrading).** This is exactly what MARK6 does (✅).

---

## 2. How hedge funds actually work (and what we can/can't copy)

### Two archetypes
- **Single-manager / concentrated (Buffett-style):** few high-conviction bets, deep
  research, high IC, low breadth. Works *only* with genuine edge per name.
- **Multi-manager "pod shops" (Millennium, Citadel):** the dominant modern machine. **[H]**
  - Raise $10–20B → **lever to $50–100B** (gross leverage ~12x, net ~4.5x).
  - Allocate to **dozens of independent PMs ("pods")**, each specialised (one sector/strategy),
    each ~**market-neutral** (net exposure ±20%), each a small uncorrelated book.
  - **Centralised risk** cuts losing pods fast, scales winners, enforces factor-neutrality.
  - Result: ~13.6% (2024) at *low volatility* — from **diversification + discipline +
    leverage**, not one genius. This is **Grinold's √breadth** industrialised.

### What we can copy (retail, ₹500 Kite plan) ✅🔭
- **Breadth** — many positions / a whole-universe basket. ✅ (MARK6)
- **Specialisation as orthogonal signals** — several low-correlation factor sleeves. ✅/🔭
- **Centralised risk discipline** — hard limits, cut what's broken. ✅ (5%DD/2%daily)
- **Low net cost** — annual rebal, tax-aware. ✅

### What we CANNOT copy — and must not fake ❌
- **Leverage** (12x) — we don't have it and shouldn't want it long-only; it's where much of
  the pod return comes from, so *don't expect pod-shop returns without it*. Honest expectation
  setting.
- **Cheap shorting / true market-neutral** — not feasible on the ₹500 equity-delivery plan.
  So we can't neutralise beta; we *substitute* with factor diversification and risk limits.
- **First-look data / co-location** — institutional infrastructure edge. Out of reach.

**Teachable takeaway:** funds win with leverage + neutrality + breadth + discipline. We can
only bring **breadth + discipline + cost control**. That caps our realistic ceiling — and
that's fine; it's still better than 93% of retail.

---

## 3. Strategy taxonomy (and feasibility for us)

| Strategy | How it makes money | Feasible for us? |
|----------|-------------------|------------------|
| **Buy-and-hold index / EW quality basket** | Beta + small EW premium, near-zero cost | ✅ [H] — our profit engine (P2) |
| **Factor tilt (momentum/value/low-vol/quality)** | Harvest risk premia | ✅/🔭 [H] — MARK6 (P1); add low-vol/quality (F2/F3) |
| **Long/short equity, market-neutral** | Spread between longs & shorts, no beta | ❌ shorting infeasible cheaply |
| **Statistical arbitrage / pairs** | Mean-reversion of cointegrated spreads | 🔭 [M] — needs intraday + shorting; hard |
| **Global macro** | Top-down bets on rates/FX/commodities | ❌ out of scope |
| **Merger/event arb, index-rebalance** | Capture deal/flow-driven moves | 🔭 [L] — event dates free; crowded |
| **Intraday (ORB, momentum, VWAP)** | Short-term price patterns | ⚠️🔭 [M] — Kite makes it testable; 70–93% retail lose; cost-skeptic |
| **Trend-following / managed futures** | Ride sustained trends, cut losers | ⚠️ [M] — trend works on futures/indices, crash-prone on single stocks net of tax |
| **Carry / cash-yield** | Earn yield while idle | ✅ [L] — minor (cash_yield.py) |

---

## 4. How to analyse a market (top-down → bottom-up)

Teach a **funnel**, not a single indicator:

1. **Regime (macro/risk):** is breadth healthy, is vol elevated, rates/liquidity direction?
   Use regime to **scale risk**, *not* to time entries (timing it was KILLED, K2). ⚠️
2. **Cross-section (factor):** rank the universe on momentum/value/low-vol/quality. This is
   where the repeatable edge lives. ✅
3. **Name (fundamental):** quality screen — ROE, debt, cash-flow stability, moat,
   governance/promoter. Filters landmines more than it picks winners. 🔭
4. **Technical (timing/risk):** trend (200-DMA), support/resistance, volume/delivery — best
   used for *risk framing and execution*, weak as standalone alpha. ⚠️
5. **Microstructure (execution):** spread, liquidity, impact — only matters at size/intraday;
   Kite data lets us measure real slippage. 🔭

**Fundamental vs technical vs quant — what each is actually good for:**
- *Fundamental* = **what to own** (quality/value). Slow-moving, robust.
- *Technical* = **risk/where to act**, weak alone, prone to overfitting.
- *Quant* = **discipline at scale** (rank, size, rebalance, control bias). The glue.

---

## 5. The honest truth about "best time to buy / sell"

This is where most teaching lies. The evidence: **[H]**

- **Time IN the market beats timing the market.** Missing the **10 best days** over 20 years
  roughly **halves** returns; missing 20 cuts them to a third. The best and worst days
  **cluster together** — flee a crash and you miss the rebound. You cannot reliably step out
  and back in.
- Our own OOS agrees exactly: every timing/regime/stop overlay **subtracted** value net of
  tax (K2–K5). 

**So what *does* survive as buy/sell discipline?**
- **Buy:** systematically and on a **schedule** (rebalance dates / SIP / DCA), not on
  prediction. Add on rebalance when a name re-qualifies on factors. Valuation-awareness
  helps at *extremes* only (don't pay euphoric multiples). ✅
- **Hold:** through drawdowns — that's where the best days hide. Our basket holds through
  −40%. ✅
- **Sell:** for a **reason rule**, not a feeling — (a) scheduled rebalance, (b) factor
  de-qualification, (c) thesis broken / quality deterioration, (d) hard risk limit hit
  (5%DD/2%daily). **Not** tight price stops in a long-only tax book (KILLED, K4). ✅
- **Tax-time the sale:** prefer holding past 365 days (LTCG) when the rebalance is
  marginal — the buffer rule. ✅

**Teachable one-liner:** *"The best time to buy is on your schedule; the best time to sell is
when your rule (not your fear) says so. Time in the market, with discipline, beats timing it."*

---

## 6. Best practices — the process that separates winners from the 93%

1. **Risk first, return second.** Decide max loss before entry; never breach the hard limits.
2. **Diversify (breadth).** Many small uncorrelated positions > few big ones (Grinold).
3. **Size by risk, not conviction.** Inverse-vol / fractional-Kelly; never full-Kelly.
4. **Minimise turnover & tax.** Annual holding, buffers — the most reliable edge you control.
5. **Rules over discretion.** Pre-commit entry/exit/size; no revenge trades, no overrides.
6. **Measure honestly.** OOS, net of all costs, vs same-universe buy-and-hold. Deflate Sharpe.
7. **Survive tails.** Never lever long-only; hold quality through drawdowns (Taleb).
8. **Keep a log.** Every hypothesis + verdict (RESEARCH_LOG) — never repeat a dead end.
9. **Beware your own biases** more than the market's (hindsight, narrative, overfitting).
10. **Expectations:** target beta + a few points of factor premium, accept −30–40% DDs. No
    one reliably picks the year's best stock.

---

## 7. The teachable ruleset (what the system should tell a user)

> - You will not beat the market by predicting it. You beat *most people* by owning quality
>   broadly, cheaply, and patiently, and by not making the mistakes they make.
> - Returns are mostly **beta + factor premia**; real skill-alpha is ~3% and rare. Don't pay
>   (in fees *or* overtrading) for fake alpha.
> - **Time in the market > timing.** Buy on a schedule, hold through fear, sell on a rule.
> - **Costs and taxes are the enemy you control** — low turnover wins.
> - **Risk limits are sacred.** Diversify, size by volatility, never lever, survive the tail.
> - Every claim of edge must be **proven out-of-sample, net of tax**, or it isn't real.

This is the philosophy MARK5 embodies and should teach: not a crystal ball — a disciplined,
honest, low-cost, broadly-diversified machine that quietly beats the people who think they
have a crystal ball.
