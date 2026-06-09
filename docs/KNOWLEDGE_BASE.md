# MARK5 Knowledge Base — The Canon, Distilled

> Curated knowledge from investing, quant, trading, and behavioural finance — books,
> papers, and practitioners — **mapped to MARK5**. This is not a book summary; every
> entry ends with what it means for *our* system. Companion to `docs/RESEARCH_LOG.md`
> (our own test results), `docs/MARKET_PLAYBOOK.md` (the teachable operational layer —
> how funds work, strategy taxonomy, how to analyse, when to buy/sell), and the
> Operating Mandate in `CLAUDE.md`.
>
> **Tags:** ✅ already doing · 🔭 open frontier · ❌ contradicted by our OOS ·
> ⚠️ applies-with-caveat. **Grade:** [H]/[M]/[L] evidence.
>
> Curated 2026-06-07.

---

## 0. The One Equation That Explains Everything We Found

**Grinold's Fundamental Law of Active Management:  `IR = IC × √Breadth`** **[H]**
(Information Ratio = forecasting skill × √(number of independent bets); refined by a
transfer coefficient ≤1 for implementation friction.)

This single law reconciles every result in `RESEARCH_LOG.md`:

- Our measured **IC ≈ 0.05–0.10** (price/factor signals vs forward returns). That is *real
  but tiny* — typical for public-data signals.
- With a tiny IC, the **only** way to a usable IR is **breadth** — many independent bets.
  That is *exactly* why single-stock / multibagger picking fails (breadth = 1 → IR≈0,
  K6 in the log) and why a **diversified factor basket works** (MARK6 = high breadth → the
  +3pp/yr survives, P1).
- It also reconciles the two greats as two valid corners of the same equation:
  - **Buffett/Munger:** very high IC (deep fundamental research, concentrated focus
    investing), low breadth. Few bets, held forever.
  - **Simons/Medallion:** very low IC (~just over 50% hit rate), astronomical breadth
    (thousands of bets/day). 
  - **Us (retail, public data):** *low IC and limited breadth* — the hardest corner.
    The honest play is to **maximise breadth** (whole-universe basket) and **stop trying to
    raise IC by overfitting** (which only adds variance). MARK6 is the law applied honestly.

**Takeaway:** stop hunting the genius single signal. Add *independent, low-correlation*
bets and let √breadth do the work. Raising breadth is reliable; raising IC is mostly
self-deception at retail.

---

## 1. Value / Fundamental investing (the Buffett–Graham school)

| Idea | Source | MARK5 verdict |
|------|--------|---------------|
| **Margin of safety**: buy below intrinsic value; Mr. Market is bipolar, exploit don't follow him | Graham, *Intelligent Investor* (Buffett's #1 rec); *Security Analysis* | 🔭 [H] — a *value tilt* (cheap on P/E·P/B·EV/EBITDA) is an untested factor here; needs fundamentals (frontier F3) |
| **Quality > cheapness**: "wonderful company at a fair price > fair company at a wonderful price"; durable moats, high ROE, low debt | Buffett (via Munger/Fisher) | 🔭 [M] — quality factor documented in India (cash-flow stability); add to MARK6 |
| **Focus investing**: concentrate in few high-conviction names | *The Warren Buffett Portfolio* | ⚠️ [M] — only works with **high IC**. We don't have it → contradicts naive concentration for us (see Grinold). |
| **Circle of competence / latticework of mental models / read 500 pages a day** | Munger | ✅ the Operating Mandate's relentless-reading rule is this idea |
| **Common Stocks & Uncommon Profits** (scuttlebutt, growth quality) | Phil Fisher (Buffett rec) | 🔭 scuttlebutt ≈ alt-data; F5 event/qualitative frontier |

**Net:** the value/quality *factors* are worth testing (F3), but Buffett-style
**concentration is wrong for us** — our edge must come from breadth, not conviction.

---

## 2. Modern quant / factor canon

| Idea | Source | MARK5 verdict |
|------|--------|---------------|
| **CAPM → multi-factor**: returns driven by systematic factors, not stock-picking | Sharpe; Fama–French 3/5-factor; Carhart (momentum) | ✅ MARK6 is multi-factor (momentum/low-vol/trend/stability) |
| **Factors are negatively correlated at inflection points** → blend them | Asness/AQR | ✅ P1; the diversification is the point |
| **Low-volatility anomaly**: low-vol stocks earn higher risk-adjusted returns (India: +11.4% vs +1.3% decile) | Haugen; Indian studies | 🔭 [H] anomaly / [M] net — dedicated low-vol tilt = F2 |
| **Momentum premium** but **crash-prone, negatively skewed** | Jegadeesh–Titman; Indian risk-managed momentum | ⚠️ [H] — gross edge real, **dies net of turnover→tax** (K3). Only survives with annual rebal + buffer |
| **Risk parity / equal risk contribution** (size by vol, not capital) | Dalio/Bridgewater | ✅ inverse-vol weighting in MARK6 (P3) |
| **Kelly criterion / optimal-f**: size by edge; over-betting ruins even a winning system | Thorp; Kelly | ⚠️ [H] — use *fractional* Kelly; full Kelly too volatile. Position sizing already vol-aware |
| **Markowitz MPT / efficient frontier; Black–Litterman** (blend views with equilibrium) | Markowitz; Black–Litterman | 🔭 [M] — BL is the principled way to add a factor *view* to a market-cap prior |

---

## 3. ML & backtest rigour (the most important section for us)

**López de Prado, *Advances in Financial Machine Learning*** — why most quant-ML fails: **[H]**
- **Backtest overfitting is near-certain**: every tweak is a hypothesis; sequential
  "improving" overfits even unintentionally. → *This is literally the V1→V10 treadmill in
  our log.* ✅ we now resist it (one-shot OOS, KILL list).
- **Standard k-fold CV leaks** in finance (trains on the future). Use **purged + embargoed
  CV** and **CPCV**. → ✅ already implemented (P4) — we were right.
- **Don't dump 100 indicators**; engineer few meaningful features; model groups not single
  names; bag predictions. → ✅ SHAP pruning, dedup; 🔭 "model the basket not the name."
- **A high Sharpe is meaningless without the probability it's luck** (deflated Sharpe,
  PBO). → 🔭 add Probability-of-Backtest-Overfitting / deflated-Sharpe checks to our harness.

**Triple-barrier labelling & meta-labelling** (López de Prado) → ✅ financial_engineer.py
already uses triple-barrier; 🔭 meta-labelling (a model that sizes/filters a primary
signal) is untested and is the *right* ML use here.

**Takeaway:** our leakage defences are textbook-correct. Our remaining ML sin was
*overfitting via iteration*, now controlled by the RESEARCH_LOG discipline.

---

## 4. Statistical arbitrage / the Simons–Thorp school

| Idea | Source | MARK5 verdict |
|------|--------|---------------|
| **Beat the market via thousands of tiny, mean-reverting bets** (55% hit rate × huge breadth) | Simons/Medallion; Thorp | ⚠️ [H] — the breadth lesson is gold; the *implementation* needs intraday/HFT data we partly get via Kite (F1) |
| **Pairs / cointegration mean-reversion** (enter at 2σ spread, hard stop) | DE Shaw; Gatev et al. | 🔭 [M] — untested here; needs liquid cointegrated pairs + intraday; crowded but real |
| **Market-neutral removes beta** (Medallion +82% in 2008) | — | ❌ for long-only us; we can't short cheaply on the ₹500 plan → substitute = factor diversification |

---

## 5. Traders & risk (Market Wizards school)

**Schwager, *Market Wizards*** — the universal lessons across ~all interviewed traders: **[H]**
- "Risk control is the most important thing." "The elements of good trading are: cut
  losses, cut losses, cut losses." Risk ≤2–2.5% per idea; **cut size during losing
  streaks**.
- Discipline > intelligence; never revenge-trade or override the system.
- Livermore: "sit tight" with winners; Tudor Jones: defense first; Dalio: diversify the
  uncorrelated.

**⚠️ Honest tension with our OOS:** "cut losses fast / use stops" is wisdom for
**leveraged, short-horizon, market-timing** trading. In a **long-only, tax-sensitive**
Indian basket, our data showed **stops *subtract* net value** (K4 — they trigger tax + miss
recoveries). *Both are true in their domain.* The transferable part for us is **position
sizing & total-risk discipline and emotional non-override**, *not* tight price stops.
Keep the hard portfolio limits (5% DD / 2% daily) — those are the Market-Wizards lesson
correctly applied.

**Taleb (*Fooled by Randomness / Black Swan / Antifragile*)** [H]: most track records are
luck; fat tails dominate; survive the tail. → ✅ why we hold through −40% DD rather than
get whipsawed out, and why we never lever.

---

## 6. Behavioural finance (where the edge *comes from*, if any)

| Idea | Source | MARK5 verdict |
|------|--------|---------------|
| **Loss aversion, overconfidence, recency, herding** create mispricings | Kahneman *Thinking Fast and Slow*; Thaler | 🔭 [H] — behavioural mispricing is the *theoretical source* of factor premia (momentum=underreaction, value=overreaction). Exploit via factors, not discretion |
| **The biases are in US too** — our own overfitting, hindsight, narrative ("HAL proves it works") | Montier *Behavioural Investing* | ✅ the RESEARCH_LOG exists precisely to defeat our own biases |
| **Disposition effect / retail loss data**: ~70–93% of Indian retail intraday traders lose | SEBI studies | ⚠️ [H] — sobering prior on F1 (intraday); be the house, not the gambler |

---

## 7. Market microstructure & execution

- **Kyle (1985) / market impact, adverse selection; VWAP/TWAP/implementation shortfall**
  [H] → 🔭 with Kite intraday data we can finally measure *real* slippage and test
  execution alpha; our backtests currently *assume* 0.10% slippage. Frontier F1.
- **Almgren–Chriss optimal execution** → 🔭 only relevant if we ever trade size/intraday.

---

## 8. The "wild" frontier (speculative, low evidence — labelled honestly)

| Idea | Grade | MARK5 verdict |
|------|-------|---------------|
| **Regime models (HMM/Markov switching)** to condition factor exposure | [M] | 🔭 our regime *router* as a return-timer was KILLED (K2); regime as a *risk-scaler* (not entry-timer) is the untested, more defensible variant |
| **Network / complexity / cross-correlation structure of NSE** (econophysics) | [L] | 🔭 arXiv work exists on NSE correlation eigenstructure; could improve diversification/risk, not alpha |
| **Reinforcement learning for execution/sizing** | [L] | ⚠️ glamorous, data-hungry, overfits badly (López de Prado caution); not now |
| **Alt-data** (satellite, credit-card, web-scrape, Google Trends, news/X sentiment) | [M] | 🔭 Two-Sigma school; `news_sentiment.py` stub exists. Real but crowding fast; needs a data source we don't pay for |
| **Foundation TS models (Kronos/Chronos/TimesFM)** for forecasting | [L] | 🟡 already wrapped as ≤10% component, no proven OOS edge (K9). Keep fail-open only |
| **Options/PCR, FII-flow, bulk/block deals as leading signals** | [M] | ⚠️ public ownership flow KILLED (K7, priced-in); options PCR / delivery-% untested |

---

## 9. India-specific canon

- **Tax is a first-class adversary** (LTCG 12.5% >1yr / STCG 20%): any US/global strategy
  must be re-derived for turnover→tax. This is *the* reason imported momentum strategies die
  here. [H]
- **Documented Indian anomalies**: low-vol (strong), momentum (crash-prone), quality
  (cash-flow stability), calendar (F&O-expiry / SIP-timing). [H]/[M]
- **Retail reality**: SEBI data — vast majority of intraday/F&O retail loses. Long-horizon,
  low-turnover, diversified is the structurally favoured retail posture. [H]

---

## 10. Curated reading list (the ONE transferable idea each)

**Foundational:** Graham *Intelligent Investor* (margin of safety) · Graham–Dodd *Security
Analysis* (intrinsic value) · Fisher *Common Stocks & Uncommon Profits* (quality/scuttlebutt)
· Cunningham *Essays of Warren Buffett* (owner mindset) · *Poor Charlie's Almanack* (mental
models, invert).
**Quant:** López de Prado *Advances in Financial ML* (**don't overfit; CPCV**) · Grinold–Kahn
*Active Portfolio Management* (**IR=IC√breadth**) · Ang *Asset Management* (factors) · Antti
Ilmanen *Expected Returns* (risk premia) · Narang *Inside the Black Box* (quant plumbing).
**Risk/trading:** Schwager *Market Wizards* (cut losses, size down on streaks) · Thorp *A Man
for All Markets* (Kelly, edge) · Taleb *Fooled by Randomness* (luck vs skill, tails).
**Behavioural:** Kahneman *Thinking, Fast and Slow* · Thaler *Misbehaving* · Montier
*Behavioural Investing* (your own biases first).
**Markets:** Mandelbrot *(Mis)behavior of Markets* (fat tails) · Lo *Adaptive Markets*
(efficiency is regime-dependent — why edges decay).

---

## 11. Synthesis — what the entire canon agrees on, for us

1. **Edge at retail with public data is small (IC≈0.05–0.10). Accept it.** The money is in
   **breadth × discipline × low cost/tax**, not in a secret signal. (Grinold)
2. **We already do the hard-but-correct things**: purged CPCV, point-in-time universe,
   inverse-vol factor basket, annual rebal. (López de Prado, AQR, Bridgewater) — keep them.
3. **Our biggest historical error was overfitting via iteration**, now fenced by the
   RESEARCH_LOG KILL list. (López de Prado, Montier)
4. **The genuinely under-exploited levers** for us: low-vol + quality factor tilts (F2/F3),
   honest intraday/microstructure tests via the Kite ₹500 plan (F1), and meta-labelling /
   risk-scaling regime use — *not* return-timing.
5. **Risk rules are sacred; price stops are not** in a long-only tax book. Survive tails,
   never lever, hold through drawdowns. (Taleb, Market Wizards correctly scoped)

> The canon does not contradict our OOS findings — it *predicts* them. That convergence is
> itself strong evidence we're now reasoning correctly.
