# MARK5 Research Log — The Second Brain

> **Purpose.** This is the project's persistent, curated memory: only *sorted and
> important* findings — what was tested, what the out-of-sample (OOS) verdict was,
> and why. Before starting any research thread, **read this file**. Do not re-run a
> KILLED approach unless new data or a new mechanism justifies it (state why).
>
> **Maintained per the Operating Mandate in `CLAUDE.md`.** Every entry carries an
> evidence grade and a verdict. Honesty over ambition — a truthful KILL is worth more
> than a flattering KEEP.
>
> **Companion:** `docs/KNOWLEDGE_BASE.md` — the distilled canon (investing, quant,
> traders, behavioural, models) mapped to MARK5. Key unifier: **Grinold's Law
> `IR = IC × √Breadth`** explains why our basket (P1) works and single-stock picking
> (K6) can't — at our IC≈0.05–0.10, edge must come from breadth, not conviction.
>
> **Verdict legend:** ✅ KEEP (validated OOS) · ❌ KILL (falsified OOS) ·
> 🟡 INCONCLUSIVE · 🔭 OPEN (untested frontier)
> **Evidence legend:** **[H]** high (our OOS data / strong literature) ·
> **[M]** medium · **[L]** low (single study / theory only)
>
> Last curated: 2026-06-07.

---

## 0. The Project in one paragraph

MARK5 is an ML-driven, **PAPER-mode** trading system for NSE Indian equities (Midcap150
/ NIFTY100 universe, ₹5cr paper capital). Hard limits: 5% max drawdown, 2% max daily
loss. After ~10 strategy iterations (V1–V10), an ML rebuild (V2), momentum, candlestick,
foundation-model and swing (markrapid) signals, and a factor portfolio (MARK6), the
evidence converged on one uncomfortable result — see §1.

---

## 1. THE CENTRAL ISSUE (the thing that was "cutting our edge")

**There is no active overlay in this system that beats same-universe buy-and-hold,
net of Indian tax and costs.** **[H]**

- The "edge" in every profitable backtest traced to **holding good stocks**, not to
  timing/selection skill. The overlays (ML gating, momentum rebalancing, stops, regime
  switches, circuit breakers) **subtracted** value once measured OOS and net of tax.
- The original ~18–24% "returns" were **cherry-picked** (HAL + TRENT buy-and-hold in
  disguise). On the full 32-ticker universe OOS, median ML net CAGR ≈ **0%**; 0/32 beat
  +20%. See [[v2-ml-system]].
- Root causes that cut the edge:
  1. **Tax drag.** Turnover converts LTCG (12.5%) into STCG (20%); 400% turnover can
     erase a +5pp gross momentum edge entirely. **[H]**
  2. **Transaction + slippage** (0.29% round-trip + 0.10% slippage) compound per rebalance.
  3. **Look-ahead / survivorship** inflated early results until purged CPCV + point-in-time
     universe + survivorship validation removed them.
  4. **Overfitting treadmill** — each new strategy version tuned itself onto the test set;
     gains evaporated OOS (the V1→V10 arc is the proof, see [[v10-production-system]]).

**Consequence:** the bar "beat HAL's +600%" is a **category error**. Medallion ~39%/yr,
Buffett ~20%/yr — both far below the best single stock in any year. Alpha = a *repeatable
edge across many bets*, not picking the one winner. See [[predictability-study]].

---

## 2. KILL LIST — tested and falsified OOS (do not re-litigate)

| # | Approach | Verdict | Evidence | Why it died |
|---|----------|---------|----------|-------------|
| K1 | **ML probability-gated trading** (XGB/LGBM/RF/CatBoost ensemble, hurdle, entropy gate) | ❌ KILL | [H] | Negative alpha OOS across full universe; ML signal IC tiny; gating removed good holding days. [[v2-ml-system]] |
| K2 | **Market-timing / regime overlays** (regime router, VIX gate, breadth gate) | ❌ KILL | [H] | Destroy value net of tax; whipsaw + missed recovery days. [[honest-oos-verdict]] |
| K3 | **Momentum rebalancing overlay** (top-N by confidence, periodic rotation) | ❌ KILL | [H] | Gross edge real (~+5pp) but dies net of turnover→STCG tax. Matches literature: momentum crash-prone, negatively skewed. **Quantified 2026-06-08** (`holding_period_sweep.py`, `reports/HOLDING_PERIOD_ANALYSIS.md`): 1-day rebal → +4.0% net (₹5L→₹7.5L) vs 1-yr → +16.8% (₹25L); even GROSS is +4.2% at 1d (costs alone bleed ~13pp/yr at 3834% turnover). Net return climbs monotonically with holding period, plateaus ~6mo-1yr. Long hold = FEATURE. |
| K4 | **Stops / trailing / ratchet as return enhancers** | ❌ KILL | [H] | Reduce DD but cut more upside than downside net; B&H through DD won. |
| K5 | **Circuit breakers as alpha** | ❌ KILL | [H] | Help DD optics, lower CAGR; CB deadlock bug found & fixed but still net-negative vs B&H. [[v6-production-system]][[v7-production-system]] |
| K6 | **Ex-ante multibagger prediction from price/volume/factors** | ❌ KILL | [H] | BEL ranked 7th-percentile right before +1214%; IC 0.05–0.10; winners scatter, never concentrate in top decile. [[predictability-study]] |
| K7 | **Public ownership-flow signal** (institutional FII+DII accumulation) | ❌ KILL | **[H]** | **Confirmed on FULL DEEP data (2026-06-07): 198 stocks, ~32q to 2018, real disclosure dates.** Δ-Institutions IC(1y)=−0.025; FII-buyers vs sellers −0.6pp (no edge). Winner study: corr(pre-run Δinst, run size)=−0.204 — top winners ALL PSU/rail (IRFC/BSE/HUDCO/RVNL/SAIL/NBCC/IRCON) with institutions FLAT/SELLING before; institutions CHASE not lead. Priced in by disclosure. Paid Trendlyne would NOT have helped. See §4 I1. |
| K8 | **Swing-trade tier as WR/return fix** (markrapid, RSI reversion) | 🟡→❌ | [M] | 47% WR, +10.9% compounded on tiny size; does not scale to beat B&H; WR-math in `HEDGE_FUND_RESEARCH.md` was pre-OOS aspiration. [[markrapid-system]] |
| K9 | **Candlestick / foundation-model (Kronos/Chronos) ranking components** | 🟡 | [M] | Integrated as ≤10% weights; improved DD optics in-sample, no proven OOS return edge; kept only as fail-open, zero-lookahead. [[candlestick-pattern-system]][[foundation-signal-system]] |
| K10 | **Heavier low-vol tilt (F2)** in MARK6 blend | ❌ KILL | **[H]** | `factor_research.py` (2026-06-07): low_vol .45/.60 weights → recent −3.4/−4.7pp, walk-fwd avgΔ −1.0/−1.8pp vs baseline. Low-vol anomaly is real but **over-tilting cuts net CAGR** (gives up too much momentum/growth). *Useful side-effect:* it MEANS-reduces MaxDD (recent −18.4% vs −25.3%) — a risk knob, not a return edge. |
| K11 | **Quality proxy = promoter-holding level (F3)** added to blend | ❌ KILL | **[H]** | `factor_research.py`: recent −1.7pp, walk-fwd avgΔ −0.1pp. No edge (promoter level ≈ neutral once momentum/low-vol/stability already in). True fundamental quality (ROE/debt) still untested — needs historical financials we don't have. |
| K12 | **Promoter-Δ / institutional-Δ as factor sleeves (F6)** | ❌ KILL | **[H]** | `factor_research.py`: F6 recent −0.3pp / walk-fwd +0.4pp (within noise); inst-Δ control +0.4pp walk-fwd / −0.0 recent (behaves as no-edge, validating the harness). The weak +IC (~+0.04) does NOT convert to a robust net edge on top of the existing blend. |
| K13 | **Leverage (and hedged leverage) to reach 20%** | ❌ KILL | **[H]** | `leverage_hedge_test.py` (2026-06-08): at Indian financing ~14% (≈ the asset's return), leverage LOWERS net CAGR and multiplies drawdown — L=2 → +8.3% CAGR / −67% DD vs unlevered +12.3% / −34%; Sharpe 0.78→0.41. Hedging removes beta → kills return (no alpha to lever). NO config reached 20%. The bottleneck to fund-like returns = can't cheaply hedge → can't safely lever; unsolvable at retail in India. |

> **Note on `docs/HEDGE_FUND_RESEARCH.md`:** that doc (2026-05-23) derived V4 from
> Renaissance/AQR/DE-Shaw/Bridgewater/Two-Sigma principles. Its *projected* WR/DD/return
> numbers were **pre-OOS aspirations and were not realised** — treat it as design
> inspiration, not validated results. The one principle that survived is "many small
> uncorrelated bets," realised as the factor basket (§3), not as the swing/regime stack.

---

## 3. KEEP LIST — validated OOS (the honest edge)

| # | What | Verdict | Evidence | Result |
|---|------|---------|----------|--------|
| P1 | **MARK6 smart-beta factor portfolio** — long-only, multi-factor (momentum/low-vol/trend/stability), inverse-vol weighted, **annual** rebalance (LTCG), buffer to cut turnover, sector/weight caps, **no timing overlay** | ✅ KEEP | [H] | +13.4% net vs Nifty +10.4% full-cycle (**+3pp/yr**), Sharpe 0.86, DD −34%. Beats Nifty 3/8 rolling windows (regime-dependent). `core/portfolio/`, `scripts/run_mark6.py`. [[mark6-smart-beta-system]] |
| P2 | **Equal-weight buy-and-hold of a quality mid/large basket**, annual rebal, held through −40% DD | ✅ KEEP | [H] | Beats cap-weighted NIFTY by ~9pp (midcap EW-B&H +23.5%/+17.1% net; survivorship-caveated). The real profit engine. [[honest-oos-verdict]] |
| P3 | **Annual rebalance + ranking buffer + inverse-vol sizing** (the tax/turnover discipline) | ✅ KEEP | [H] | Recovers a robust ~+0.5pp net where naive momentum lost; the mechanism that lets any tilt survive tax. |
| P4 | **Leakage defences** — purged CPCV (5 splits, 2 test, 20-bar embargo), point-in-time universe, survivorship validation, feature dedup | ✅ KEEP | [H] | Not a return source — the *truth* source. Every result above cleared this bar. [[system-audit-2026-05-25]] |
| P5 | **Concentrate the factor book: n_hold 20→12, tilt_strength 0.5→1.5** | ✅ KEEP | **[H]** | `risk_dial_test.py` + `validate_concentrated.py` (2026-06-08): beats old n_hold=20 in **8/8** rolling 3-yr walk-forward windows, **+2.3pp avg net**, full-period ~13%→~16% net, Sharpe 0.86→0.93, DD −34%→−39% (modest). The 20-name book was over-diversified, diluting the signal. Now the production default in `run_mark6.py`. Momentum-heavy variant (7/8, +2.6pp) is a higher-return/higher-DD option. |
| P8 | **3-sleeve global diversification: 70% equity / 15% gold / 15% US-Nasdaq100 (MON100)** | ✅ KEEP | **[H]** | `multiasset_v2_test.py` (2026-06-08): the three sleeves are mutually ~uncorrelated (eq-gold −0.00, eq-US +0.04, gold-US 0.00). Adding 15% US lifts **Sharpe 0.88→0.99 (full) / 1.01 (walk-forward)**, CAGR +15.8→**+17.3%**, MaxDD −28%, alpha vs Nifty **+9.7%/yr**, ₹5L→₹26.4L over 10.4y. Hits the Sharpe~1.0 target. The diversification (Sharpe/DD) benefit is robust; US's 26%/yr return is regime-dependent (hence modest 15%). Equity book UNCHANGED — pure allocation improvement. Now the deployed default (`generate_portfolio.py`, `institutional_report.py`). Markowitz "only free lunch" + Bridgewater risk-parity, confirmed. |
| P7 | **Multi-asset: add ~20% GOLD (GOLDBEES) to the equity book** | ✅ KEEP (→ superseded by P8) | **[H]** | `multiasset_voltarget_test.py` (2026-06-08): eq80/gold20 is a Pareto win full-period — CAGR +13.6→+15.0%, Sharpe 0.76→0.84, **MaxDD −34.6→−28.0%**, Calmar 0.39→0.54; walk-forward Calmar 0.68→0.79, worst DD −40→−32%. Mechanism robust: **equity-gold daily corr = −0.001** (zero) → diversification benefit holds even if gold's 16%/yr INR run doesn't repeat. eq70/gold30 = more DD protection. The honest "better portfolio" win — attacks the −35% drawdown, the system's worst feature. |
| K14 | **Portfolio volatility-targeting** (scale equity to target vol) | ❌ KILL | **[H]** | Same test: voltarget cut CAGR to +11.0% to shave DD to −27% — gives up too much upside (de-risks into recoveries, the timing-failure pattern again). Calmar NOT improved. Gold diversification dominates it. |
| K15 | **Fundamental quality as a TILT (F3)** — ROCE/low-debt/FCF/stability from indianapi.in | ❌ KILL | **[H]** | `fundamentals_quality_test.py` (2026-06-08, real 12-yr fundamentals, 98 tickers): all quality-weighted configs fail the bar — walk-forward avgΔ −1 to −4.5pp, beats ≤5/8. Regime-dependent: HELPS 2016-21 flight-to-quality (q_light holdout +2.7pp, Sharpe→1.12), HURTS 2022-26 PSU/junk rally (−1.5pp). A 66-ticker partial run falsely flagged a KEEP candidate that **evaporated as data grew** (textbook small-sample lesson). Mild DD help only. Quality-as-SCREEN (exclude junk) untested — needs full fundamentals coverage. |
| BUG | **ETF contamination of equity universe (FIXED)** | ✅ FIXED | [H] | GOLDBEES/LIQUIDBEES cached for the multi-asset test leaked into `discover_tickers()` → LIQUIDBEES (≈cash, lowest vol) got inverse-vol-OVERWEIGHTED, dragging the equity book (visible: F3 baseline 14.8% vs clean 16.2%). Fixed: STRUCTURAL_EXCLUDE + `_is_etf()` filter in `universe.py`. Re-verified all numbers clean. Caught by `generate_portfolio.py` showing LIQUIDBEES as the #1 holding — deploy-time sanity check working. |
| BUG2 | **Data-staleness contamination (FIXED 2026-06-10)** | ✅ FIXED | **[H]** | 137/345 cache files were frozen at 2026-04-01 (40% of universe) while the rest reached 2026-06-05 → stale names silently lost price-eligibility and dropped from the point-in-time universe, corrupting the 2025–26 recent window (the "system only makes 10%" perception). Fixed: `refetch_all.py` re-pulled all to uniform 2026-06-09 + new **freshness guard** in `DataPanel` (warns/raises if any ticker ends >7d before panel end). Only GSPL unfetchable (dead symbol). Clean full-period number barely moved (+16.1% factor) → the bug hurt recent-window optics, not the headline. |
| P9 | **Momentum-heavy equity weights** (mom .45 / trend .25 / low_vol .15 / stability .15, n=12) | ✅ KEEP | **[H]** | `momentum_quality_screen_test.py` (2026-06-10, clean data): +1.4pp avg net vs baseline blend across rolling 3-yr walk-forward, **beats 6/8 windows**; full-period +16.3% vs +16.1%, holdout16-21 +20.0% vs +16.7%. Higher DD (-37.9% equity-only) bought back by the sleeve wrapper. Now the production equity book in `run_mark6.py` + `institutional_report.py`. (Confirms the P5 note that the momentum variant is the higher-return option.) |
| P10 | **Allocation eq50/gold25/US25** (was 70/15/15 → 60/20/20 → 50/25/25) | ✅ KEEP | **[H]** | `multisleeve_riskparity_test.py` + institutional report (2026-06-10, FULL-period real data): **+18.8% net CAGR, Sharpe 0.89, MaxDD −26.7%, Calmar 0.70, alpha +12.8%/yr, beta 0.60**, ₹5cr→₹30.1cr/10.4y. Robust Pareto win over 60/20/20 (+0.8pp CAGR, −2.2pp DD, +0.07 Calmar) across walk-forward (prior test: avgSharpe 0.96 / worstDD −24.8%). The honest deployed default. NOT a regime cherry-pick (spans 2016-21 when US/gold were cooler). |
| K16 | **Naive sleeve risk-parity** (inverse-trailing-vol sleeve weights) | ❌ KILL | **[H]** | `multisleeve_riskparity_test.py` (2026-06-10): inverse-vol sleeve weighting → +19.3% CAGR but **MaxDD −56%, Sharpe 0.56** (independently corroborates the earlier `risk_parity_3` −54% finding). Inverse-vol over-levers into the highest-Sharpe sleeve mix and removes the DD protection that fixed weights give. Fixed allocation dominates. |
| K17 | **Quality-as-SCREEN** (exclude bottom-30% fundamental quality before ranking) | 🟡 UNTESTED | [L] | Hook built (`Backtester(screen=...)`, `make_quality_screen`) + `momentum_quality_screen_test.py` — but `data/cache/fundamentals/` is EMPTY (0 coverage) so the screen never fired (0.0pp effect = no-op, not a result). Distinct from quality-as-TILT (K15 KILL). Needs `fetch_fundamentals.py` with an indianapi.in key to actually test. Open. |
| F7 | **Multi-sleeve expansion (+silver +long-gilt), 5 sleeves** | ❌ KILL (overfit) | **[H]** | `multisleeve_riskparity_test.py` (2026-06-10): a 5-sleeve fixed-eq50 blend hits ALL targets in 2022-26 (**+20.4% CAGR, Sharpe 1.18, MaxDD −17.9%, Calmar 1.14**) — BUT silver (SILVERBEES data from 2022) + gilt (LTGILTBEES 2018) have NO pre-existence, so this is **un-backtestable full-period** and rides a once-in-a-decade silver run (+40% 2024 / +24% H1-2025) that external research shows became MORE equity-correlated in 2025. **3-advisor council unanimous: window-selection overfitting, "a disclosure not a deliverable."** Rejected per the user's "no overfitting" rule. Silver/gilt may be added later as small *forward* structural diversifiers (not because of the 20.4% number). |
| P11 | **Fiscal-year tax NETTING in the backtest engine** (2026-06-11) | ✅ KEEP | **[H]** | `efficiency_research.py` + promoted to `backtest.py` (`fy_netting=True` default). The old model taxed every winning sell but gave NO credit for losses — actual Indian law nets losses against gains within the FY (STCL vs STCG then LTCG; LTCL vs LTCG; 8-yr carry-forward), settled each April (positions sold pro-rata if cash short — no implicit leverage). Same trades: +0.5pp full-period, +0.47pp avg walk-forward, 7/8 windows. Not a strategy — a TRUTH fix. Tax paid 0.80→0.60 NAV-units. |
| P12 | **Semi-annual equity rebalance (rebal_bars 252→126)** under honest netting | ✅ KEEP | **[H]** | `efficiency_research.py` (2026-06-11): K3's "longer is monotonically better" was an ARTIFACT of the no-loss-credit tax model, which over-penalised turnover. Under P11 netting: equity sleeve +16.8→**+20.2%** full-period (Sharpe 0.82→0.94), **+2.84pp avg walk-forward, beats 7/8 windows** (worst −7.0pp = 2019-21 COVID-V where annual rode the recovery). FULL SYSTEM (50/25/25 wrapper): 19.0→**+20.8% net CAGR, Sharpe 0.96, MaxDD −26.6%, Calmar 0.78**; wrapper-level walk-forward +1.21pp avg, 6/8. Mechanism: momentum decays at the 6-12mo horizon (matches literature); netting removes the tax wall. Quarterly (63d) has higher mean (+3.98pp) but fatter tails (worst −10.6pp) and 480% turnover — rejected for robustness. Ex-ante hypothesis, single structural parameter, not weight-tuning. NOW THE DEPLOYED DEFAULT. |
| K18 | **Tax-loss harvesting** (monthly check, sell loser & rebuy — India has no wash-sale rule) | ❌ KILL | **[H]** | `efficiency_research.py` (2026-06-11): −0.40pp avg walk-forward, 0/8 windows beat netting-only. Mechanism understood: rebuying RESETS the holding clock → future LTCG (12.5%) converts to STCG (20%) on the recovery, plus ~0.5% churn cost per harvest; the annual rebalance already books natural losses that the FY netting absorbs. TLH works in the US (no clock reset on the replacement-security workaround); in India it's net-negative for this book. |
| K19 | **Frog-in-the-pan momentum quality (FIP, Da-Gurun-Warachka)** as 10% component | ❌ KILL | **[H]** | `efficiency_research.py` (2026-06-11): full-period +0.9pp looked promising but walk-forward says noise — +0.13pp avg, 4/8 windows, worst −2.3pp. US evidence is UP-market-conditional; does not replicate as a robust component here. |
| K20 | **Sleeve-rebalance frequency** (wrapper 50/25/25 quarterly/semi-annual vs annual) | ❌ KILL | **[H]** | `efficiency_research.py` (2026-06-11): 18.8/19.1/19.0% — noise-level spread; faster sleeve rebalance adds STCG drag for no rebalancing-premium gain at these correlations. Annual sleeves stay. |
| BUG3 | **Nifty benchmark silently overwritten with partial data (FIXED 2026-06-11)** | ✅ FIXED | **[H]** | `refetch_all.py`'s ^NSEI refresh saved a PARTIAL yfinance response (2007–2017 only) over the good benchmark file → every vs-Nifty figure computed after that was garbage (Nifty showed +1.5% CAGR). Re-fetched full history (2007–2026-06-09); added a guard: never overwrite unless >4000 rows AND reaches the requested END year. Lesson = same as BUG2: NEVER trust a fetch without a recency+length check. |
| P6 | **BUGFIX: `backtest.py` warmup_skip 1→0** (full-codebase audit, 2026-06-08) | ✅ KEEP | **[H]** | The backtester left the book in CASH for the first ~252 bars (one year) of EVERY window because `warmup_skip=1` skipped the first scheduled rebalance. Factors are valid at the window start (built from pre-window history — no look-ahead), so this was pure drag. Impact: ~0 on full period (1 lost yr in 10) but **distorted the walk-forward badly** (a third of each 3-yr window in cash) — corrected walk-forward avg ~13%→~20%, **beats Nifty 7/8 (was 3/8)**, and made the vs-Nifty comparison fair (Nifty is day-1 invested). Verified legit (full-period stayed +16.0%, not inflated → not look-ahead). 22/22 tests pass. **This is the "core-file bug" the prior walk-forward pessimism partly rested on.** |

**Bottom line:** the deliverable is a **portfolio, not a strategy** — an equal-weight /
inverse-vol quality basket, annually rebalanced, held through drawdowns, beating
cap-weighted NIFTY by a few points a year. That is what alpha actually looks like at
retail with public data.

---

## 4. IN-PROGRESS / INCONCLUSIVE

- **I1 — Deep ownership-accumulation re-test.** ✅ **[COMPLETE 2026-06-07 → verdict K7]**
  Free NSE XBRL shareholding archive (~32 quarters, back to mid-2018, **real disclosure
  dates** = zero look-ahead) via `scripts/fetch_shareholding_nse.py`, re-running
  `scripts/ownership_signal_study.py` on data that **covers the 2019–2024 HAL/BEL/TRENT runs**.
  - **⚠️ Data-quality bug found & fixed (2026-06-07):** SEBI XBRL has **three** taxonomy
    generations, not two. The middle era (~Sep 2022–Mar 2025) uses context IDs
    `InstitutionsForeignI` / `InstitutionsDomesticI` (suffix `I`), which the first parser
    missed → silently emitted **FII=0 / Institutions=0** for ~10 quarters/stock (~30% of the
    panel). Zeros poisoned the Δ signal. A **preliminary IC run on the corrupt data was
    discarded** (do not trust it). Fix: added the missing contexts + residual reconciliation
    (any two of FII/DII/Inst give the third) + a guard that drops any 0% institutional total
    as a parse failure. Re-validated HAL/BEL/TRENT = 0 corrupt quarters, smooth trajectories.
  - **VERDICT (2026-06-07, FULL clean data, 198/202 stocks):** ❌ confirms K7. Institutional
    accumulation has NO usable edge even on deep data covering the multibagger runs —
    Δ-Institutions IC(1y)=−0.025 (slightly negative), FII tercile spread −0.6pp (no edge).
    Winner study: 64% of big winners had institutions buying prior-year, but corr(Δinst,
    run)=**−0.204** — and the 15 biggest winners are ALL the 2023-24 PSU/railway rally
    (IRFC/BSE/HUDCO/RVNL/SAIL/NBCC/IRCON/RAILTEL) where institutions were flat or selling
    while retail/momentum drove it. Institutions chase, they don't lead. The paid-data path
    (Trendlyne) would NOT have helped. (The "never trust silently-corrupted data" rule worked:
    the first pass on a 3rd-taxonomy parse bug was discarded before this clean run.)
  - Minor data gap: `M&M`/`M&MFIN` (the `&` breaks URL encoding) and 2 non-tickers failed;
    198/202 usable — does not affect the verdict.
  - **One nuance → new frontier F6:** Δ-Promoters has a weak but *consistent* positive IC
    (+0.034 / +0.023 / +0.042 across 1q/2q/1y). Too weak alone, but per Grinold a weak-IC
    signal can add value as a small input in a high-breadth basket. Candidate, not a KILL.

---

## 4b. Code audit — 3 passes (2026-06-08)

Triggered by "find the core-file bug." Read the full return-critical path + execution-traced
+ experiment-verified. Findings:
- **P6 (FIXED, big): warmup_skip year-of-cash bug** — see KEEP table. The one real
  return-distorting bug; corrected the walk-forward assessment (3/8→7/8 beats Nifty).
- **No look-ahead** — verified: factor@date is identical with/without future data. CPCV +
  point-in-time defences hold. [H]
- **Survivorship bias (caveat, INFLATES ~2-3pp):** universe = today's 200 survivors; failed/
  delisted names absent. The ~16% full-cycle is an UPPER bound; true ≈13-14% after failure
  injection. Honest, not a bug. [H]
- **Data staleness (hygiene TODO):** only 101/200 cache files reach 2026-05-21; ~99 end Mar30–
  Apr07 → recent-window tail partly frozen. Fix = re-fetch all to a uniform END. Minor.
- **Weight-churn tax drag:** annual reweight realizes STCG just to nudge weights on kept names.
  A `no_trade_band` (now in BacktestConfig, default 0) recovers ~+1pp (let-winners-run) at
  ~+3pp DD — marginal/possibly regime-fit, NOT shipped. Turnover (130%/yr) is dominated by the
  momentum rotation, which earns its keep (2-yr rebal = worse, +9.4%).

## 4c. Statistical-significance / overfitting audit (2026-06-09)

`scripts/overfitting_analysis.py` + `core/portfolio/stats.py` (Bailey & López de Prado),
`reports/OVERFITTING_ANALYSIS.md`. On 60 strategy variants tried:
- **Deflated Sharpe Ratio = 99%** — the deployed config's Sharpe (0.90 ann.) is REAL, not the
  luckiest of 60 draws (luck ceiling only 0.15). PSR-vs-0 = 99.7%. ✅ The edge is significant.
- **PBO = 76%** — picking the in-sample-BEST variant overfits (the 60 near-identical configs
  are statistically indistinguishable; the IS-winner mean-reverts OOS). Lesson: DON'T optimise
  weights — deploy a central robust blend (which we did: n=12 even-ish blend, not an extreme).
  The PBO analysis vindicates that choice. This is the correct, nuanced reading (edge real,
  tuning noise), not "strategy bad."

## 5. 🔭 OPEN FRONTIERS — untested levers worth pursuing

Ranked by plausible edge × feasibility. Each: hypothesis → how to test → realistic ceiling.

- **F1 — Intraday / microstructure (the real new frontier).** **[M]**
  Everything KILLED used only **daily** OHLCV. The **Kite Connect ₹500 dev plan** gives
  intraday/tick data + execution. Documented intraday effects (opening-range breakout,
  intraday momentum/reversal, VWAP) claim 55–60% WR — **but every public backtest
  excludes costs**, and SEBI data shows **~70–93% of retail intraday traders lose money**.
  *Test:* pull intraday bars via Kite, backtest ORB/intraday-momentum **with full STT +
  brokerage + realistic slippage**, walk-forward. *Ceiling:* unproven; treat with extreme
  cost-skepticism. This is where to spend new effort *because it's genuinely untested
  here*, not because it's likely to win.

- **F2 — Low-volatility factor, properly.** ❌ **TESTED → KILL (K10), 2026-06-07.**
  Heavier low-vol tilt cut net CAGR (recent −3.4/−4.7pp). MARK6's existing low_vol .30 +
  inverse-vol weighting already captures the anomaly; more is worse. *Residual value:* a
  low-vol-max config is a **drawdown-reduction knob** (recent MaxDD −18.4% vs −25.3%) if a
  user ever wants lower vol at the cost of ~4-5pp CAGR. Not a return edge.

- **F3 — Quality factor.** 🟡 **promoter-level proxy TESTED → KILL (K11).** A *governance*
  proxy (promoter holding level) adds nothing on top of the blend. **STILL OPEN:** true
  *fundamental* quality (ROE / low debt / cash-flow stability) is untested — it needs
  historical financial statements, which we do NOT have (the XBRL we fetched is shareholding,
  not P&L/balance-sheet). Real frontier = find a free historical-fundamentals source first.

- **F4 — Calendar / structural effects.** **[M]**
  Documented: F&O-expiry timing and SIP-timing edges in Nifty (22-yr study). Small but
  cheap. *Test:* expiry-week entry/exit timing on the basket; budget-day / RBI-policy gates.

- **F5 — Event-driven (index inclusion, earnings drift).** **[L]**
  Index rebalance front-running and post-earnings-announcement drift are documented
  globally; untested here. Needs event dates (free from NSE). Crowded but worth a look.

- **F6 — Δ-Promoter holding as a weak factor input.** ❌ **TESTED → KILL (K12), 2026-06-07.**
  Added as a sleeve via `core/portfolio/external_factors.py` + `factor_research.py`. The weak
  +IC (~+0.04) did NOT convert to a robust net edge (recent −0.3pp, walk-fwd +0.4pp — noise).
  The inst-Δ control behaved identically to no-edge, validating the harness. Confirms the
  Grinold caveat: an IC this small needs far more breadth/orthogonality than one quarterly
  sleeve provides. Mechanism (`extra_factors`) is built & tested for any future sleeve.

---

### Implementation reference for the frontiers (so the next session can just build)

**Kite Connect API (F1 enabler)** — [H], `kite.trade/docs/connect/v3`:
- Historical candles (OHLCV + OI), intervals: minute / 3m / 5m / 10m / 15m / 30m / 60m / day.
- **Per-request history caps:** minute=60d, 3m/5m/10m=100d, 15m/30m=200d, 60m=400d,
  day=2000d → paginate windows for longer ranges.
- **Rate limits:** historical = **3 req/s**; orders = 8 req/s (180/min). Use `kiteconnect`
  (official) or `kitetrader` (built-in throttling).
- **Orders need a static IP** (since 1 Apr 2025). For *research* (data only) no static IP
  needed. Adapter already exists: `core/data/adapters/kite_adapter.py`,
  `core/execution/adapters/kite_exec.py`.

**Low-vol / quality factor construction (F2/F3)** — [H], NSE/MSCI methodology:
- Low-vol: rank by **1-yr stdev of daily returns**, take least-volatile decile/50-of-300;
  weight **inverse-vol** (or hybrid vol×free-float-cap to avoid illiquid small-caps).
- Rebalance **semi-annual** (NSE indices) — but for us prefer annual for LTCG (tax > tracking).
- Multi-factor: NSE Multi-Factor indices = 30 stocks from ≥2 of {Alpha, Quality, Value,
  Low-Vol} — validates MARK6's blend design.
- This is a near-drop-in extension of `core/portfolio/factors.py` + `construction.py`.

## 6. EXTERNAL KNOWLEDGE BASE (curated literature)

| Finding | Grade | Source |
|---------|-------|--------|
| Low-vol anomaly real in India (decile spread +11.4% vs +1.3%, 2001–2015); but conflicting in some periods | [H]/[M] | Pandey, *Low Volatility Anomaly in Indian Stock Market*; Quantpedia |
| Momentum profits large in India but **negatively skewed, crash-prone** → needs risk-management | [H] | Singh et al. 2022, *Risk-Managed Momentum (Indian)*, SAGE |
| Some ML composite anomaly predictors stay significant net of up to ~300bps cost (global, not India-specific) | [M] | *Enhancing stock market anomalies with ML*, Rev. Quant. Fin. Acc. 2022 |
| ORB/intraday WR claims 55–60% **exclude costs**; real returns ~15–25% lower | [M] | IntradayLab Nifty ORB 8-yr backtest |
| **~70–93% of retail intraday traders lose money** in India | [H] | SEBI retail trading studies |
| F&O-expiry / SIP-timing calendar edge exists in Nifty over 22 yrs | [M] | arXiv 2507.04859 |
| Quality factor: cash-flow variability > profitability as quality proxy (India) | [M] | ScienceDirect S097038961730023X |

---

## 7. HARD GROUND TRUTHS (constraints that bound everything)

- **Tax:** LTCG 12.5% (>365d), STCG 20% (≤365d). Turnover is the silent killer — favour
  annual holding. **[H]**
- **Costs:** 0.29% round-trip + 0.10% slippage (equity delivery). Intraday adds STT/day. **[H]**
- **Realistic ceiling (retail, public data) — re-measured 2026-06-11 (P11+P12, clean data &
  clean benchmark):** the DEPLOYED system (momentum-heavy factor book refreshed every 6mo
  under FY tax netting, 50/25/25 sleeves) does **+20.8% net CAGR, Sharpe 0.96, MaxDD −26.6%,
  Calmar 0.78, +9.8pp excess vs Nifty**, ₹5cr→₹35.9cr/10.4y. **The 20% CAGR + Calmar 0.8
  targets ARE now (just) reached — honestly, via a tax-truth fix + momentum-decay capture,
  not overfitting.** Sharpe 1.1 remains out of reach unlevered (vol ~22.7%). Higher raw
  return = deeper drawdown (risk-dial, not skill). **−27% drawdowns are unavoidable and must
  be held through.** The honest forward expectation for ₹5cr: ~19-21%/yr *averaged over a
  full cycle*, single years −10% to +40%. No single-year guarantee exists. (Superseded
  2026-06-10 figure: +18.8%/0.89/0.70 under the old no-loss-credit tax model.) **[H]**
- **PAPER mode always.** Risk limits (5% DD, 2% daily) are survival, never to be relaxed.
- **Data honesty:** never fabricate or fill unavailable data; verify a source is real and
  clean before trusting it (the leakage trap that produced false +4% CAGR once). **[H]**

---

## 8. How to add an entry

```
| Kxx/Pxx/Ixx/Fxx | <approach> | <verdict> | <[H]/[M]/[L]> | <one-line OOS evidence + why> |
```
1. State the hypothesis and the exact test (script + window + universe).
2. Run it OOS, net of tax & costs, vs same-universe buy-and-hold.
3. Record the verdict here with evidence. Link related memories with `[[slug]]`.
4. If it KILLs, it stays KILLed — saving the next session from repeating it.
