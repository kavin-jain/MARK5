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
| K17 | **Quality-as-SCREEN** (exclude bottom-30% fundamental quality before ranking) | 🟡 BLOCKED (API quota) | [L] | Hook built (`Backtester(screen=...)`). 2026-06-11 status: the prior background fetch ran with exhausted quota and wrote 38 `{"error":"no data"}` stubs (now purged); `fetch_fundamentals.py` hardened (no stubs on failure, long 429 backoff, aborts on quota exhaustion, resumable) and re-launched. indianapi.in free tier currently 429-limited — test runs as soon as coverage lands. Honest estimate: ±0-1pp CAGR, mainly DD reduction (per K15's regime pattern). Distinct from quality-as-TILT (K15 KILL). |
| F7 | **Multi-sleeve expansion (+silver +long-gilt), 5 sleeves** | ❌ KILL (overfit) | **[H]** | `multisleeve_riskparity_test.py` (2026-06-10): a 5-sleeve fixed-eq50 blend hits ALL targets in 2022-26 (**+20.4% CAGR, Sharpe 1.18, MaxDD −17.9%, Calmar 1.14**) — BUT silver (SILVERBEES data from 2022) + gilt (LTGILTBEES 2018) have NO pre-existence, so this is **un-backtestable full-period** and rides a once-in-a-decade silver run (+40% 2024 / +24% H1-2025) that external research shows became MORE equity-correlated in 2025. **3-advisor council unanimous: window-selection overfitting, "a disclosure not a deliverable."** Rejected per the user's "no overfitting" rule. Silver/gilt may be added later as small *forward* structural diversifiers (not because of the 20.4% number). |
| P11 | **Fiscal-year tax NETTING in the backtest engine** (2026-06-11) | ✅ KEEP | **[H]** | `efficiency_research.py` + promoted to `backtest.py` (`fy_netting=True` default). The old model taxed every winning sell but gave NO credit for losses — actual Indian law nets losses against gains within the FY (STCL vs STCG then LTCG; LTCL vs LTCG; 8-yr carry-forward), settled each April (positions sold pro-rata if cash short — no implicit leverage). Same trades: +0.5pp full-period, +0.47pp avg walk-forward, 7/8 windows. Not a strategy — a TRUTH fix. Tax paid 0.80→0.60 NAV-units. |
| P12 | **Semi-annual equity rebalance (rebal_bars 252→126)** under honest netting | ✅ KEEP | **[H]** | `efficiency_research.py` (2026-06-11): K3's "longer is monotonically better" was an ARTIFACT of the no-loss-credit tax model, which over-penalised turnover. Under P11 netting: equity sleeve +16.8→**+20.2%** full-period (Sharpe 0.82→0.94), **+2.84pp avg walk-forward, beats 7/8 windows** (worst −7.0pp = 2019-21 COVID-V where annual rode the recovery). FULL SYSTEM (50/25/25 wrapper): 19.0→**+20.8% net CAGR, Sharpe 0.96, MaxDD −26.6%, Calmar 0.78**; wrapper-level walk-forward +1.21pp avg, 6/8. Mechanism: momentum decays at the 6-12mo horizon (matches literature); netting removes the tax wall. Quarterly (63d) has higher mean (+3.98pp) but fatter tails (worst −10.6pp) and 480% turnover — rejected for robustness. Ex-ante hypothesis, single structural parameter, not weight-tuning. NOW THE DEPLOYED DEFAULT. |
| K18 | **Tax-loss harvesting** (monthly check, sell loser & rebuy — India has no wash-sale rule) | ❌ KILL | **[H]** | `efficiency_research.py` (2026-06-11): −0.40pp avg walk-forward, 0/8 windows beat netting-only. Mechanism understood: rebuying RESETS the holding clock → future LTCG (12.5%) converts to STCG (20%) on the recovery, plus ~0.5% churn cost per harvest; the annual rebalance already books natural losses that the FY netting absorbs. TLH works in the US (no clock reset on the replacement-security workaround); in India it's net-negative for this book. |
| K19 | **Frog-in-the-pan momentum quality (FIP, Da-Gurun-Warachka)** as 10% component | ❌ KILL | **[H]** | `efficiency_research.py` (2026-06-11): full-period +0.9pp looked promising but walk-forward says noise — +0.13pp avg, 4/8 windows, worst −2.3pp. US evidence is UP-market-conditional; does not replicate as a robust component here. |
| K20 | **Sleeve-rebalance frequency** (wrapper 50/25/25 quarterly/semi-annual vs annual) | ❌ KILL | **[H]** | `efficiency_research.py` (2026-06-11): 18.8/19.1/19.0% — noise-level spread; faster sleeve rebalance adds STCG drag for no rebalancing-premium gain at these correlations. Annual sleeves stay. |
| BUG3 | **Nifty benchmark silently overwritten with partial data (FIXED 2026-06-11)** | ✅ FIXED | **[H]** | `refetch_all.py`'s ^NSEI refresh saved a PARTIAL yfinance response (2007–2017 only) over the good benchmark file → every vs-Nifty figure computed after that was garbage (Nifty showed +1.5% CAGR). Re-fetched full history (2007–2026-06-09); added a guard: never overwrite unless >4000 rows AND reaches the requested END year. Lesson = same as BUG2: NEVER trust a fetch without a recency+length check. |
| K21 | **Faster symmetric rebalance (21/42/63d) under honest netting** | ❌ KILL (21/42d) · 🟡 (63d) | **[H]** | `exit_speed_research.py` (2026-06-11): full-period means flatter (21d +22.6%!) but walk-forward fails — 21d −1.38pp avg 3/8, 42d −2.34pp 1/8. 63d: +1.13pp avg but only 5/8, worst −6.2pp = higher-mean/fatter-tail coin-flip, NOT robust enough to displace 126d (which beat annual 7/8). The response curve under TRUE tax: flat 21-126d on mean, 126d dominates on consistency; cliff at 189d+. Losing windows are always crash-recoveries (2018-20/2019-21) — fast re-ranking sells V-recovery names at the bottom. |
| K22 | **Asymmetric fast derank-exits** (126d entries + 21/42/63d exit checks, exit_rank 24/18) | ❌ KILL | **[H]** | `exit_speed_research.py` (2026-06-11): the "cut faders faster" hypothesis. Full-period optics excellent (check21/x18: +22.8%, Sharpe 1.03, hold 173d) but walk-forward = +0.25pp avg, **4/8 windows, worst −6.1pp** (2018-20). Same whipsaw mechanism as K21: in crash-recovery regimes the fast exit dumps temporarily-deranked names that then lead the rebound. Momentum needs ~6mo to re-form after shocks; 126d full-cycle stays the deployed default (avg hold 262d). |
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

## 4c. Statistical-significance / overfitting audit (2026-06-09, re-run 2026-06-11 for v7.0)

`scripts/overfitting_analysis.py` + `core/portfolio/stats.py` (Bailey & López de Prado),
`reports/OVERFITTING_ANALYSIS.md`. **v7.0 re-run (2026-06-11)** on **77 trials** — the full
factor-weight grid (60) + rebalance frequencies (5) + asymmetric/TLH/FIP/sleeve variants (12),
deployed = mom_heavy/n12/t1.5/126d under FY netting:
- **Deflated Sharpe Ratio = 99.3%** — the deployed Sharpe (0.96 ann.) is REAL, not the
  luckiest of 77 draws (luck ceiling 0.16). PSR-vs-0 = 99.8%. ✅ Edge significant.
- **PBO = 74.5%** — unchanged lesson: picking the in-sample-BEST variant overfits (the
  near-identical configs are statistically indistinguishable; the IS-winner mean-reverts).
  This is exactly why we deployed 126d (7/8 walk-forward consistency) and NOT the
  full-period-best 21d (+22.6% IS, fails OOS 3/8) — the PBO analysis vindicates choosing
  on robustness, not the max. Nuanced reading stands: edge real, fine-tuning is noise.

## 4d. v7.1 — Adversarial audit & engine truth fixes (2026-07-22)

A 16-agent black-box audit (every line read, every script run, 7 headline claims
adversarially verified with synthetic-trade probes) confirmed the core engine and
found the following, ALL FIXED in v7.1:

| # | Finding | Fix | Headline impact |
|---|---------|-----|-----------------|
| A1 | Benchmark was the ^NSEI PRICE index while the book earns dividends (auto_adjust) — flattered vs-Nifty ~1pp/yr | Nifty **TRI** via NIFTYBEES-adjusted series (`load_nifty`), taxed like the strategy; bad-print filter for the Dec-2019 split glitch | excess +9.8 → **+9.6pp** |
| A2 | Same-close execution (signal and fill on day-d close) | `exec_lag=1` next-close fills (default; 0 = legacy) | ~−0.1pp |
| A3 | Average-cost lots + blended entry date ≠ statutory FIFO — misclassified STCG/LTCG on top-up-then-sell paths | Per-lot FIFO tracking in `backtest.py`, per-lot term classification | ~0 (netting absorbs) |
| A4 | Buys not cash-constrained → permanent ~0.25% costless overdraft | Buys scaled to available cash | ~0 |
| A5 | Suspended/delisted names compounded at 0% and exited at full frozen value; `eligible()` never enforced its documented "Priced" check | Stale-print watchdog (21d) + 25% haircut force-exit; `max_stale_days` in `eligible()` | 0 today (survivor cache), matters for failure injection |
| A6 | Sharpe reported with rf=0 (0.96 ≈ 0.68 excess at 6.5% rf); Sortino non-standard | `metrics()` reports raw + excess Sharpe, LPM2 Sortino | presentation |
| A7 | `efficiency_research.py` sleeve-rebalance costs always 0 (turnover computed after reset); "A baseline" label silently ran netting | Both fixed | immaterial (verified <0.05pp) |
| A8 | Presentation drift: README carried stale v6 stats (60 trials/PBO 75.6%), dead links (setup.sh, 2 scripts, 2 reports), stale committed reports 4.5pp below fresh output, hardcoded "+5.3pp" claim vs computed 4.9pp | Full README rewrite from regenerated artifacts; vs-EW alpha now computed in-script; universe pinned (`config/universe_tickers.json`); trade ledger committed; CI slimmed | trust |

**v7.1 verdict [H]:** deployed system on the honest engine = **+20.7% net CAGR /
raw Sharpe 0.96 / excess Sharpe 0.68 / MaxDD −26.5% / Calmar 0.78 vs Nifty TRI-net
+11.1%** (excess +9.6pp; engine alpha vs same-universe EW **+4.7pp**, computed).
Equity sleeve +20.0%, walk-forward **7/8 vs Nifty TRI, 8/8 vs EW**. DSR 99.3% (77
trials), PBO 76.7% — still an honest FAIL of fine-tuning, still the reason we
deploy the 7/8-consistency config and not the in-sample best. The audit's meta
lesson: the biased components contributed almost nothing — the edge is structural
(diversified beta + tax discipline + momentum refresh), which is why honesty was
cheap. Survivorship (~1-2pp) remains the largest disclosed inflation.

## 4e. v7.2 — TRUE point-in-time universe + tranching (2026-07-22)

| # | What | Verdict | Evidence | Result |
|---|------|---------|----------|--------|
| P13 | **Survivorship SOLVED: universe rebuilt from NSE daily bhavcopy** | ✅ KEEP | **[H]** | `fetch_bhavcopy.py` + `build_pit_cache.py`. 3,064 trading days (2014-2026), 1,333 symbols, **178 (13.4%) stopped trading** — delisted names a yfinance cache structurally cannot hold. Bhavcopy is UNADJUSTED (the claim that PREVCLOSE is split-adjusted is FALSE — verified on IRCTC 1:5, CLOSE 913.5 vs PREVCLOSE 4130.15); joined 25k corporate-action records. Validation: IRCTC split adjusted, 0/1333 residual >45% single-day moves, 12/12 daily-return correlation >0.99 vs the independently-adjusted yfinance series. ETFs excluded structurally by **ISIN prefix INF** (name heuristics missed SETFGOLD/LICMFGOLD/AXISGOLD/GROWWGOLD). |
| K23 | **Absolute rupee liquidity floor (`min_turnover`)** | ❌ KILL as primary screen | **[H]** | Rs 20cr/day admitted **0 names in 2016 but 436 in 2026** — a fixed rupee threshold is NOT time-invariant as market turnover grows ~4x. Worse, the fallback silently degraded to *no filter*, so the early years traded 1,333 micro-caps (fake -53% DD). Replaced by `top_n_liquid` (top-N by turnover), which is time-invariant and capacity-meaningful, mirroring NSE's own Nifty 500 turnover-rank rule. Kept only as a secondary hard floor with a defined degrade path. |
| P14 | **Rebalance tranching** (3 sleeves, 42-bar stagger) | ✅ KEEP | **[H]** | Anchor lottery measured over 19 offsets: single-anchor CAGR spans 18.59-26.79% (std **2.17pp**) purely by start date; 3-tranche blend std **0.51pp** (**-67% dispersion**) for a -0.29pp mean shift. On the PIT universe it also *raised* the equity sleeve 18.6→20.6% and cut DD -54.4→-51.9%. Variance reduction by averaging — not an alpha claim, which is why it is trustworthy. |
| P15 | **Nested walk-forward: config selection is NOISE** | ✅ KEEP (as knowledge) | **[H]** | `nested_walkforward.py`, 40 configs, config re-picked yearly on prior data only. IS→OOS config rank Spearman = **-0.126**. Chained OOS: learned rule **+21.85%**, 1/N ensemble **+24.00%** — the learned rule LOSES to averaging everything. Never deploy a learned config; deploy a fixed economically-motivated one. (Caution logged: WFE=1.07 looked like a "PASS" but only reflects 2020-25 being a kinder regime than 2016-19 — regime artefact, not skill.) |

**The v7.2 honest headline (survivorship-free, top_n=300):** equity sleeve **+18.6% net CAGR,
excess Sharpe 0.56, MaxDD -54.4%**; full 50/25/25 system **+21.3% CAGR, excess Sharpe 0.67,
MaxDD -31.2%, Calmar 0.68** vs Nifty TRI-net +11.1% → **+10.4pp/yr excess**, ₹5cr→₹37.9cr/10.4y.

**Three lessons that matter more than the numbers:**
1. **Survivorship cost is ~5pp/yr at matched breadth** (PIT top_n=150 → 14.6% vs survivor-cache
   20.1%), closely matching the published Indian estimate of 4.94pp (arXiv 2603.19380). The
   deployed headline only returns to ~21% by deliberately reaching *wider* (300 names) — the
   bias was not harmless, it was traded away for capacity.
2. **Survivorship hid RISK, not return.** True equity-sleeve MaxDD is **-54%**, not -38%. The
   multi-asset wrapper is what makes the system holdable, and that is now a load-bearing fact
   rather than a nice-to-have.
3. **The factor engine's edge over equal-weight GREW on honest data: +4.7pp → +7.4pp (8/8
   windows).** Momentum/trend systematically avoid names that die; equal-weight rides them to
   zero. A survivor-only backtest structurally cannot show this — previously invisible real alpha.

## 4f. v7.3 — Structural-lever sweep on the honest PIT universe (2026-07-26)

Full re-read of every core and script file, then a walk-forward sweep of the levers
that had never been tested on the survivorship-free universe. Bar as always:
beat the deployed baseline in **≥6/8 rolling 3-year windows**, counted **per metric**
(judging a risk lever by a return win-count is the wrong axis — a mistake this sweep
corrected mid-run). Baseline = deployed config on `data/pit_cache`, 2016→2026-07-21:
equity sleeve **+18.92% / shExc 0.60 / MaxDD −47.0%**, full 50/25/25 system
**+20.67% / shExc 0.89 / MaxDD −24.9% / Calmar 0.83** (reproduces `docs/data/mark6.json`
exactly, confirming the published headline).

| # | Approach | Verdict | Evidence | Result |
|---|----------|---------|----------|--------|
| K24 | **Correlation-aware weighting** (min-variance / ERC / max-diversification instead of inverse-vol) | ❌ KILL **before testing** | **[H]** | `risk_model_diagnostic.py`: the book holds 20 names and makes **14.9 independent bets** (eigenvalue entropy), mean pairwise corr **0.204**, diversification ratio **2.03**. For that correlation level the theoretical minimum portfolio vol is 23.5%; the book realises **23.7%**. Inverse-vol is already at the diversification limit — there is no headroom for a covariance model to recover. The −47% sleeve DD is NOT a correlation problem: selected names average **47.6% annualised vol**. Diagnostic killed the idea for the cost of one script. |
| K25 | **`n_hold=12` (i.e. P5) re-tested on honest PIT data** | ❌ **KILL — FALSIFIES P5** | **[H]** | `edge_research_2026_07.py`: −5.42pp avg, **1/8 windows**, MaxDD −47.0→−54.3%, Sharpe 0.60→0.51. P5's "12 beats 20 in 8/8" was measured on the **survivor cache**; concentration is only safe when the universe cannot contain names that die. On honest data the concentrated book is strictly worse. **n_hold=20 stands.** Textbook demonstration that a survivor-biased backtest mis-ranks *risk* decisions, not just return. |
| K26 | **Breadth expansion** (`n_hold` 30 / 40) | ❌ KILL | **[H]** | 2/8 and 2/8. Grinold's √breadth does not pay here — the marginal name is worse-scoring by exactly enough to offset the diversification. n_hold≈20 is a genuine optimum, not a tuned one. |
| K27 | **Turnover reduction via wider hold buffer** (`buffer_mult` 3.0 / 4.0) | 🟡 marginal | [M] | 4/8 and 4-5/8 — inside noise on every metric. Does cut turnover 255%→216%/yr for no measurable cost, so it is free, but it is not an edge. |
| K28 | **Max-volatility exclusion screen** (drop top 10%/25% most-volatile before ranking) | ❌ KILL as a return lever | **[H]** | 3/8 CAGR. Genuinely cuts vol (24.1→19.9%) and lifts Sharpe (0.60→0.70) but fails the walk-forward return bar — the same shape as K10: the low-vol anomaly is real, harvesting it harder costs CAGR. |
| P16 | **Rebalance tranching (P14) — CONFIRMED but CAPITAL-GATED** | ✅ KEEP · ⛔ not deployable at current capital | **[H]** | 3 tranches × 42-bar stagger, n_hold=20: sleeve +18.92→**+21.49%**, shExc 0.60→0.70, MaxDD −47.0→−44.6%, **6/8 on CAGR, Sharpe AND MaxDD** — the only clean win in the sweep, and a validated KEEP that had **never been wired into production**. **But it requires 3×20 = 60 whole-share slots.** At ₹5L capital that is ₹4,167/slot: 4 of the 20 current holdings cost more than that per share (THANGAMAYL ₹6,442, MTARTECH ₹5,852, GVT&D ₹4,520, NETWEB ₹4,355) and average weight error from whole-share rounding hits **33.6%**. Clean execution needs ≈**₹1.55cr**. Capital-efficient variants were tested and **fail**: 3×n_hold-7 (21 slots) gives +20.61% CAGR but MaxDD **−55.0%** (worse than baseline) and **1/8** on MaxDD — the benefit came from holding 60 names, not from staggered anchors alone. **Deploy at ≥₹1.5cr; do not deploy at ₹5L.** |
| P17 | **Rank-transform the factor cross-section** (z-score ranks, not raw values) | ✅ KEEP as a RISK lever only | **[H]** | Momentum is heavily right-skewed; one name up 400% sets the z-scale and squashes every real distinction below it even after 3σ clipping. Ranking makes the score depend only on ORDER, which is all the composite uses. Result is unambiguous and one-sided: **MaxDD 7/8 windows** (sleeve −47.0→−39.7%, system −24.9→−22.2%), Calmar 0.83→0.93, Sharpe 0.89→0.92 — but **2/8 on CAGR**. Honest reading: a reliable drawdown reducer that buys no return. Combined T+R reaches **8/8 on MaxDD** at sleeve level. |
| P18 | **Capacity / market-impact analysis** (never previously done) | ✅ KEEP (as knowledge) | **[H]** | `capacity_analysis.py`, square-root impact law (Almgren), 20d median rupee ADV per held name. Median held name trades **₹30.8cr/day**, 10th-pctile ₹4.9cr/day. Every position stays under 10% of daily volume to **₹1cr**; ≤5% of positions breach it to **₹10cr**. At the quoted **₹5cr** headline: worst-case participation 16.8%, modelled drag **0.24%/yr** — so the published number survives, minus a ~0.24pp haircut the backtest does not model. Breaks down by ₹50cr (26% of positions over the limit, 0.75%/yr drag). **Honest capacity: ₹10–25cr.** |

**v7.3 meta-lesson.** Ten structural levers tested; **one** cleared the bar, and it is
gated behind ~30x the current capital. The strategy is at its practical ceiling *for
its capital and constraints* — the binding constraint on this system is no longer
ideas, it is capital, track-record length, and the long-only/unlevered retail
structure. Two logged KEEPs were also corrected: P5 (n_hold=12) is falsified on
honest data, and P14 (tranching) is real but not executable at ₹5L.

### Engine defects found in the same pass (all fixed)

| # | Defect | Impact |
|---|--------|--------|
| BUG4 | `paper_track.py` accrued tax to `book["tax_accrued"]` but **never deducted it from NAV or cash** | The live NAV would have overstated itself permanently from the first rebalance (due ~Jan 2027). Caught before it fired — 0 rebalances so far. Fixed with real FY netting (`net_fy_tax`), so the live book and the backtest engine now apply the **same tax law** (P11) instead of two different ones. |
| BUG5 | `generate_portfolio.py` screened the universe by `liquidity_pct=0.40` while the deployed book uses `top_n=300` | The "executable deliverable" printed a **different portfolio than the live book holds**. Fixed to match. |
| BUG6 | `construction.py` gave a selected name with NaN volatility **zero weight**, silently holding fewer than `n_hold` names | Rare but invisible. Fixed by imputing the median vol of the picks. |
| BUG7 | `max_sector_weight=0.30` is **dead code** — `sector_map` is never passed to `PortfolioConstructor` anywhere in production | Measured impact small (top sector averages 20% of the book, breaching 30% in only 3/21 rebalances) but the cap is advertised and not enforced. Also 7.7 names/rebalance are absent from `config/sector_map.json` and would escape the cap regardless. |
| — | `config/system_config.json` is a **dead MARK3 artifact** referenced by no code | 300 lines still advertising "Advanced AI Stock Prediction System", XGBoost/LSTM/GRU ensembles, news-sentiment, 5% stop-losses, Redis/TimescaleDB — every one of which this log KILLED. A reviewer reading the public repo finds it. Flagged, not deleted. |

## 4g. v7.3 — DEPLOYED changes, attribution, and the factor-regression reality check (2026-07-26)

### Deployed to the live book on 2026-07-26 (day 4, 36 trades, all in the append-only ledger)

| # | Change | Why | Effect (full system, walk-forward) |
|---|--------|-----|------------------------------------|
| P17 | **Rank-transform** the factor cross-section (`ConstructionConfig.rank_transform=True`) | momentum is right-skewed; one name up 400% sets the z-scale and squashes every real distinction below it — clipping caps that name but not the inflated σ it created | MaxDD **7/8** windows |
| BUG7fix | **Sector cap ENFORCED** — `load_sector_map()` now passed to `PortfolioConstructor` in all 5 production scripts | the 30% cap was configured and dead since inception | MaxDD **8/8** windows (sleeve) |
| P19 | **Largest-remainder share allocator** in `paper_track.py` (init + rebalance residual sweep) | naive `floor()` rounds every position DOWN, stranding cash and pulling weights one-way off target | see below |

Combined **R+S**, validated ≥6/8 on MaxDD *and* Calmar at BOTH levels:
**CAGR +20.67→+20.87%, excess Sharpe 0.89→0.94, MaxDD −24.9→−22.2% (7/8), Calmar 0.83→0.94 (6/8).**
Deliberately deployed **without** `n_hold=25`, which showed better full-period optics
(Sharpe 0.97) but **worse** walk-forward consistency (Calmar 6/8→4/8) — the same PBO
discipline that chose 126d over the in-sample-best 21d.

| # | Finding | Verdict | Evidence | Result |
|---|---------|---------|----------|--------|
| P19 | **Whole-share granularity is a real, fixable cost at small capital** | ✅ KEEP | **[H]** | `capital_flexibility.py`, measured at each real rebalance against real forward prices. Naive `floor()` drags **−0.35pp/yr at ₹5L** and **−1.26pp/yr at ₹1L**, stranding 1.9%/9.8% of the book in idle cash and pulling weights **6.6pp** off target — always downward. Largest-remainder apportionment cuts the drag to **+0.11pp at ₹5L** (i.e. zero) for no strategy change, no added risk and no overfitting surface. Verified live: idle cash **₹16,245 → ₹48** at the 2026-07-26 rebalance. **This is the answer to "make a small book behave like a large one."** |
| P20 | **Attribution: what is skill vs what anyone can buy** | ✅ KEEP (as knowledge) | **[H]** | `attribution.py`. Of the total 7.52x gain: **gold 25% + US Nasdaq 31% = 55% came from two passive ETFs**; the equity book contributed **45%**. Inside the equity sleeve, against equal-weight of the SAME point-in-time universe, net of the same tax and costs: **+8.63pp/yr selection alpha** (+20.06% vs +11.44%). Both halves must always be reported together — quoting +22.5% system CAGR as "stock picking" would be false. |
| **K29** | **Is the equity alpha real once you control for known factors?** | ⚠️ **NOT PROVEN** | **[H]** | `risk_report.py` regresses the book on long/short tercile factors **built from this same point-in-time universe** (market, size=small−big by turnover, momentum, low-vol), rebuilt every 21 bars. **Equity sleeve: annualised alpha +4.42%/yr but t = 1.19 — NOT statistically significant.** R² = 0.71, explained by market β 0.757, momentum β 0.633, low-vol β **−0.535** (i.e. a deliberate tilt INTO high-volatility names, confirming the 47.6% average name vol from K24's diagnostic). The full 50/25/25 system shows alpha +10.47%/yr at t = 3.46, but that is significant largely because gold and Nasdaq are **not in the factor model** — it is diversification, not stock selection. **Honest characterisation: MARK6 is efficient, tax-disciplined harvesting of the momentum premium plus genuine multi-asset diversification. It is NOT demonstrated idiosyncratic alpha.** The +8.63pp vs equal-weight in P20 is real as a comparison but is substantially momentum-factor exposure, which equal-weight simply does not have. This is the single most important honest statement about the system and it belongs on any page or application that describes it. |
| P21 | **Tail-risk profile** | ✅ KEEP (as knowledge) | **[H]** | Daily skew **−0.93**, kurtosis **9.76** (normal 3.0) — materially fat-tailed. 99% 1-day historical VaR **−2.89%** vs parametric **−2.05%**: a normal model **understates** the bad days by ~40%, so any risk figure quoted parametrically is optimistic. 21-day 99% CVaR **−13.4%**. Worst rolling 1-year **−15.8%**; **12%** of rolling 1-year windows were negative. Drawdown attribution confirms the equity sleeve drives every major drawdown, with gold offsetting in 2018 (+0.8%) and 2025 (+3.3%) — the multi-asset structure is load-bearing, not decorative. |

**v7.3 closing position.** Twelve structural levers tested across two sweeps; two were
deployed, both risk levers, neither adding return. The system's honest description is
now precise: **a momentum-tilted, high-volatility, long-only Indian equity book (β_mkt
0.76, β_mom 0.63) at 50%, diversified with gold and US tech, harvested tax-efficiently,
capacity ~₹10–25cr.** Its measured edge over the index is large and real; its edge over
*the factors it is made of* is +4.4%/yr and not statistically significant on 10.6 years.
Both statements are true and both should be said.

## 4h. v7.4 — Can this system reach Sharpe 1.1? Solved, not searched (2026-07-26)

The question was attacked analytically rather than by tuning. For any set of assets
the maximum Sharpe of ANY fixed combination is closed-form, `S* = sqrt(mu' Sigma^-1 mu)`,
so the ceiling is computable before any search — and the requirement for closing a
gap can be *solved for* instead of hunted.

| # | Finding | Verdict | Evidence | Result |
|---|---------|---------|----------|--------|
| P22 | **Deployed 50/25/25 is badly risk-unbalanced** | ✅ KEEP (as knowledge) | **[H]** | `allocation_robustness.py`. 50% of CAPITAL in the equity sleeve is **66% of the portfolio's RISK** (gold contributes just **8%**). The allocation was chosen by grid search on returns (P10), and it shows. |
| P23 | **Risk parity (ERC)** — validated, then **DECLINED on design grounds** (see decision note below) | ✅ VALID · ⛔ **NOT DEPLOYED** | **[H]** | Equal Risk Contribution uses **only the covariance matrix — no expected returns at all**, so it cannot be return-chasing by construction. It lands at **~29% eq / 45% gold / 26% US** and is **stable in all 8 rolling 3-year subperiods** (std 2.7 / 3.6 / 2.6pp), including 2016-2018 when gold did nothing. Measured through the real wrapper net of tax: **Sharpe 0.93→0.99, MaxDD −22.1%→−17.9%, Calmar 0.94→1.11.** Walk-forward: **MaxDD 8/8 windows**, Calmar 6-7/8, Sharpe 5/8. The drawdown result is the most robust finding in the entire project. |
| K30 | **Learned / optimised allocation, re-picked yearly on prior data** | ❌ KILL | **[H]** | `allocation_walkforward.py`: learned +22.35%/yr vs fixed 50/25/25 +24.15%/yr, beating it in only **3/8 years**. Extends P15 (config selection is noise, ρ=−0.126) from the equity config to the ASSET ALLOCATION. Deploy a fixed, economically-motivated allocation; never a fitted one. |
| K31 | **Faster sleeve rebalancing to capture the risk-parity Sharpe** | ❌ KILL | **[H]** | `sleeve_rebalance_erc.py`, charging **real ETF friction (0.15% round trip) and realised STCG/LTCG**, which the dashboard's `wrap()` does not. Hypothesis was that ERC's 45% gold sleeve drifts far more than 25% does, so K20's "cadence is noise" might not hold at these weights. It holds: 21d→504d spans Sharpe 0.97–0.99 with no ordering. The theory-vs-measurement gap is **not** drift. |
| **K32** | **Sharpe 1.1 is NOT attainable — the binding constraint is TAX, not strategy** | ❌ **KILL the target** | **[H]** | `sharpe_ceiling.py` + `path_to_sharpe_11.py`. Decomposition at ERC weights: three sleeves perfectly uncorrelated would give **1.278**; the real eq-US correlation of **0.289** costs −0.12 → **1.155** theoretical; Indian tax + transaction friction costs a further **−0.16** → **~0.99 measured**. So the single largest obstacle between this book and hedge-fund Sharpe is **the tax and cost regime it operates in**, not the signal, the weighting or the assets. An offshore or tax-exempt vehicle running the identical book would score ~1.15. That is not available at Indian retail. |
| K33 | **The whole gold tilt is conditional on gold** | ⚠️ DISCLOSURE | **[H]** | Gold earned **17.65%/yr (excess 11.1%)** over this sample — an exceptional decade. Stress: force gold's excess to a normal **4%**, keeping its real vol and real correlations. ERC then scores **0.90**, and the **best possible allocation of the three assets reaches only 0.971** — 1.1 becomes unreachable by any weighting. The DIVERSIFICATION benefit (eq-gold correlation **0.005**) is structural and survives; the RETURN contribution does not. Any claim built on the gold sleeve must carry this caveat. |
| K34 | **Candidate 4th sleeves** | ❌ KILL (all) | **[H]** | Marginal-value rule `S_new > rho x S_port`: LTGILTBEES passes but with Sharpe 0.10 over only 8.2y; SILVERBEES (0.87) and MAFANG (0.95) pass but hold 4.4y and 5.2y of a single favourable regime — exactly F7's failure mode — and MAFANG is US mega-cap tech, i.e. a second helping of MON100 rather than a diversifier. LIQUIDBEES and GILT5YBEES "pass" only via a NEGATIVE optimal weight, i.e. shorting cash/bonds = leverage, already killed by K13 at Indian financing costs. **No deployable fourth sleeve exists in the available data.** |

### DECISION (2026-07-26): allocation stays 50/25/25 — deliberately, not by default

P23 (risk parity, ~29/45/26) is statistically the strongest result in this log:
Sharpe 0.93→0.99, MaxDD −22.1%→−17.9%, Calmar 0.94→1.11, **MaxDD better in 8/8
walk-forward windows**, derived without expected returns and stable in every
subperiod. It was nevertheless **NOT deployed**, and the reason is a design
constraint rather than a statistical one:

> Risk parity would cut the Indian equity sleeve from 50% to ~29% of capital.
> MARK6 is intended to be an **Indian stock-market system**. At 29% equity it
> would be a multi-asset fund holding a minority stock sleeve, and describing it
> as a stock-picking system would stop being accurate. The +8.63pp/yr selection
> alpha (P20) is the part of this project that represents actual skill; diluting
> it to buy ~0.06 of Sharpe and ~4pp of drawdown was judged the wrong trade.

This is recorded so the distinction survives: **P23 was validated and declined,
not falsified.** If the design goal ever changes — a pure risk-adjusted-return
mandate rather than a stock-picking one — the evidence to act on is already here
and needs no re-testing. A 40/30/30 middle option (Sharpe 0.97, MaxDD −20.8%,
Calmar 1.00, MaxDD 8/8) also remains available and keeps equity the largest sleeve.

**Target status: Sharpe 1.1 is formally ABANDONED as unattainable (K32), accepted
as such rather than pursued further.** The measured unlevered long-only ceiling under
Indian tax is ~1.00. Continuing to chase the last 0.1 would mean re-litigating
settled KILLs (K13 leverage, F7/K34 short-history sleeves) or overfitting. The
deployed system stands at **Sharpe 0.94, MaxDD −22.2%, Calmar 0.94, CAGR +20.87%**,
which is honest, reproducible, and defensible.

**v7.4 verdict.** The honest ceiling for this system, unlevered and long-only under
Indian tax, is **Sharpe ≈ 1.00**. Risk parity gets there and simultaneously delivers
the best drawdown and Calmar the project has ever measured (**−17.9% / 1.11**), on the
most robust evidence in the log (8/8 windows, derived without expected returns, stable
in every subperiod). **Sharpe 1.1 requires either escaping the tax regime or finding a
genuinely uncorrelated long-history asset that does not exist in the data.** Both of the
project's stated risk goals — lowest drawdown, highest Calmar — ARE reachable; the
Sharpe target is not, and chasing it further would mean overfitting.

## 4i. v7.5 — Attacking the tax regime, the binding constraint (2026-07-26)

K32 identified tax + friction as the single largest obstacle between this book and
hedge-fund Sharpe (−0.16). This is the follow-up. Scope note: this is quantitative
research on **tax-efficient portfolio construction** using Indian law as the engine
models it. It is not tax advice, and a real-money implementation needs a qualified
professional.

**Exact size of the prize, measured (equity sleeve, zero-tax counterfactual):**
tax costs **2.91pp of CAGR and 0.112 of Sharpe** (+20.06% / 0.673 taxed vs
+22.97% / 0.784 untaxed).

| # | Approach | Verdict | Evidence | Result |
|---|----------|---------|----------|--------|
| K35 | **LTCG-aware exit deferral** — hold a position past 365 days when it is close, converting STCG 20% → LTCG 12.5% | ❌ KILL — **prize is already captured** | **[H]** | Looked mechanically attractive: **75% of SELLS are STCG**. But by VALUE only **12% of gains** are — the big winners are already held long, so 88% of gains already receive 12.5%. Worse, the holding-period histogram has **zero sells in the 300-365d window**: the 126-bar cadence lands rebalances at ~182 and ~365 calendar days, so nothing sits just below the threshold waiting to be nudged over. Converting every STCG gain would be worth ≈**0.07pp/yr**. The semi-annual cadence plus the ranking buffer already deliver the deferral this lever was meant to add. Not built. |
| **P24** | **Sec 112A ₹1.25 lakh annual LTCG exemption — modelled for the first time** | ✅ **KEEP, DEPLOYED** | **[H]** | The engine had never modelled it (documented as "conservative"). It is real law, it is the largest *legitimate* tax lever available, and its value is **capital-dependent** — which is exactly why a scale-free backtest misses it. Now implemented via `BacktestConfig.capital_inr` + `ltcg_exemption_inr`, applied **after** loss set-off, and **only to Sec 112A-eligible gains** (listed Indian equity). GOLDBEES and MON100 fall under different provisions and are deliberately excluded — exempting them would understate the bill. |

**Measured effect of P24 (equity sleeve CAGR, and the decay that validates the model):**

| total capital | sleeve CAGR | vs scale-free | tax paid |
|---|---|---|---|
| scale-free (old headline) | +20.06% | — | 1.109 |
| ₹1L | +21.73% | **+1.67pp** | −45% |
| **₹5L (the live book)** | **+21.27%** | **+1.21pp** | **−28%** |
| ₹25L | +20.41% | +0.35pp | −7% |
| ₹5cr | +20.06% | +0.00pp | −0% |

The benefit decays monotonically to exactly zero as capital grows, which is the
correct shape for a fixed rupee allowance and is the model's own validation.

**Full system 50/25/25 at the live book's ₹5,00,000: +21.44% CAGR, excess Sharpe
0.97, Calmar 0.97** — versus the published scale-free +20.87% / 0.94 / 0.94.

> **The headline UNDERSTATES a small book by +0.56pp CAGR and +0.03 Sharpe.** Both
> figures are correct; they answer different questions. The scale-free number is the
> right one to quote for institutional capacity (₹5cr+), the capital-aware number is
> the right one for what this ₹5 lakh paper book will actually experience. The public
> page must show both and say which is which.

Also deployed to the live book: `paper_track.net_fy_tax` now applies the same Sec 112A
exemption with the same equity-only eligibility rule, so the live record and the
research engine continue to use identical tax law. Verified against 7 hand-computed
cases including ETF-sourced LTCG correctly receiving **no** exemption.

**Where the remaining tax drag actually sits.** After P24 the residual is *not*
avoidable by construction: 88% of gains are already long-term at 12.5%, losses
already net within the fiscal year (P11), harvesting resets the clock and loses
(K18), and faster or slower cadences have been tested to exhaustion (K3/K20/K21/K31).
**The remaining ~2pp is the statutory rate on real gains.** The only structures that
remove it — offshore domicile, tax-exempt vehicles — are not available to Indian
retail and are outside this project's scope. K32's verdict stands: Sharpe ~1.00 is
the honest unlevered ceiling, now reached at **0.97 on the live book's actual capital**.

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

- **F4 — Calendar / structural effects.** ✅ **TESTED → CLOSED (2026-06-11).** **[H]**
  `rebalance_date_sensitivity.py`: deployed config run from 13 staggered anchors
  (0–120 bars). CAGR mean +22.1%, std 1.7pp, range +19.4%…+25.7%. Two conclusions:
  (1) **the deployed Jan-anchor headline (+20.2% equity sleeve) sits near the BOTTOM of
  the anchor distribution — the reported number is conservative, not anchor-lucky**;
  (2) dispersion tracks the period skipped (late-2016 anchors miss the H1-16 drawdown),
  NOT a stable month-of-year effect → no exploitable calendar edge at our cadence, and
  picking the best anchor (+25.7%) would be textbook anchor-mining. Honest forward
  statement: ~20-22% ±2pp depending on cycle phase at entry.

- **F5 — Event-driven (index inclusion, earnings drift).** **[L→downgraded 2026-06-11]**
  Data-source scan: Indian PEAD studies (100 NSE firms, 2014-18) find the market largely
  EFFICIENT post-announcement — no robust drift to harvest. News sentiment in India is
  short-lived (1-10 days; strongest documented effect ≈13bps/mo in a LONG-SHORT setup we
  can't run), and free news sources have no point-in-time archive → lookahead trap.
  Index-inclusion front-running remains documented but hits ~1-2 of our 12 names/yr →
  breadth too low to matter (≤+0.2pp est.). Verdict: news/PEAD/event data sources are NOT
  worth integration effort at our 6-month cadence. The one actionable open data item
  remains **K17 quality-as-SCREEN** (needs `fetch_fundamentals.py` + indianapi key;
  honest estimate ±0-1pp CAGR, mainly DD reduction in flight-to-quality regimes).

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
