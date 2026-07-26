<div align="center">

# MARK5 / MARK6 — Honest Quantitative Equity Research

**A research-grade, survivorship-free, tax-aware quantitative portfolio system for NSE Indian equities.**

*Built not to claim an edge, but to find out — rigorously — whether one exists.*

`PAPER MODE ONLY` · `Net of Indian tax & costs` · `Survivorship-free universe` · `Walk-forward validated`

[![CI](https://github.com/kavin-jain/MARK5/actions/workflows/ci.yml/badge.svg)](https://github.com/kavin-jain/MARK5/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20619267-blue)](https://doi.org/10.5281/zenodo.20619267)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Live dashboard](https://img.shields.io/badge/live-kavinjain.in%2Fmark6-red)](https://kavinjain.in/mark6)

</div>

---

## Read this first

This project's most important result is a **negative** one, and it belongs at the top rather than in a footnote:

> **Once you control for the factors this book is made of, its stock-selection alpha is
> +5.8%/yr with a t-statistic of 1.61 — which is _still not statistically significant_.**
>
> Regressed on long/short market, size, momentum and low-volatility factors built from
> its own point-in-time universe, the equity sleeve has R² = 0.71 on market β 0.77,
> momentum β 0.57 and low-vol β −0.50. The honest description is **efficient,
> tax-disciplined harvesting of the momentum premium, plus genuine multi-asset
> diversification** — *not* demonstrated proprietary alpha. (The t-statistic rose from
> 1.19 when a non-price signal was added in v7.7 — real movement, still short of the
> conventional 2.0 bar.)

And the second thing you should know before any performance number:

> **54% of the system's total gain came from two passive ETFs anyone can buy** (gold
> and US Nasdaq-100). The Indian stock book contributed 46%. Quoting the headline
> return as "stock picking" would be false.

Everything below is reported with those two facts in force.

---

## Headline results

**Deployed configuration · survivorship-free universe · 2016-01-01 → 2026-07-21 (10.6 years) · net of Indian tax and costs.**

| Metric | MARK6 (deployed) | Nifty 50 **TRI** B&H |
|---|---:|---:|
| Net CAGR | **+21.2%** | +10.9% |
| Volatility (annualised) | 14.9% | 14.9% |
| **Sharpe (excess of 6.5% risk-free)** | **0.97** | 0.36 |
| Sharpe (raw, rf = 0) | 1.40 | 0.79 |
| Sortino | 1.28 | 0.48 |
| Max drawdown | **−23.8%** | −36.3% |
| Calmar | **0.89** | 0.30 |
| Beta vs Nifty | 0.61 | 1.00 |
| Excess return vs Nifty 50 TRI | **+10.2pp/yr** | — |

₹5,00,000 → **₹38,17,000** over 10.6 years, net. 857 trades · 69% win rate · profit factor 3.28 · average hold 227 days.

**The equity sleeve alone** (the part this repo actually builds): +21.1% CAGR, **MaxDD −40.2%**, Calmar 0.52. It beats equal-weight of the same universe by **+9.6pp/yr** and does so in **8/8** rolling 3-year windows. It is also far too volatile to hold on its own — the multi-asset wrapper is what makes the system holdable, and that is a load-bearing fact, not a nicety.

### After-tax return depends on how much money you have

India exempts the first **₹1.25 lakh of long-term equity gain per year** (Sec 112A). That is a fixed rupee allowance, so it matters enormously to a small book and not at all to a large one. The engine now models it, applied after loss set-off and **only to listed Indian equity** — the gold and US ETF sleeves fall under different provisions and do not qualify.

| Total capital | Net CAGR | Excess Sharpe |
|---|---:|---:|
| Scale-free (the headline above) | +21.2% | 0.97 |
| **₹5,00,000 — the live paper book** | **+21.8%** | **1.01** |
| ₹25,00,000 | +21.3% | 0.98 |
| ₹5,00,00,000 (institutional) | +21.2% | 0.97 |

Both figures are correct and answer different questions. The scale-free number is the right one for institutional capacity; the capital-aware number is what a ₹5 lakh book actually experiences. **The headline understates a small book by +0.6pp — and at that size the excess Sharpe crosses 1.00.**

---

## Where the honesty is uncomfortable

These are the results that would normally be buried. They are the reason to trust the rest.

**1 · The walk-forward record against the index is mixed — 5/8, not 8/8.**

| Window | MARK6 | Nifty 50 | vs Nifty |
|---|---:|---:|---:|
| 2016–2018 | +6.6% | +10.8% | **−4.2** |
| 2017–2019 | +9.8% | +14.0% | **−4.1** |
| 2018–2020 | +6.0% | +10.1% | **−4.1** |
| 2019–2021 | +38.0% | +16.2% | +21.9 |
| 2020–2022 | +36.0% | +13.6% | +22.4 |
| 2021–2023 | +45.6% | +15.2% | +30.4 |
| 2022–2024 | +32.2% | +10.2% | +22.0 |
| 2023–2025 | +35.3% | +12.5% | +22.8 |

The system **lost to the index in the first three windows** and won the last five. That is a regime pattern, not a uniform edge. Against equal-weight of the same universe it wins 8/8 — so the *selection* is consistent; the *index-beating* is not.

**2 · It fails the standard overfitting test.**

| Test | Result | Reading |
|---|---|---|
| **Deflated Sharpe Ratio** | **96.1%** across **124 trials** | Passes the 95% bar — but only just, and it fell from 99.3% once this project honestly counted the trials it had actually run (77 → 124, luck ceiling 0.16 → 0.41). |
| **Probability of Backtest Overfitting** (CSCV) | **42.4% — FAILS the conventional <20% bar** | Picking the in-sample-best variant from this family overfits. It is why the deployed config was chosen on walk-forward consistency, never on the highest backtest number. |

A caveat neither statistic can capture: they deflate for 124 *counted* trials, not for the entire multi-year research program (ML, timing overlays, stops — all killed) that preceded this strategy family. There is no standard correction. We state it rather than hide it.

**3 · Sharpe 1.1 is unreachable, and the reason is tax.**

The ceiling was solved analytically, not searched for. At the best available weights:

```
three sleeves if perfectly uncorrelated ......  1.278
− real equity↔Nasdaq correlation (0.289) ....  −0.12  →  1.155 theoretical
− Indian tax + transaction friction .........  −0.16  →  ~0.99 measured
```

**The single largest obstacle between this book and hedge-fund Sharpe is the tax regime it operates in — not the signal, the weighting, or the assets.** An offshore or tax-exempt vehicle running the identical book would score ~1.15. That is not available to Indian retail. Tax costs the equity sleeve **2.91pp of CAGR and 0.112 of Sharpe** against a zero-tax counterfactual. The Sharpe-1.1 target is formally abandoned as unattainable rather than pursued into overfitting.

*One honest update to that verdict:* the ceiling above was computed **scale-free**. Modelling the ₹1.25L Sec 112A exemption at a ₹5 lakh book, and adding the delivery factor, lifts the measured excess Sharpe to **1.01** — so 1.0 is crossed, at small capital, for reasons that are real but **do not scale**. The exemption is worth nothing at ₹5cr and the delivery factor is provisional. The ~1.0 structural ceiling stands.

**4 · Capacity is ~₹10–25 crore, not unlimited.** Modelled with the square-root impact law on 20-day median rupee turnover: every position stays under 10% of a day's volume to ₹1cr; ≤5% of positions breach it to ₹10cr. At the ₹5cr headline, worst-case participation is 16.8% and modelled impact drag is 0.24%/yr — which the backtest does **not** charge. Beyond ₹50cr the strategy breaks down (26% of positions over the limit).

**5 · The gold sleeve's contribution is regime-conditional.** Gold earned 17.65%/yr over this sample — an exceptional decade. Force it to a normal 4% excess return, keeping its real volatility and correlations, and the best possible allocation of these three assets reaches only Sharpe 0.97. The *diversification* benefit (equity–gold correlation **0.005**) is structural and survives; the *return* contribution may not.

**6 · The live track record is 4 days old.** It is real — real prices, whole shares, real Zerodha costs, an append-only ledger — and it is far too short to mean anything. Judge it in 2027, not now.

**7 · Fat tails are real.** Daily skew −1.00, kurtosis 9.91 (normal = 3). The 99% one-day historical VaR is −2.83% against a parametric −2.05%: a normal model **understates the bad days by ~40%**. 21-day 99% CVaR is −13.4%. Worst rolling 1-year return −15.8%; 12% of rolling 1-year windows were negative.

---

**8 · The newest factor is deployed but unproven.** See below — it is labelled PROVISIONAL everywhere, including in the code.

---

## The one new data source that worked (provisionally)

Every factor this system had ever tested — momentum, trend, low-vol, stability, candlesticks, foundation models — is a **function of the same OHLCV series**, so all of them lie in the same span. That is the structural reason a decade of signal research kept failing, and why the alpha above is not significant. New alpha requires genuinely new information.

**NSE publishes `sec_bhavdata_full` free and daily**, carrying `DELIV_PER`: what share of a day's traded volume was **actually delivered to a demat account** rather than squared off intraday. Price cannot express that. It is point-in-time safe (published same day, never restated). Archive: **1,764 trading days fetched, Oct 2019 → Jul 2026** (1,678 usable after intersecting with the price panel; one day, 2022-08-08, is permanently 404 at NSE and is recorded as a known hole rather than silently skipped) — `scripts/fetch_delivery.py`.

Two signals were built and judged against a bar written **before the data arrived**:

| Signal | IC (126d) | t | Orthogonal? | Portfolio test | Verdict |
|---|---:|---:|---|---|---|
| `deliv_chg` (21d vs 126d mean delivery %) | +0.023 | 1.02 | ✅ max corr 0.19, size 0.06 | **4/4 windows**, +30.7% → **+34.8%**, Sharpe 1.06 → 1.22 | **DEPLOYED @10%, PROVISIONAL** |
| `deliv_per_z` (delivery % vs own history) | +0.020 | 0.95 | ✅ | −0.75pp, 2/4 windows | ❌ **KILLED** |

Per-window for `deliv_chg`: +5.56 / +10.13 / +4.12 / +4.59pp — all positive, **+4.76pp even excluding its best window**. Two artefact checks passed: it is **not** a disguised size tilt (correlation with log-turnover +0.057; median liquidity rank of held names moves only 152 → 160 of 300), and it is **not** one lucky config (it improves return in every variant tested).

**Why it is flagged PROVISIONAL and not a clean win — the case against, in full:**

1. **It fails the bar set in advance**, which required drawdown not to worsen. Drawdown is worse in **4/4** windows (−1 to −2pp). Deploying it was a deliberate, documented override — not a quiet goalpost move.
2. Its underlying **IC is not statistically significant** (t = 1.02).
3. The realised effect is **~4.6× larger than Grinold's law predicts** from that IC (+1.3pp expected vs ~+6pp observed). Plausibly conditional IC at the selection margin exceeding average IC across 300 names — but this is **unexplained**, and is the strongest reason to distrust it.
4. **Six years, all post-2020.** The archive begins Oct 2019, so it cannot be tested against 2016–2019 at all. The standing precedent is unfavourable: a *stronger* signal (Δ-promoter, IC +0.034) was tested the same way on a longer window and failed to convert.

It degrades safely — if the (gitignored, 130MB) archive is absent, every script falls back to the price-only book.

**Paid data was also evaluated and rejected on arithmetic**, not taste: at ₹5 lakh a *good* orthogonal signal (IC 0.05) is worth ~₹5,400/year in total, so almost no subscription can repay itself. Trendlyne's shareholding path was measured at **IC −0.025** using free data covering the same ground; Screener's fundamentals-as-tilt was falsified in 2026-06. Paid data becomes rational around ₹1–5 crore. See `scripts/data_source_breakeven.py`.

---

## Survivorship: solved, not caveated

The universe is rebuilt point-in-time from **3,064 days of NSE daily bhavcopy — 1,341 symbols, including those that stopped trading** — so delisted names are present until the day they delist. This is the fix, not an estimate.

It mattered, and mostly in a direction people do not expect:

| Universe | Candidates | CAGR | Excess Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| Old survivor-only cache | ~115 | 20.1% | 0.66 | −37.6% |
| Point-in-time, matched breadth | 150 | **14.6%** | 0.43 | −55.7% |
| Point-in-time, deployed breadth | 300 | 18.6% | 0.56 | −54.4% |

Like-for-like, survivorship was inflating returns by **~5pp/yr**, closely matching the published Indian estimate of 4.94pp (arXiv 2603.19380). **But where it really hid was risk**: the equity sleeve's true drawdown is −54% on matched breadth, not −38%.

**One finding runs the other way and is the most interesting result here:** the factor engine's edge over equal-weight *grew* on honest data, **+4.7pp → +9.6pp**. Momentum and trend systematically avoid names that die; equal-weight rides them to zero. A survivor-only backtest structurally cannot show this.

*Residual limits:* the corporate-action feed carries no demergers, so 67 symbols with unexplained post-adjustment jumps are excluded; history starts 2014, bounding the earliest window.

---

## What was tested and killed

Most hypotheses in this project **failed**, and the kill list is the point. Full decision log with evidence grades: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

❌ **Killed with out-of-sample evidence:** ML signal prediction · market-timing and regime overlays · stop-losses · circuit breakers · ex-ante multibagger picking · institutional-flow signals · leverage · volatility targeting · fundamental-quality tilts · tax-loss harvesting · frog-in-the-pan momentum · fast exit rules · correlation-aware weighting · learned/optimised asset allocation · LTCG-aware exit deferral · faster sleeve rebalancing · `deliv_per_z` · **every paid data source evaluated**.

✅ **Kept after validation:** multi-factor smart beta · fiscal-year tax netting · semi-annual momentum refresh · gold + US diversification · rank-transformed factor scoring · enforced sector caps · largest-remainder share allocation · Sec 112A exemption modelling · leakage defences.
🟡 **Deployed but PROVISIONAL:** `deliv_chg` @10% — strong portfolio evidence, insignificant IC, six years of history. Labelled as such everywhere.

**Two previously-logged "wins" were overturned by better data**, and both corrections are recorded rather than quietly dropped:
- **`n_hold=12`** was validated on the survivor-only cache and is **falsified on the honest universe** — 1/8 walk-forward windows, −5.42pp. Concentration is only safe when the universe cannot contain names that die. The deployed book holds **20**.
- **Rebalance tranching** is real (+2.7pp, 6/8 on every metric) but requires 60 whole-share slots ≈ **₹1.55cr**. It is not executable at ₹5 lakh, and the capital-efficient variant fails (MaxDD −55%, worse than baseline).

**One validated result was deliberately declined.** Risk parity (~29% equity / 45% gold / 26% US) scores Sharpe 0.99, MaxDD −17.9%, Calmar 1.11, with better drawdown in **8/8** windows — derived using only the covariance matrix, so it cannot be return-chasing. It was **not deployed** because it would cut the Indian equity sleeve to 29%, and MARK6 is intended to be an Indian stock-market system. Logged as *validated and declined*, not *falsified*.

---

## Methodological rigor

| Technique | Purpose |
|---|---|
| Point-in-time universe from NSE bhavcopy | Delisted names present until they delist — survivorship structurally removed |
| FIFO tax lots, FY loss netting (Sec 70/74), Sec 112A exemption | Tax model matches actual Indian law, verified against hand-computed cases |
| Next-close execution (`exec_lag=1`) | You cannot trade the close you just measured |
| Cash-constrained buys, stale-print force-exits | No phantom leverage; no dead names compounding at 0% |
| Net-of-tax accounting, **applied to the benchmark too** | Any reported alpha is net-to-net |
| Walk-forward (rolling 3-year), judged per metric | A risk lever judged on a return win-count is judged on the wrong axis |
| Deflated Sharpe Ratio + PBO | Is the Sharpe real, or the luckiest of many trials? |
| Factor regression on own-universe long/short factors | Is it alpha, or a disguised momentum bet? |
| Square-root market-impact model | Does the headline capital actually trade? |

**Documented approximations** (direction stated): dividends are taxed as capital gains rather than at slab (~+0.1–0.3pp strategy-favourable); the multi-asset wrapper applies a flat 15% terminal tax; modelled costs of 0.49%/round-trip *exceed* real Zerodha delivery costs (conservative); market impact is **not** charged in the backtest (subtract ~0.24%/yr at ₹5cr).

---

## Architecture

```
NSE bhavcopy ─► PIT universe ─► Causal factors ─► Portfolio constructor ─► Tax-aware backtester
  (1,341 syms,    (top-300 by      (momentum /      (inverse-vol, rank      (FIFO lots, FY netting,
   incl. dead)     turnover)        low-vol /        scores, name +          Sec 112A, next-close
                                    trend /          sector caps)            execution, costs)
                                    stability)                │
                        Multi-asset overlay ◄─────────────────┘
                        (50% equity / 25% gold / 25% US, annual rebalance)
```

| Path | Module |
|---|---|
| Causal, OHLCV-derived factor library | `core/portfolio/factors.py` |
| Point-in-time universe, benchmark, sector map | `core/portfolio/universe.py` |
| Delivery-derived factors (non-price information) | `core/portfolio/external_factors.py` |
| Portfolio construction (inverse-vol, caps, buffer) | `core/portfolio/construction.py` |
| Tax-aware walk-forward backtester | `core/portfolio/backtest.py` |
| Overfitting statistics (DSR, PBO) | `core/portfolio/stats.py` |

---

## Reproduce it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/                     # 35 tests: causality, FIFO, FY netting,
                                            # exec-lag, cash constraint, DSR/PBO sanity

# rebuild the survivorship-free universe from NSE bhavcopy (slow, ~1h)
.venv/bin/python scripts/fetch_bhavcopy.py
.venv/bin/python scripts/build_pit_cache.py

export MARK5_CACHE=data/pit_cache MARK5_TOP_N=300     # the deployed universe screen

.venv/bin/python scripts/run_mark6.py               # factor book vs EW vs Nifty TRI
.venv/bin/python scripts/institutional_report.py    # full evaluation + trade ledger
.venv/bin/python scripts/overfitting_analysis.py    # Deflated Sharpe + PBO
.venv/bin/python scripts/attribution.py             # what is skill vs what anyone can buy
.venv/bin/python scripts/risk_report.py             # VaR/CVaR, factor exposures, DD attribution
.venv/bin/python scripts/capacity_analysis.py       # how much money can this actually run
.venv/bin/python scripts/sharpe_ceiling.py          # the mathematical Sharpe ceiling
.venv/bin/python scripts/fetch_delivery.py          # free NSE delivery archive (~30 min)
.venv/bin/python scripts/delivery_signal_study.py   # IC / orthogonality / tercile test
.venv/bin/python scripts/delivery_factor_test.py    # does it earn its keep in the book?
.venv/bin/python scripts/data_source_breakeven.py   # is any PAID source worth it?
.venv/bin/python scripts/generate_portfolio.py --capital 500000   # today's holdings
```

Every quantitative claim in this README is emitted by one of these scripts. Nothing is hand-typed.

### Evidence
- [`reports/INSTITUTIONAL_REPORT.md`](reports/INSTITUTIONAL_REPORT.md) — performance, 857-trade ledger, stress tests, Monte Carlo
- [`reports/RISK_REPORT.md`](reports/RISK_REPORT.md) — VaR/CVaR, factor exposures, drawdown attribution
- [`reports/trade_ledger.csv`](reports/trade_ledger.csv) — every simulated trade, committed
- [`reports/OVERFITTING_ANALYSIS.md`](reports/OVERFITTING_ANALYSIS.md) — DSR & PBO
- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) — every hypothesis, verdict and evidence grade
- **Live paper track record:** [kavinjain.in/mark6](https://kavinjain.in/mark6) · [`data/paper/`](data/paper/) (append-only)

---

## Honest disclaimers

- **PAPER MODE ONLY.** This has never traded real money. By its own rule it must track its backtest for 6–12 months before anyone considers funding it. The live record is currently **4 days old** and means nothing yet.
- **The equity sleeve's honest max drawdown is −40%** (−54% at matched historical breadth). The multi-asset wrapper reduces the deployed system to −24%, but this book is volatile and you must be able to hold it.
- **Excess Sharpe 0.97 is strong-institutional tier, not hedge-fund tier**, and the measured unlevered ceiling under Indian tax is ~1.00. Claiming more would be dishonest.
- **The stock-selection alpha is not statistically significant** once known factors are controlled for (t = 1.61).
- **The newest factor (`deliv_chg`) is provisional**: strong portfolio evidence over six post-2020 years, an insignificant underlying IC, and an effect size larger than theory explains. The system's demonstrated strengths are factor harvesting, tax discipline and diversification.
- The edge was measured in a single decade (2016–2026) that was kind to Indian equities, gold and US tech. Regimes change, and the first three walk-forward windows show what that looks like.
- **Not investment advice, and not tax advice.** The tax modelling reflects Indian law as implemented in this engine; real-money use requires a qualified professional.

## Tech stack
Python 3.12 · NumPy / pandas / SciPy · NSE bhavcopy + corporate actions + delivery archive · yfinance (live marks) · pytest · 35-test CI

---

<div align="center">
<i>Ambition is welcome. Self-deception is fatal. When the two conflict, honesty wins — every time.</i>
</div>
