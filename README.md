<div align="center">

# MARK5 / MARK6 — Honest Quantitative Equity Research

**A research-grade, survivorship- and tax-aware quantitative portfolio system for NSE Indian equities.**

*Built not to claim an edge, but to find out — rigorously — whether one exists.*

`PAPER MODE ONLY` · `Net of Indian tax & costs` · `Out-of-sample validated` · `Adversarially audited`

[![CI](https://github.com/kavin-jain/MARK5/actions/workflows/ci.yml/badge.svg)](https://github.com/kavin-jain/MARK5/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20619267-blue)](https://doi.org/10.5281/zenodo.20619267)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Methodology: DSR+PBO](https://img.shields.io/badge/Validated-DSR%20%2B%20PBO-green)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)

</div>

---

## TL;DR

A months-long, end-to-end quantitative research program that asked one question — *can a retail investor, with public data, beat the Indian market?* — and answered it **honestly**, with out-of-sample evidence, instead of with an overfit backtest.

The conclusion is deliberately humble: **you cannot reliably beat same-universe buy-and-hold by prediction or timing — but you _can_ build a disciplined, diversified, tax-efficient portfolio that beats the cap-weighted index.** That system is the deliverable.

### Headline results — deployed system, v7.1 engine (net of tax, 2016–2026)

| Metric | MARK6 (deployed) | Nifty 50 **TRI** B&H |
|---|---:|---:|
| Net CAGR | **+20.7%** | +11.1% |
| Sharpe (raw, rf=0) | 0.96 | 0.79 |
| **Sharpe (excess of 6.5% risk-free)** | **0.68** | 0.37 |
| Max drawdown | **−26.5%** | −36.3% |
| Calmar | **0.78** | 0.30 |
| Excess return vs Nifty 50 TRI | **+9.6pp/yr** | — |
| Beta (defensive) | 0.67 | 1.00 |
| Equity-sleeve walk-forward | **beats Nifty TRI 7/8, equal-weight 8/8** rolling 3-yr windows | — |

> **Benchmark honesty:** the benchmark is Nifty 50 **total-return** (dividends reinvested, real NIFTYBEES-adjusted series), taxed at terminal LTCG exactly like the strategy. The strategy book runs on dividend-adjusted prices, so a price-only index — which most retail backtests use — would flatter the excess by ~1pp/yr. We don't.
>
> **What drives the +9.6pp:** the factor engine (momentum-heavy ranking + 6-month refresh under fiscal-year tax netting) contributes **+4.7pp/yr over equal-weight buy-and-hold of the same universe** (computed, not asserted — see `reports/INSTITUTIONAL_REPORT.md`); the remainder is deliberate multi-asset allocation (gold + US Nasdaq diversification) that any passive multi-asset fund also captures. Calling the whole gap "alpha" would be misleading.
>
> **Survivorship caveat (subtract before believing):** universe *eligibility* is point-in-time, but the candidate list is today's surviving index constituents — fully-delisted names are absent. Estimated inflation **~1–2pp/yr**; the honest forward expectation is **~18–19% CAGR over a full cycle**, single years anywhere from −15% to +40%.

**The deployed portfolio:** 50% concentrated 12-stock momentum-heavy factor book (refreshed every 6 months under FY tax netting) · 25% gold (GOLDBEES) · 25% US Nasdaq-100 (MON100) — three near-uncorrelated sleeves, rebalanced annually.

---

## What makes this different

Most student/retail quant projects show one flattering backtest and stop. This is a **scientific program**: every hypothesis was tested out-of-sample, net of tax, against the honest benchmark — and **most were killed**.

- ❌ **Killed** (with evidence): ML signal prediction, momentum-timing overlays, stop-losses, circuit-breakers, ex-ante multibagger picking, institutional-flow signals, leverage, volatility-targeting, fundamental-quality tilts, tax-loss harvesting, frog-in-the-pan momentum, fast exit rules.
- ✅ **Kept** (validated OOS): multi-factor smart-beta, concentration, fiscal-year tax netting + semi-annual momentum refresh, gold + US diversification, leakage defences.
- 🔍 **Adversarially audited** (2026-07-22, 16-agent black-box audit): the audit *found and we fixed* — a price-index benchmark that flattered the excess ~1pp/yr (now TRI), same-close execution (now next-close), average-cost tax lots (now statutory FIFO), a phantom cost-free overdraft (buys now cash-constrained), suspended names compounding at 0% (now haircut + force-exited), and Sharpe quoted without a risk-free rate (both now reported). **The headline moved only 20.8% → 20.7%** — the edge never lived in the biased parts. Full decision log: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

---

## Statistical validation (the tests most retail backtests never run)

| Test | Result | Honest reading |
|---|---|---|
| **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) | **99.3%** across **77 trials** | The deployed Sharpe survives multiple-testing correction — it is not the luckiest of the 77 variants tried (luck ceiling: Sharpe 0.16). PSR vs 0 = 99.8%. |
| **Probability of Backtest Overfitting** (CSCV, Bailey et al. 2017) | **76.7% — FAILS the conventional <20% bar** | An honest red flag, honestly reported: picking the *in-sample-best* variant from this family overfits, because near-identical smart-beta configs are statistically indistinguishable. **That is exactly why the deployed config was chosen on walk-forward consistency (7/8 windows), not on the highest full-period number** — the in-sample winner (21-day rebalancing, +22.6% full-period) fails out-of-sample (3/8). |

Both computed by [`scripts/overfitting_analysis.py`](scripts/overfitting_analysis.py) → [`reports/OVERFITTING_ANALYSIS.md`](reports/OVERFITTING_ANALYSIS.md). A caveat the DSR cannot capture: it deflates for 77 counted trials, not for the entire multi-year research program (ML, overlays, stops — all killed) that preceded this strategy family. There is no standard correction for that; we state it instead of hiding it.

---

## Methodological rigor

| Technique | Purpose |
|---|---|
| Point-in-time eligibility + pinned universe | No look-ahead in membership; reproducible candidate list |
| FIFO tax lots, FY loss netting (Sec 70/74) | Tax model matches actual Indian law, verified by engineered-trade tests |
| Next-close execution (`exec_lag=1`) | You cannot trade the close you just measured |
| Cash-constrained buys, stale-print force-exits | No phantom leverage; no dead names compounding at 0% |
| Net-of-tax accounting (LTCG 12.5% / STCG 20%) | Returns reflect the real Indian tax drag — on the benchmark too |
| Walk-forward (rolling 3-yr) validation | Robustness, not single-window cherry-picking |
| Deflated Sharpe Ratio + PBO (Bailey & López de Prado) | Is the Sharpe real, or the luckiest of many trials? |
| Monte Carlo (block-bootstrap) | Stress against unpredicted-event paths |

**Documented approximations** (direction stated, none flatter the headline materially): dividends are taxed as capital gains rather than at slab (~+0.1–0.3pp strategy-favorable); the ₹1.25L LTCG exemption is not modelled (conservative); the multi-asset wrapper applies a flat 15% terminal tax (approximation); modelled costs of 0.49%/round-trip *exceed* real Zerodha delivery costs (conservative).

---

## Architecture

```
Historical OHLCV ─► Causal Factor Library ─► Point-in-Time Universe ─► Portfolio Constructor ─► Tax-Aware Backtester
   (yfinance,           (momentum / low-vol      (pinned candidate list,  (inverse-vol + tilt,     (FIFO lots, FY netting,
    NSE XBRL)            / trend / stability)     liquidity-screened)      weight caps)             next-close exec, costs)
                                                                                   │
                              Multi-asset overlay  ◄───────────────────────────────┘
                              (equity + gold + US, annual rebalance)
```

| Path | Module |
|---|---|
| Factor library (causal, OHLCV-derived) | `core/portfolio/factors.py` |
| Point-in-time universe & benchmark data | `core/portfolio/universe.py` |
| Portfolio construction (inverse-vol, caps, buffer) | `core/portfolio/construction.py` |
| Tax-aware walk-forward backtester | `core/portfolio/backtest.py` |
| Overfitting statistics (DSR, PBO) | `core/portfolio/stats.py` |

---

## Reproduce it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/                        # 32 tests: causality, FIFO, FY netting,
                                               # exec-lag, cash constraint, DSR/PBO sanity

.venv/bin/python scripts/refetch_all.py        # rebuild data cache from the pinned
                                               # universe (config/universe_tickers.json)
.venv/bin/python scripts/run_mark6.py                    # factor book vs EW vs Nifty TRI
.venv/bin/python scripts/institutional_report.py         # full evaluation + trade ledger
.venv/bin/python scripts/overfitting_analysis.py         # Deflated Sharpe + PBO
.venv/bin/python scripts/system_vs_buyhold_final.py      # does the factor engine earn its keep?
.venv/bin/python scripts/generate_portfolio.py --capital 500000   # today's holdings
.venv/bin/python scripts/paper_track.py init --capital 500000     # start the paper track record
```

Prices come from yfinance, so a fresh clone reproduces the *methodology* exactly and the *numbers* to within data-vendor revisions. Every quantitative claim in this README is emitted by one of the scripts above — nothing is hand-typed.

### Evidence
- [`reports/INSTITUTIONAL_REPORT.md`](reports/INSTITUTIONAL_REPORT.md) — performance, 489-trade ledger (72% win rate, profit factor 3.27), stress tests, Monte Carlo, industry scorecard
- [`reports/trade_ledger.csv`](reports/trade_ledger.csv) — every trade, committed to the repo
- [`reports/MARK6_REPORT.md`](reports/MARK6_REPORT.md) — equity sleeve vs benchmarks, walk-forward table
- [`reports/OVERFITTING_ANALYSIS.md`](reports/OVERFITTING_ANALYSIS.md) — DSR & PBO
- Knowledge base: [`docs/KNOWLEDGE_BASE.md`](docs/KNOWLEDGE_BASE.md) · [`docs/MARKET_PLAYBOOK.md`](docs/MARKET_PLAYBOOK.md) · [`docs/MARK6_SMART_BETA.md`](docs/MARK6_SMART_BETA.md)

---

## Honest disclaimers

- **PAPER MODE ONLY.** This is a research system, not investment advice. It has never traded real money — and by its own rule it must first track its backtest through 6–12 months of `paper_track.py` before anyone considers funding it.
- Survivorship-adjusted expectation is **~18–19% CAGR**, not the +20.7% headline.
- Drawdowns of **−25% to −35% are real and unavoidable** in any long-only equity book. Abandoning the system mid-drawdown converts the entire edge into a loss.
- The excess Sharpe of 0.68 is strong-mutual-fund tier, **not** hedge-fund tier. Sharpe ≳ 1 (excess) requires leverage and infrastructure unavailable to an Indian retail investor; claiming otherwise would be dishonest.
- The edge was measured in a single decade (2016–2026) that was kind to Indian equities, gold, and US tech. Regimes change.

## Tech stack
Python 3.12 · NumPy / pandas / SciPy · yfinance + official NSE/BSE XBRL data · pytest · 32-test CI

---

<div align="center">
<i>Ambition is welcome. Self-deception is fatal. When the two conflict, honesty wins — every time.</i>
</div>
