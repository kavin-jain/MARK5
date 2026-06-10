<div align="center">

# MARK5 / MARK6 — Honest Quantitative Equity Research

**A research-grade, survivorship- and tax-aware quantitative portfolio system for NSE Indian equities.**

*Built not to claim an edge, but to find out — rigorously — whether one exists.*

`PAPER MODE ONLY` · `Net of Indian tax & costs` · `Out-of-sample validated`

[![CI](https://github.com/kavin-jain/MARK5/actions/workflows/ci.yml/badge.svg)](https://github.com/kavin-jain/MARK5/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Methodology: DSR+PBO](https://img.shields.io/badge/Validated-DSR%20%2B%20PBO-green)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)

</div>

---

## TL;DR

This is a months-long, end-to-end quantitative research program that asked a simple question — *can a retail investor, with public data, beat the Indian market?* — and answered it **honestly**, with out-of-sample evidence, instead of with an overfit backtest.

The conclusion is deliberately humble and fully evidenced: **you cannot reliably beat same-universe buy-and-hold by prediction or timing — but you _can_ build a disciplined, diversified, tax-efficient portfolio that beats the cap-weighted index on a risk-adjusted basis.** That system is the deliverable.

### Headline results — deployed system (net of tax, 2016–2026)

| Metric | MARK6 (deployed) | Nifty 50 B&H |
|---|---:|---:|
| Net CAGR | **+17.3%** | +11.1% |
| Sharpe | **0.99** | 0.74 |
| Max drawdown | **−28%** | −38% |
| Annualised alpha vs Nifty | **+9.7%/yr** | — |
| Beta (defensive) | 0.68 | 1.00 |
| Walk-forward hit-rate | **beats Nifty 7/8 windows** | — |

**Statistical honesty:** Deflated Sharpe Ratio = **99%** (the edge survives multiple-testing); Probability of Backtest Overfitting (PBO) flagged that *fine-tuning the weights* is noise — so the system deploys a robust blend, not an in-sample-optimal one. *(These are the tests professional quant funds use and most retail backtests never compute.)*

> **The deployed portfolio:** 70% concentrated 12-stock factor book · 15% gold (GOLDBEES) · 15% US Nasdaq-100 (MON100) — three near-uncorrelated sleeves, annual rebalance.

---

## What makes this different

Most student / retail quant projects show one flattering backtest and stop. This one is a **scientific program**: every hypothesis was tested out-of-sample, net of tax, against the honest benchmark — and **most were killed**. The value is in the rigor and the truth, not a get-rich claim.

- ✅ **Killed** (with evidence): ML signal prediction, momentum-timing overlays, stop-losses, circuit-breakers, ex-ante multibagger picking, institutional-flow signals, leverage, volatility-targeting, fundamental-quality tilts.
- ✅ **Kept** (validated OOS): multi-factor smart-beta, concentration, annual-rebalance tax discipline, gold + US diversification, leakage defences.
- ✅ **Found & fixed real bugs**: a backtest warm-up bug that was *understating* performance; ETF data contamination of the equity universe.

Full decision log: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

---

## Methodological rigor

| Technique | Purpose |
|---|---|
| Purged Combinatorial CV (CPCV) + embargo | Zero look-ahead in model validation |
| Point-in-time universe + failure injection | Survivorship-bias control |
| Net-of-tax accounting (LTCG 12.5% / STCG 20%) | Returns reflect the real Indian tax drag |
| Walk-forward (rolling 3-yr) validation | Robustness, not single-window cherry-picking |
| Deflated Sharpe Ratio + PBO (Bailey & López de Prado) | Is the Sharpe real, or the luckiest of many trials? |
| Random Matrix Theory + Minimum Spanning Tree | Econophysics view of *true* diversification |
| Monte Carlo (block-bootstrap) | Stress against unpredicted-event paths |

---

## Architecture

```
Historical OHLCV ─► Causal Factor Library ─► Point-in-Time Universe ─► Portfolio Constructor ─► Tax-Aware Backtester
   (yfinance,           (momentum / low-vol      (survivorship-aware,    (inverse-vol + tilt,     (LTCG/STCG lots,
    NSE XBRL)            / trend / stability)     liquidity-screened)      sector/weight caps)      costs, slippage)
                                                                                   │
                              Multi-asset overlay  ◄───────────────────────────────┘
                              (equity + gold + US, annual rebalance)
```

| Path | Module |
|---|---|
| Factor library (causal, OHLCV-derived) | `core/portfolio/factors.py` |
| Point-in-time universe & eligibility | `core/portfolio/universe.py` |
| Portfolio construction (inverse-vol, caps, buffer) | `core/portfolio/construction.py` |
| Tax-aware walk-forward backtester | `core/portfolio/backtest.py` |
| Overfitting statistics (DSR, PBO) | `core/portfolio/stats.py` |
| External / fundamental factors | `core/portfolio/external_factors.py`, `fundamentals.py` |

---

## Reproduce it

```bash
bash setup.sh                 # environment (Python 3.12, deps)
pytest tests/                 # 22 portfolio tests pass

python3 scripts/run_mark6.py                  # core smart-beta vs benchmarks
python3 scripts/multiasset_v2_test.py         # 3-sleeve diversification (Sharpe ~1.0)
python3 scripts/institutional_report.py       # full evaluation + trade ledger
python3 scripts/overfitting_analysis.py       # Deflated Sharpe + PBO
python3 scripts/market_network.py             # RMT / network econophysics
python3 scripts/generate_portfolio.py --capital 500000   # today's holdings
```

### Reports (evidence)
- [`reports/INSTITUTIONAL_REPORT.md`](reports/INSTITUTIONAL_REPORT.md) — full performance, trade ledger, stress tests, Monte Carlo, industry scorecard
- [`reports/OVERFITTING_ANALYSIS.md`](reports/OVERFITTING_ANALYSIS.md) — Deflated Sharpe & PBO
- [`reports/MARKET_NETWORK.md`](reports/MARKET_NETWORK.md) — RMT eigenstructure & market skeleton
- [`reports/HOLDING_PERIOD_ANALYSIS.md`](reports/HOLDING_PERIOD_ANALYSIS.md) — why long holds win net of tax
- Knowledge base: [`docs/KNOWLEDGE_BASE.md`](docs/KNOWLEDGE_BASE.md) · [`docs/MARKET_PLAYBOOK.md`](docs/MARKET_PLAYBOOK.md)

---

## Statistical validation

Most retail backtests report a Sharpe ratio without asking whether it's real or just the luckiest result of many trials. This system applies the two tests the professional quant literature uses to answer that question.

| Test | Result | What it means |
|---|---|---|
| **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) | **DSR = 99%** | After correcting for 60 strategy variants tested, skewness, and kurtosis — the edge has a 99% probability of being genuine, not a statistical artefact |
| **Probability of Backtest Overfitting** (Bailey et al., 2017) | **PBO = 75.6%** | The optimal *fine-tuned* config beats the median OOS in only 24.4% of train/test splits — so the system deploys a robust blend, not an in-sample-optimal one |

> These are the same tests used by institutional quant funds. Most retail backtests never compute them.

**References**
- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management. [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- Bailey, D.H., Borwein, J.M., López de Prado, M. & Zhu, Q.J. (2017). *The Probability of Backtest Overfitting.* Journal of Computational Finance. [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

---

## Honest disclaimers

- **PAPER MODE ONLY.** This is a research system. It is **not** investment advice, and it has never traded real money.
- Backtested returns are **survivorship-caveated** — the true sustainable figure is ~2–3pp below the headline.
- Drawdowns of **−28% to −35% are real and unavoidable** in any long-only equity book.
- This does **not** achieve "20%+ / Sharpe 2" — those require leverage and infrastructure unavailable to a retail investor, and saying otherwise would be dishonest.

## Tech stack
Python 3.12 · NumPy / pandas / SciPy · scikit-learn / XGBoost / LightGBM / CatBoost (research) · yfinance + official NSE/BSE XBRL data · pytest

---

<div align="center">
<i>Ambition is welcome. Self-deception is fatal. When the two conflict, honesty wins — every time.</i>
</div>
