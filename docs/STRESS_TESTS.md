# Stress tests — how this book behaved in every crash since 2007

Source: `reports/stress_test.json`, `reports/drawdown_research.json`,
`reports/currency_stress.json`, `reports/capacity_analysis.json`.
Window **2007-09-17 → 2026-07-21** (19 years), survivorship-free universe, net of
Indian tax and costs.

---

## Read this before any number below

**The allocation stress-tested here is 30% equity / 30% gold / 10% US / 30% bond.
That is NOT the deployed book.** The live book is 50/25/25 (equity/gold/US) and
January's config is four equal 25% sleeves. The tested mix is *more defensive
than what is deployed*, so every crisis figure below flatters the live book.

Stated plainly because it is the kind of caveat that gets dropped in a summary
and turns a research artifact into a marketing one. The deployed 50/25/25 book's
own worst dip over the same 19 years is **−41.80%**, against the −17.83% shown
below for the defensive mix. Both are real; they describe different portfolios.

---

## Every crisis in the window

| Episode | This book | Nifty 50 |
|---|---:|---:|
| 2008 Global Financial Crisis | **−1.6%** | −50.8% |
| 2011 Euro debt / India inflation | **+24.1%** | −24.4% |
| 2013 Taper Tantrum (INR −20%) | +2.4% | −4.4% |
| 2015–16 China devaluation / oil crash | +3.0% | −18.6% |
| 2018 IL&FS / midcap collapse | +0.2% | +3.4% |
| 2020 COVID crash | +6.3% | −15.8% |
| **2022 inflation / rate shock** | **−11.3%** | **+2.2%** |

**The row that matters is the last one.** 2022 is the only episode where the book
lost while the index gained, and it is the honest description of what this
structure costs: gold and bonds are insurance, and insurance has a premium. In a
rate shock, bonds fall with equities and the diversification stops working — that
is precisely the scenario it does not cover.

Anyone shown the 2008 row without the 2022 row has been sold something.

---

## Drawdown

| | This book | Nifty 50 |
|---|---:|---:|
| Worst peak-to-trough | **−17.83%** | −59.86% |
| Days underwater | 397 | 727 |

Deployed 50/25/25 over the same window: **−41.80%**. On ₹5 lakh that is ₹2.09
lakh gone, temporarily, and it has happened.

---

## Rolling windows — the question "what if I'd started at the worst moment"

| Hold for | Worst | Median | Best | % positive |
|---|---:|---:|---:|---:|
| 1 year | −9.7% | +16.5% | +41.8% | 90.9% |
| 3 years | **+8.0%** | +15.3% | +26.2% | **100%** |
| 5 years | **+10.5%** | +15.2% | +21.2% | **100%** |

Nifty's worst 1-year over the same period was **−53.4%**.

No 3-year or 5-year window lost money — but note what that does *not* say. 19
years contains about 6 independent 3-year windows, so "100%" rests on a handful
of observations, not on a law. It is evidence that holding period matters, not a
guarantee that it always will.

**Worst possible entry date**: someone who bought on 2015-01-27 still made
**+8.5%/yr** over the following five years. The equivalent worst entry into the
Nifty (2008-01-04) returned **−1.9%/yr**.

---

## Year by year

| Year | Book | Nifty |
|---|---:|---:|
| 2007 | +4.8% | +36.6% |
| 2008 | **−2.9%** | **−51.8%** |
| 2009 | +14.4% | +70.7% |
| 2010 | +16.7% | +17.2% |
| 2011 | +23.7% | −24.9% |
| 2012 | +16.1% | +23.9% |
| 2013 | −0.9% | +5.2% |
| 2014 | +31.7% | +33.1% |

2009 is the mirror image of 2008 and belongs beside it: the structure that lost
almost nothing in the crash also captured only a fifth of the rebound. That is
the same property, not a separate flaw.

---

## Full-period summary (defensive allocation)

| | |
|---|---:|
| CAGR | +14.81% |
| Volatility | 11.27% |
| Sharpe | 1.32 |
| Max drawdown | −17.83% |
| Calmar | 0.83 |

---

## What is NOT stress-tested here

Named because an untested scenario silently reads as a passed one.

- **A rupee crisis with capital controls.** `reports/currency_stress.json` shows
  the book is structurally short the rupee (75–80% of assets are USD- or
  gold-denominated). That helped in every episode above. A forced repatriation or
  an LRS restriction is a scenario this cannot model.
- **A liquidity event in the equity sleeve.** `reports/capacity_analysis.json`
  covers ordinary capacity; it does not model a market where the bid disappears.
- **Broker or custodian failure.** Outside the model entirely.
- **The strategy itself decaying.** Every figure here assumes the momentum
  premium keeps existing. The live record is 18 days long.

---

## Why the numbers are believable, and where they are not

**Believable:** point-in-time universe with delisted names held to their last
print; net of Indian tax at statutory rates with FY loss netting; costs set above
real broker rates; next-close execution.

**Not proven:** the deflated Sharpe passes at 96.1% across 124 counted trials —
just over the 95% bar, and it does not deflate for the multi-year research
programme that preceded this strategy family. The probability of backtest
overfitting is **42.4%, which fails the conventional 20% bar**. The equity
sleeve's stock-selection alpha is +5.8%/yr at **t = 1.61 — not statistically
significant**.

Those three facts are in `README.md` under "Where the honesty is uncomfortable"
and are repeated here because a stress-test document is exactly where a reader
is most inclined to stop being sceptical.
