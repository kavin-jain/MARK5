# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs, on the v7.1 engine (FIFO tax lots, next-close execution, cash-constrained). Benchmark = **Nifty 50 TOTAL-RETURN** buy-and-hold (dividends reinvested, via NIFTYBEES-adjusted series), net of terminal LTCG — the strategy book earns dividends, so a price-only index would flatter it ~1pp/yr.

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 TRI B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +12.6% | +9.5% | +11.1% | +1.5pp | +3.1pp |
| OOS-era 2016-2021 | +6.3% | +7.2% | +13.8% | -7.4pp | -0.9pp |
| recent 2022-2026 | +12.2% | +11.8% | +6.8% | +5.4pp | +0.3pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 5/8 windows; beats EqualWeight in 7/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | -1.9% | +3.1% | +10.8% | -12.8pp |
| 2017-2019 | -1.8% | -5.5% | +14.0% | -15.8pp |
| 2018-2020 | -3.8% | -4.0% | +10.1% | -13.9pp |
| 2019-2021 | +16.4% | +11.4% | +16.2% | +0.3pp |
| 2020-2022 | +21.5% | +14.1% | +13.6% | +8.0pp |
| 2021-2023 | +33.0% | +21.0% | +15.2% | +17.8pp |
| 2022-2024 | +21.7% | +18.2% | +10.2% | +11.5pp |
| 2023-2025 | +20.8% | +17.4% | +12.5% | +8.3pp |

## Honest caveats

- Survivorship: the candidate universe is today's surviving constituents (fully-delisted names absent), so headline CAGR is inflated an estimated ~1-2pp/yr. `survivorship_validation.py` bounds this via failure injection on the equal-weight basket; the concentrated momentum book has NOT been separately failure-injected.
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
