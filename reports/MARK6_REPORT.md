# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs, on the v7.1 engine (FIFO tax lots, next-close execution, cash-constrained). Benchmark = **Nifty 50 TOTAL-RETURN** buy-and-hold (dividends reinvested, via NIFTYBEES-adjusted series), net of terminal LTCG — the strategy book earns dividends, so a price-only index would flatter it ~1pp/yr.

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 TRI B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +20.0% | +15.4% | +11.1% | +9.0pp | +4.7pp |
| OOS-era 2016-2021 | +17.5% | +14.4% | +13.8% | +3.7pp | +3.1pp |
| recent 2022-2026 | +23.1% | +16.5% | +6.8% | +16.4pp | +6.6pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 7/8 windows; beats EqualWeight in 8/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | +10.6% | +9.5% | +10.8% | -0.2pp |
| 2017-2019 | +16.0% | +9.9% | +14.0% | +2.0pp |
| 2018-2020 | +12.2% | +6.0% | +10.1% | +2.1pp |
| 2019-2021 | +29.3% | +19.6% | +16.2% | +13.1pp |
| 2020-2022 | +27.1% | +20.5% | +13.6% | +13.5pp |
| 2021-2023 | +36.4% | +25.3% | +15.2% | +21.2pp |
| 2022-2024 | +31.6% | +22.1% | +10.2% | +21.4pp |
| 2023-2025 | +33.4% | +23.5% | +12.5% | +20.9pp |

## Honest caveats

- Survivorship: the candidate universe is today's surviving constituents (fully-delisted names absent), so headline CAGR is inflated an estimated ~1-2pp/yr. `survivorship_validation.py` bounds this via failure injection on the equal-weight basket; the concentrated momentum book has NOT been separately failure-injected.
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
