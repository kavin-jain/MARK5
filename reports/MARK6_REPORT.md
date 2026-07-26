# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs, on the v7.1 engine (FIFO tax lots, next-close execution, cash-constrained). Benchmark = **Nifty 50 TOTAL-RETURN** buy-and-hold (dividends reinvested, via NIFTYBEES-adjusted series), net of terminal LTCG — the strategy book earns dividends, so a price-only index would flatter it ~1pp/yr.

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 TRI B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +21.1% | +11.4% | +11.1% | +10.0pp | +9.6pp |
| OOS-era 2016-2021 | +19.4% | +9.3% | +13.8% | +5.7pp | +10.1pp |
| recent 2022-2026 | +20.2% | +13.2% | +6.8% | +13.5pp | +7.0pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 5/8 windows; beats EqualWeight in 8/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | +6.6% | +4.9% | +10.8% | -4.2pp |
| 2017-2019 | +9.8% | -0.0% | +14.0% | -4.1pp |
| 2018-2020 | +6.0% | -6.0% | +10.1% | -4.1pp |
| 2019-2021 | +38.0% | +14.5% | +16.2% | +21.9pp |
| 2020-2022 | +36.0% | +16.2% | +13.6% | +22.4pp |
| 2021-2023 | +45.6% | +23.8% | +15.2% | +30.4pp |
| 2022-2024 | +32.2% | +18.4% | +10.2% | +22.0pp |
| 2023-2025 | +35.3% | +19.8% | +12.5% | +22.8pp |

## Honest caveats

- Survivorship: the candidate universe is today's surviving constituents (fully-delisted names absent), so headline CAGR is inflated an estimated ~1-2pp/yr. `survivorship_validation.py` bounds this via failure injection on the equal-weight basket; the concentrated momentum book has NOT been separately failure-injected.
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
