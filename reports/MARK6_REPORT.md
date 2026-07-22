# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs, on the v7.1 engine (FIFO tax lots, next-close execution, cash-constrained). Benchmark = **Nifty 50 TOTAL-RETURN** buy-and-hold (dividends reinvested, via NIFTYBEES-adjusted series), net of terminal LTCG — the strategy book earns dividends, so a price-only index would flatter it ~1pp/yr.

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 TRI B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +18.2% | +11.8% | +11.1% | +7.2pp | +6.4pp |
| OOS-era 2016-2021 | +20.8% | +10.9% | +13.8% | +7.0pp | +9.9pp |
| recent 2022-2026 | +10.7% | +12.5% | +6.8% | +3.9pp | -1.8pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 5/8 windows; beats EqualWeight in 6/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | +3.5% | +5.7% | +10.8% | -7.3pp |
| 2017-2019 | +7.3% | -1.4% | +14.0% | -6.7pp |
| 2018-2020 | +9.0% | -6.8% | +10.1% | -1.2pp |
| 2019-2021 | +31.1% | +16.5% | +16.2% | +14.9pp |
| 2020-2022 | +23.8% | +21.5% | +13.6% | +10.2pp |
| 2021-2023 | +32.8% | +27.5% | +15.2% | +17.6pp |
| 2022-2024 | +23.3% | +20.7% | +10.2% | +13.1pp |
| 2023-2025 | +16.0% | +17.1% | +12.5% | +3.5pp |

## Honest caveats

- Survivorship: the candidate universe is today's surviving constituents (fully-delisted names absent), so headline CAGR is inflated an estimated ~1-2pp/yr. `survivorship_validation.py` bounds this via failure injection on the equal-weight basket; the concentrated momentum book has NOT been separately failure-injected.
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
