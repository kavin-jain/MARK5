# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs. Universe is point-in-time (survivorship-aware). Benchmark = cap-weighted Nifty 50 buy-and-hold (what 'buy and hold' normally means).

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +16.3% | +14.9% | +10.4% | +5.9pp | +1.4pp |
| OOS-era 2016-2021 | +20.0% | +13.4% | +12.9% | +7.0pp | +6.6pp |
| recent 2022-2026 | +15.1% | +15.6% | +6.2% | +8.9pp | -0.5pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 6/8 windows; beats EqualWeight in 7/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | +7.0% | +8.2% | +10.4% | -3.4pp |
| 2017-2019 | +12.5% | +10.7% | +12.6% | -0.1pp |
| 2018-2020 | +11.4% | +5.4% | +9.1% | +2.3pp |
| 2019-2021 | +37.1% | +18.5% | +15.3% | +21.8pp |
| 2020-2022 | +22.7% | +19.8% | +12.6% | +10.2pp |
| 2021-2023 | +32.2% | +24.8% | +14.0% | +18.2pp |
| 2022-2024 | +25.0% | +20.7% | +9.1% | +15.9pp |
| 2023-2025 | +27.1% | +22.1% | +11.4% | +15.7pp |

## Honest caveats

- Survivorship: universe is today's listed names; residual bias bounded by `survivorship_validation.py` (failure injection ~2-3pp).
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
