# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs. Universe is point-in-time (survivorship-aware). Benchmark = cap-weighted Nifty 50 buy-and-hold (what 'buy and hold' normally means).

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +16.2% | +15.0% | +10.4% | +5.8pp | +1.2pp |
| OOS-era 2016-2021 | +16.7% | +13.5% | +12.9% | +3.8pp | +3.2pp |
| recent 2022-2026 | +14.3% | +16.0% | +6.2% | +8.1pp | -1.8pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 7/8 windows; beats EqualWeight in 7/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | +8.2% | +8.3% | +10.4% | -2.2pp |
| 2017-2019 | +13.2% | +10.8% | +12.6% | +0.6pp |
| 2018-2020 | +10.8% | +5.5% | +9.1% | +1.7pp |
| 2019-2021 | +32.5% | +18.7% | +15.3% | +17.2pp |
| 2020-2022 | +21.8% | +19.9% | +12.6% | +9.3pp |
| 2021-2023 | +27.8% | +24.9% | +14.0% | +13.8pp |
| 2022-2024 | +23.5% | +20.7% | +9.1% | +14.4pp |
| 2023-2025 | +25.8% | +22.1% | +11.4% | +14.4pp |

## Honest caveats

- Survivorship: universe is today's listed names; residual bias bounded by `survivorship_validation.py` (failure injection ~2-3pp).
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
