# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs, on the v7.1 engine (FIFO tax lots, next-close execution, cash-constrained). Benchmark = **Nifty 50 TOTAL-RETURN** buy-and-hold (dividends reinvested, via NIFTYBEES-adjusted series), net of terminal LTCG — the strategy book earns dividends, so a price-only index would flatter it ~1pp/yr.

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 TRI B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +22.3% | +12.3% | +11.1% | +11.3pp | +10.1pp |
| OOS-era 2016-2021 | +19.3% | +10.5% | +13.8% | +5.5pp | +8.7pp |
| recent 2022-2026 | +18.3% | +13.9% | +6.8% | +11.5pp | +4.3pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 5/8 windows; beats EqualWeight in 8/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | +7.2% | +5.7% | +10.8% | -3.6pp |
| 2017-2019 | +9.3% | +1.0% | +14.0% | -4.7pp |
| 2018-2020 | +8.3% | -5.1% | +10.1% | -1.8pp |
| 2019-2021 | +36.6% | +15.1% | +16.2% | +20.4pp |
| 2020-2022 | +36.4% | +17.3% | +13.6% | +22.8pp |
| 2021-2023 | +46.6% | +24.4% | +15.2% | +31.4pp |
| 2022-2024 | +29.0% | +19.5% | +10.2% | +18.8pp |
| 2023-2025 | +30.6% | +20.3% | +12.5% | +18.1pp |

## Honest caveats

- Survivorship: none. The universe is point-in-time (1337 symbols from `data/pit_cache`), and **180** of them delisted inside the window — they are held until the day they stop trading, so their failure is priced in. The concentrated momentum book has still NOT been separately failure-injected by `survivorship_validation.py`.
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
