# MARK6 — Honest Smart-Beta Portfolio: Performance Report

All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and transaction costs, on the v7.1 engine (FIFO tax lots, next-close execution, cash-constrained). Benchmark = **Nifty 50 TOTAL-RETURN** buy-and-hold (dividends reinvested, via NIFTYBEES-adjusted series), net of terminal LTCG — the strategy book earns dividends, so a price-only index would flatter it ~1pp/yr.

## Headline windows

| Window | MARK6 net CAGR | EqualWeight | Nifty50 TRI B&H | vs Nifty | vs EW |
|---|---|---|---|---|---|
| FULL 2016-2026 | +20.7% | +12.1% | +11.1% | +9.7pp | +8.6pp |
| OOS-era 2016-2021 | +19.5% | +10.9% | +13.8% | +5.8pp | +8.6pp |
| recent 2022-2026 | +17.9% | +13.0% | +6.8% | +11.2pp | +4.9pp |

## Rolling 3-year walk-forward

**Beats Nifty50 in 5/8 windows; beats EqualWeight in 7/8 windows.**

| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |
|---|---|---|---|---|
| 2016-2018 | +3.9% | +5.7% | +10.8% | -6.9pp |
| 2017-2019 | +9.2% | -1.3% | +14.0% | -4.8pp |
| 2018-2020 | +8.3% | -6.8% | +10.1% | -1.8pp |
| 2019-2021 | +34.0% | +16.5% | +16.2% | +17.9pp |
| 2020-2022 | +37.9% | +21.4% | +13.6% | +24.4pp |
| 2021-2023 | +41.0% | +27.6% | +15.2% | +25.8pp |
| 2022-2024 | +29.1% | +20.6% | +10.2% | +18.8pp |
| 2023-2025 | +21.4% | +17.1% | +12.5% | +9.0pp |

## Honest caveats

- Survivorship: none. The universe is point-in-time (1341 symbols from `data/pit_cache`), and **185** of them delisted inside the window — they are held until the day they stop trading, so their failure is priced in. The concentrated momentum book has still NOT been separately failure-injected by `survivorship_validation.py`.
- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces but cannot eliminate them. The 5% hard-stop design is incompatible with equity returns and was proven to destroy the edge.
- The edge over the cap-weighted index is real but regime-dependent; it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax).
