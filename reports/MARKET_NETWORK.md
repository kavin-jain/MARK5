# MARK6 — Market Network & Random-Matrix-Theory Analysis

NSE as a complex system: 186 stocks x 1902 daily returns (2018-01-01..2026-06-05). Pure price data. This is the kind of econophysics analysis used to understand systemic risk and *true* diversification beyond naive correlation.

## 1. Random Matrix Theory — how many REAL bets exist?

- Marchenko-Pastur noise band (Q=T/N=10.2): eigenvalues in [0.47, 1.72] are statistically indistinguishable from NOISE.
- **Eigenvalues above the noise band (genuine collective modes): 10 of 186.** So only ~10 statistically-real, independent sources of variation exist in a 186-stock market — the rest is noise. (Grinold's 'breadth' is far smaller than N.)
- **Market mode (largest eigenvalue) carries 28% of total variance** — the systemic 'everything moves together' risk that diversification CANNOT remove.

## 2. Minimum Spanning Tree — the market skeleton

Most-connected HUBS (systemic — when these move, the market moves):

| stock | MST degree |
|---|---|
| DLF | 18 |
| ZYDUSLIFE | 8 |
| CANBK | 8 |
| SBIN | 6 |
| NBCC | 6 |
| ULTRACEMCO | 6 |
| BAJFINANCE | 6 |
| MARUTI | 5 |

## 3. True diversifiers vs systemic names (for portfolio construction)

**Lowest average correlation = genuine risk reducers** (hold these to cut portfolio vol):

| stock | avg corr to market |
|---|---|
| MON100 | 0.02 |
| IPCALAB | 0.14 |
| ALKEM | 0.16 |
| TORNTPHARM | 0.17 |
| CIPLA | 0.17 |
| SUZLON | 0.17 |
| VBL | 0.17 |
| DRREDDY | 0.18 |
| ABBOTINDIA | 0.18 |
| CGPOWER | 0.18 |

**Highest average correlation = systemic / redundant** (overweighting these adds little diversification):

| stock | avg corr to market |
|---|---|
| DLF | 0.37 |
| CANBK | 0.36 |
| LICHSGFIN | 0.35 |
| SBIN | 0.35 |
| LT | 0.35 |
| SAIL | 0.34 |

## Why this matters for the system

- Only **~10 independent factors** drive 186 stocks → there is a hard ceiling on diversification within Indian equity alone. This is WHY adding uncorrelated *asset classes* (gold, US — corr ~0) lifted Sharpe far more than adding more Indian stocks.
- 28% systemic variance is the floor of unavoidable equity drawdown risk.
- The low-correlation names above are the empirically-best within-equity diversifiers; the factor book's inverse-vol + sector caps already tilt toward this structure.
