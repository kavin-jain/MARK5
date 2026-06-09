"""
MARK6 — Market Network & Random-Matrix-Theory Analysis (econophysics)
=====================================================================
Treats the NSE as a complex system. From the daily-return correlation matrix:

  1. RMT (Random Matrix Theory): compare the eigenvalue spectrum to the
     Marchenko-Pastur null. Eigenvalues ABOVE the noise band = genuine collective
     modes (real structure); the rest is noise. The largest eigenvalue = the
     'market mode' (systemic risk). Tells us how many INDEPENDENT bets really exist.
  2. Minimum Spanning Tree (distance = sqrt(2(1-corr))): the skeleton of the market
     — hubs (systemic names) vs leaves (peripheral diversifiers).
  3. Diversification read: most-central (systemic) vs most-peripheral (independent)
     names — directly informs which holdings genuinely reduce risk.

Pure price data, no new sources. Writes reports/MARKET_NETWORK.md.

  python3 scripts/market_network.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import DataPanel, discover_tickers
REPORTS = os.path.join(_ROOT, "reports")
START, END = "2018-01-01", "2026-06-05"


def main():
    panel = DataPanel(discover_tickers(), END)
    px = panel.close.loc[START:END]
    # keep names with near-full history (clean correlation estimate)
    px = px.dropna(axis=1, thresh=int(len(px) * 0.9)).ffill().dropna()
    rets = px.pct_change(fill_method=None).dropna()
    tickers = list(rets.columns)
    T, N = rets.shape
    print(f"Correlation network: {N} stocks x {T} days\n", flush=True)

    C = np.corrcoef(rets.values, rowvar=False)
    eig, vec = np.linalg.eigh(C)
    eig = eig[::-1]; vec = vec[:, ::-1]

    # Marchenko-Pastur noise band for Q = T/N, sigma^2 = 1
    Q = T / N
    lam_max = (1 + np.sqrt(1 / Q)) ** 2
    lam_min = (1 - np.sqrt(1 / Q)) ** 2
    n_signal = int(np.sum(eig > lam_max))
    market_var = eig[0] / N * 100        # % of total variance in the market mode

    # MST: distance d_ij = sqrt(2(1-corr))
    D = np.sqrt(2 * (1 - C))
    mst = minimum_spanning_tree(D).toarray()
    mst = mst + mst.T
    degree = (mst > 0).sum(axis=1)
    # 'centrality' ~ degree in MST; hubs = systemic, leaves = peripheral
    order = np.argsort(degree)
    hubs = [(tickers[i], int(degree[i])) for i in order[::-1][:8]]
    # peripheral diversifiers: lowest avg correlation to the rest
    avg_corr = (C.sum(axis=1) - 1) / (N - 1)
    div_order = np.argsort(avg_corr)
    diversifiers = [(tickers[i], avg_corr[i]) for i in div_order[:10]]
    systemic = [(tickers[i], avg_corr[i]) for i in div_order[::-1][:6]]

    L = ["# MARK6 — Market Network & Random-Matrix-Theory Analysis", "",
         f"NSE as a complex system: {N} stocks x {T} daily returns ({START}..{END}). "
         "Pure price data. This is the kind of econophysics analysis used to understand "
         "systemic risk and *true* diversification beyond naive correlation.", "",
         "## 1. Random Matrix Theory — how many REAL bets exist?", "",
         f"- Marchenko-Pastur noise band (Q=T/N={Q:.1f}): eigenvalues in "
         f"[{lam_min:.2f}, {lam_max:.2f}] are statistically indistinguishable from NOISE.",
         f"- **Eigenvalues above the noise band (genuine collective modes): {n_signal} of {N}.** "
         f"So only ~{n_signal} statistically-real, independent sources of variation exist in "
         f"a {N}-stock market — the rest is noise. (Grinold's 'breadth' is far smaller than N.)",
         f"- **Market mode (largest eigenvalue) carries {market_var:.0f}% of total variance** — "
         "the systemic 'everything moves together' risk that diversification CANNOT remove.", "",
         "## 2. Minimum Spanning Tree — the market skeleton", "",
         "Most-connected HUBS (systemic — when these move, the market moves):",
         "", "| stock | MST degree |", "|---|---|"]
    for t, d in hubs:
        L.append(f"| {t} | {d} |")
    L += ["", "## 3. True diversifiers vs systemic names (for portfolio construction)", "",
          "**Lowest average correlation = genuine risk reducers** (hold these to cut portfolio vol):",
          "", "| stock | avg corr to market |", "|---|---|"]
    for t, c in diversifiers:
        L.append(f"| {t} | {c:.2f} |")
    L += ["", "**Highest average correlation = systemic / redundant** (overweighting these adds little diversification):",
          "", "| stock | avg corr to market |", "|---|---|"]
    for t, c in systemic:
        L.append(f"| {t} | {c:.2f} |")
    L += ["", "## Why this matters for the system", "",
          f"- Only **~{n_signal} independent factors** drive {N} stocks → there is a hard ceiling "
          "on diversification within Indian equity alone. This is WHY adding uncorrelated *asset "
          "classes* (gold, US — corr ~0) lifted Sharpe far more than adding more Indian stocks.",
          f"- {market_var:.0f}% systemic variance is the floor of unavoidable equity drawdown risk.",
          "- The low-correlation names above are the empirically-best within-equity diversifiers; "
          "the factor book's inverse-vol + sector caps already tilt toward this structure.", ""]
    os.makedirs(REPORTS, exist_ok=True)
    open(os.path.join(REPORTS, "MARKET_NETWORK.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print("\nSaved -> reports/MARKET_NETWORK.md")


if __name__ == "__main__":
    main()
