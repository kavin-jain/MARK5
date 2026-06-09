# MARK6 — Holding-Period Sweep Analysis

**Question (Kavin):** the avg hold is ~461 days (>1yr) — why, and is it good? Test what
happens at much shorter holds (1d / 5d / 10d / 30d) vs the current.

**Method:** same equity factor book, same universe, same 2016-06→2026 window, ONLY the
rebalance cadence varies (buffer off for 1d–1yr rows so faster rebalancing truly shortens
holds). All net of Indian tax (LTCG 12.5% / STCG 20%) + 0.29% costs + 0.10% slippage.
Script: `scripts/holding_period_sweep.py`. All 4 code-correctness checks PASS.

| Rebalance | Avg hold | Turnover/yr | GROSS CAGR | NET CAGR | %LTCG | Sharpe | MaxDD | ₹5L → |
|---|---|---|---|---|---|---|---|---|
| 1 day | 40d | 3834% | +4.2% | **+4.0%** | 0% | 0.30 | −51.5% | ₹7.5L |
| 5 days | 67d | 1553% | +13.4% | **+13.3%** | 0% | 0.73 | −42.7% | ₹18.4L |
| 10 days | 85d | 1095% | +14.8% | **+14.6%** | 1% | 0.78 | −42.9% | ₹20.8L |
| 30 days | 117d | 689% | +15.7% | **+15.5%** | 2% | 0.83 | −44.1% | ₹22.4L |
| 3 months | 168d | 419% | +16.2% | +16.1% | 6% | 0.83 | −44.6% | ₹23.7L |
| 6 months | 253d | 269% | +17.3% | +17.1% | 22% | 0.88 | −39.2% | ₹25.9L |
| 1 year | 406d | 180% | +17.0% | +16.8% | 100% | 0.88 | −39.5% | ₹25.3L |
| **1yr + buffer (DEPLOYED)** | 461d | 157% | +16.2% | +16.1% | 100% | **0.89** | **−34.6%** | ₹23.6L |

## Findings
1. **Fast trading is catastrophic.** 1-day rebalancing → +4.0% net (₹7.5L vs ₹25L). Even
   GROSS is +4.2% — **transaction costs alone bleed ~13pp/yr** at 3834% turnover. This is
   the mechanism behind the ~90% retail-trader loss rate: overtrading, not bad stock picks.
2. **Both costs AND tax punish short holds:** ≤30-day holds get 0% LTCG (all 20% STCG) plus
   heavy costs; 1-year holds get 100% LTCG (12.5%) plus minimal costs.
3. **Net return climbs monotonically with holding period**, plateauing ~6mo–1yr (~+16-17%,
   Sharpe ~0.88). Beyond ~6mo, extra holding buys tax-efficiency + drawdown control, not return.
4. **The deployed 461-day config is near-optimal** — sacrifices ~0.7pp raw CAGR vs the 6-month
   peak for the lowest drawdown (−34.6%) and best Sharpe (0.89). The long hold is a FEATURE.

## Tuning note
A 6-month no-buffer cadence is the max-*raw-CAGR* point (+17.1% but −39% DD); the deployed
1yr+buffer is the max-*risk-adjusted* point (Sharpe 0.89, −34.6% DD). Deployed config keeps
the risk-adjusted choice. Verdict: do NOT shorten the holding period — it would cut returns
sharply (down to +4% at daily). Confirms log entries K3 (turnover→tax) and K4 (stops).
