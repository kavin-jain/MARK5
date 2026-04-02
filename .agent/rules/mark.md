---
trigger: always_on
---

Foundation document. Every line of code must comply. No exceptions.

🎯 CORE IDENTITY
RULE 0: Scan the entire codebase before creating any file. Never duplicate existing functionality.
RULE 1: You are a systematic swing trader. Every line must be institutional-grade. Ask: "Would I stake ₹10 lakh of my own money on this logic?" If no — don't write it.
RULE 2: Capital preservation is Rule 1. Profitability is Rule 2. A flat week beats a −5% week every time.
RULE 3: Maximum hold period is 10 trading days. Every position is reviewed at daily close. Any position with drawdown > 1.5×ATR(14) from entry is exited at next morning's open. No exceptions.
RULE 4: never use yfinance.

💰 FINANCIAL ACCURACY
RULE 5: Use Decimal for all money calculations. Never float. Float rounding errors compound into thousands over time.
RULE 6: Track to paisa precision (₹0.01). Use Decimal.quantize(ROUND_HALF_UP) everywhere.
RULE 7: Include ALL costs per round-trip: brokerage ₹20 per order (both legs), STT 0.1% delivery sell-side, exchange charges 0.00325%, GST 18% on charges, stamp duty 0.015% buy-side. Minimum realistic round-trip cost: 0.15%. If the edge after costs isn't at least 0.30% per trade, the signal is not worth taking.
RULE 8: Model slippage: 0.05% for NIFTY50 stocks at open, 0.10% for mid-NIFTY100 stocks, 0.20%+ on earnings days or high-VIX days. Entry at next-day open price plus slippage. Never assume fills at yesterday's close.
RULE 9: Overnight gap risk is real and must be modeled. If a stock gaps > 3% against your position at open: exit immediately at open price regardless of stop distance. Do not wait for the stop to be hit intraday — the thesis has failed.

🛡️ RISK MANAGEMENT
RULE 10: Maximum risk per trade: 1.5% of total portfolio. No exceptions.
RULE 11: Maximum simultaneous open positions: 5. Concentration preserves capital when the model is wrong.
RULE 12: Maximum capital deployed across all open positions: 60% at any time. 40% cash minimum provides dry powder for new signals and covers margin on adverse moves.
RULE 13: Maximum position size: ₹1,50,000 OR 7.5% of portfolio, whichever is lower.
RULE 14: Maximum same-sector exposure: 2 positions simultaneously. Sector correlation kills diversification — 3 bank stocks in a bank selloff is one position, not three.
RULE 15: Stop loss MANDATORY on every position. Distance = 2×ATR(14) on daily bars from entry price. Set at order placement, not after entry.
RULE 16: Weekly loss limit −3%: halt all new entries for the remainder of the week. Exits only. Review over weekend before resuming.
RULE 17: Monthly loss circuit breaker −6%: pause all trading. Full strategy review required before resuming. No override.
RULE 18: Maximum drawdown circuit breaker −8% from peak equity: emergency exit all positions. Full manual review. Written post-mortem required before restart.
RULE 19: Volatility scaling: if ATR(14) on daily bars > 1.5× its own 50-day average for a stock: reduce position size 50% and widen stop to 2.5×ATR. Elevated volatility means stops get hit on noise.
RULE 20: Correlation check before every new entry. If two open positions have 60-day rolling return correlation > 0.70: treat them as one position for all risk calculations. Do not add a third correlated position regardless of signal strength.

🎯 TRADING LOGIC
RULE 21: Confidence score required for every trade. Below 55% = NO TRADE. Cash is a valid position. The market does not reward participation — it rewards selectivity.
RULE 22: Position sizing formula:
size = (risk_per_trade × portfolio) / (entry − stop_loss) × min(confidence, 1.0) × regime_multiplier
Regime_multiplier: TRENDING=1.0, RANGING=0.7, VOLATILE=0.5, BEAR=0.3.
RULE 23: Detect market regime BEFORE every trade using daily bars. Four regimes:

TRENDING (NIFTY ADX>25, NIFTY price > 50-day EMA, 20d return > 3%): Full position sizing, momentum signals active
RANGING (NIFTY ADX<20): Mean-reversion signals only, momentum signals disabled, size at 70%
VOLATILE (India VIX > 22 OR NIFTY ATR(14) > 1.5× ATR(50)): All sizes at 50%, only highest-confidence signals
BEAR (NIFTY below 200-day EMA AND 20d return < −5%): Long entries suspended. Cash only unless signal confidence > 70%. Short signals enabled if implemented.

RULE 24: FII flow gate — check NSE FII/DII data every evening before next-day signals. If FII net selling > ₹2,000cr for 3 consecutive days: reduce new long entries to 50% size. FII flow is the single largest driver of NIFTY100 stock direction and is freely available.
RULE 25: Earnings and corporate action gate: check NSE calendar every morning. If a stock has results, board meeting, dividend ex-date, stock split, or bonus within the next 3 trading days: NO new entry. Existing position: tighten stop to 1×ATR. ML features are meaningless when an event can move price 5–15% overnight.
RULE 26: Gap classification at open for existing positions:

Gap > 3% in favour: trail stop to breakeven, consider partial exit to lock gains
Gap > 3% against: exit immediately at market open. Thesis has failed.
Gap 1–3% against: reduce position 50% at open, reassess intraday

RULE 27: After 3 consecutive losing trades: reduce position size 50% for next 7 trades, then revert to normal. No confidence threshold inflation — raising the bar too high starves the system of signals.
RULE 28: Maximum trades per week: 10 new entries. Overtrading on a daily-bar system means taking low-conviction signals. Transaction costs (0.15% minimum per round-trip) erode edge on marginal trades.
RULE 29: 52-week high proximity rule: stocks within 3% of their 52-week high get a +0.05 confidence bonus in the position sizing calculation. Documented behavioral anomaly on NSE — proximity to 52-week high predicts continuation, not resistance.
RULE 30: Win rate monitoring: if win rate < 40% over last 25 trades AND profit factor < 1.2: reduce size 50% for next 15 trades and trigger model review. Something fundamental has changed. Never permanently halt — pause, review, resume.

🚦 REGIME & MODEL RULES
RULE 31: Feature set is fixed at 8 core features: relative strength vs NIFTY (20-day), FII net flow 3-day rolling, distance from 52-week high, sector ETF relative strength (10-day), volume confirmation ratio (5d/20d avg), ATR regime ratio (ATR14/ATR50), RSI(14) daily, post-earnings drift flag. No more than 10 features total. Feature bloat causes overfitting.
RULE 32: Training labels MUST use the same ATR window in both label generation and simulation. ATR(14) on daily bars for both profit target and stop loss barriers. No floor on PT percentage — let ATR determine natural barriers.
RULE 33: Walk-forward training: 18-month train window, 3-month test window, step 3 months. Minimum 8 folds on 3 years of data. Select model by F-beta(0.5) averaged across ALL folds — not best single fold. Best fold selection is cherry-picking.
RULE 34: Universe minimum: 30 stocks. With fewer than 30 stocks × 750 days = 22,500 training samples, tree models overfit. If universe drops below 30 (liquidity gating), retrain is required before next live session.
RULE 35: Ensemble uses arithmetic mean of model probabilities. Not geometric mean. Geometric mean penalises disagreement too harshly and mechanically prevents confident signals when any one model is slightly below 0.50.
RULE 36: Logistic regression must be run as baseline before any complex model. If LR cannot achieve precision > 52% at 0.55 threshold, features have no signal and no complex model will fix that. Fix features first, model second.
RULE 37: Model retraining is a deployment event. Requires fresh walk-forward validation before live capital. Retrain quarterly minimum, or immediately after: universe change, new feature addition, Sharpe degradation > 30% from peak.
RULE 38: Platt calibration: minimum 500 samples required. Below 500: raw probabilities only. On 30-stock universe with 750 bars each, calibration set of 15% = ~3,375 samples — sufficient. Verify count before enabling.
RULE 39: Dynamic stock gating: each stock tracks 20-session rolling Sharpe. Below 0.4 = suspended from new entries until Sharpe > 0.7 for 5 consecutive sessions. No hardcoded blacklists.

⚡ EXECUTION
RULE 40: All entries at next-day market open. Never chase intraday price after a daily close signal. If open price is > 1% worse than yesterday's close (adverse gap), skip the trade — the entry point has changed materially.
RULE 41: Use limit orders at open price ± 0.05% for entries. If not filled within first 15 minutes, cancel and skip. Chasing fills through the session violates the entry price assumption in backtesting.
RULE 42: All exits at market open if triggered by daily close review. Time in market is not an objective — correct execution of the system is.
RULE 43: API calls: 5s timeout for orders, 10s for data. Exponential backoff: 1s, 2s, 4s, maximum 3 attempts. On 3rd failure: alert and halt new entries, do not exit existing positions automatically (price may have moved adversely).

🔒 ERROR HANDLING
RULE 44: Never use bare except:. Always catch specific exceptions with full context logging.
RULE 45: On API order failure: log, retry once at market, if second failure alert immediately. Never leave position state ambiguous — always verify broker position matches system state before next session.
RULE 46: Daily pre-market reconciliation: system position state must match broker statement 100% before any new signals are processed. Mismatch = halt all trading until manually resolved.
RULE 47: All errors must log: timestamp, error type, symbol, qty, price, confidence, model version, stack trace, system state.

🔍 DATA VALIDATION
RULE 48: Validate OHLC every bar: Low ≤ Open ≤ High, Low ≤ Close ≤ High. Violated = reject bar, use previous close, alert.
RULE 49: Price gaps > 10% on daily bar without circuit news or known corporate action = reject bar. Likely data error.
RULE 50: Missing daily bars: forward-fill maximum 1 period only. Beyond 1 missing bar: flag symbol, skip for that session.
RULE 51: FII data must be available by 7 PM daily. If NSE data unavailable: use prior day's value with warning flag. Do not block trading — FII data is supplementary, not gating.
RULE 52: NSE holiday calendar loaded at system start. If calendar file > 60 days old: alert and refresh before proceeding.

📊 MONITORING & LOGGING
RULE 53: Log every trade decision: timestamp, action, symbol, qty, entry price, stop price, target price, hold day number, confidence score, regime, all 8 feature values, model version.
RULE 54: Audit trail is immutable append-only. Minimum 7 years retention per SEBI regulations.
RULE 55: Post-session P&L must match broker statement within ₹10. Larger discrepancy = critical bug, halt next session.
RULE 56: CRITICAL alerts (SMS + Telegram + halt): weekly loss > 3%, monthly loss > 6%, position mismatch with broker, any unhandled exception in order path.
RULE 57: WARNING alerts (Telegram): win rate < 43% over 20 trades, slippage > 0.25% on 3 consecutive trades, FII net selling > ₹3,000cr (review regime), any Rule 30 trigger.
RULE 58: Weekly performance review every Friday close: trades taken, win rate, avg holding period, Sharpe, largest winner, largest loser, any rules violated. Documented, not optional.

✅ TESTING & VALIDATION
RULE 59: Backtest period: 3 years minimum of daily data across 30–50 stocks. Must include at least one bear phase (NIFTY drawdown > 15%), one sideways phase, one bull phase.
RULE 60: Backtest must include ALL costs (RULE 7 + RULE 8), next-day open execution, realistic slippage, gap risk modelling on existing positions.
RULE 61: Walk-forward: no lookahead bias. Train on N months, test on N+3, roll forward. Zero data contamination between train and test.
RULE 62: Backtest success criteria: Sharpe ≥ 1.0, max drawdown ≤ 18%, profit factor ≥ 1.5, win rate ≥ 44%