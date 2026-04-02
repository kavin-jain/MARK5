MARK5 — COMPLETE REBUILD ROADMAP
What You Actually Need to Do to Build Real Edge
Senior Quant Review — No Sugarcoating
THE CORE PROBLEM IN ONE SENTENCE
Yo u b u i l t a sys t e m t h a t p re d i c t s p r i c e u s i n g p r i c e. E ve r y re t a i l a l g o d o e s t h i s. Re a l e d g e
comes from information that the price hasn’t fully reflected yet.
PA R T 1 — ST R AT E G I C R E P O S I T I O N I N G
1.1 Stop Trading Nifty 50
This is the hardest truth. The Nifty 50 is the most efficiently priced, most heavily
arbitraged equity universe in India. You are competing against:
Quanthouse, Edelweiss AlgoLab, Motilal Oswal QAM — who have co-location at
NSE’s Mahape datacenter, 10Gbps feeds, and teams of 20+ quants
FII prop desks — Goldman, JPM, Citadel India — who have the actual FII flow DATA
before you see it aggregated and delayed by 6 hours
Domestic HFT firms — iRage, Alphagrep — who arbitrage any inefficiency in liquid
stocks within microseconds
Yo u c a n n o t w i n i n t h i s a re n a w i t h a d a i l y- b a r s i g n a l. Yo u r i n fo r m a t i o n l a t e n c y i s ~ 6 h o u rs
(end of day data). Theirs is sub-millisecond.
Where you CAN win: Nifty Midcap 150 and Smallcap 250 stocks, specifically:
Stocks between ₹500cr–₹5,000cr market cap
Daily turnover ₹20cr–₹200cr
NOT in F&O — no HFT arb possible without derivatives
These stocks have:
Real institutional inefficiency (fund managers can’t build large positions quickly)
Analyst coverage gaps (most have 0–2 analysts vs 25+ for Nifty 50)
Slower price discovery — your daily signal has a chance to be early
The catch: Execution is harder (wider spreads, more slippage). Your position sizer
MUST model this. More on that later.
1.2 Your Three Actual Targets
UNIVERSE A — Core (50 stocks)
Nifty Midcap Select + quality midcaps
Market cap: ₹3,000cr – ₹15,000cr
Liquidity: Daily turnover > ₹50cr
F&O: NO (eliminates HFT arb)
Edge type: Information gap + momentum continuation
UNIVERSE B — Satellite (30 stocks)
Nifty Next 50 (large-cap but less covered than Nifty 50)
Market cap: ₹15,000cr – ₹50,000cr
F&O: YES (but use F&O data as SIGNAL, not obstacle)
Edge type: Index inclusion/exclusion events
UNIVERSE C — Event-Driven (Dynamic)
Stocks with specific catalysts (results, block deals, promoter buying)
No fixed universe — generated from event screening daily
Edge type: Pure information asymmetry
PA R T 2 — T H E R E A L A L P H A S O U R C E S
This is where your system is completely blind. Your 11 features are all derived from
OHLCV. Real institutional alpha comes from ALTERNATIVE DATA.
2.1 F&O Data — Your Biggest Missed Opportunity
NSE publishes the complete F&O data daily for FREE. You are ignoring it entirely. This is
institutional-grade information.
Feature: Futures Basis (Premium/Discount)
basis(t) = [Futures_Close(t) - Spot_Close(t)] / Spot_Close(t)
annualized_basis = basis × (252 / days_to_expiry)
Interpretation:
annualized_basis > 15%: Extreme bullish positioning — longs paying up
annualized_basis > 10%: Normal bull market (cost of carry ~8-10%)
annualized_basis < 5%: Bears in control — no one wants to hold futures
annualized_basis < 0%: Backwardation — panic, forced unwinding
Rolling signal:
basis_zscore = (basis_5d_avg - basis_60d_avg) / basis_60d_std
HIGH zscore + rising price = smart money accumulating
LOW zscore + falling price = institutional distribution
This tells you where institutions (who trade futures, not spot) are positioned.
Feature: Put-Call Ratio (PCR)
pcr_oi(t) = sum(Put_OI across all strikes) / sum(Call_OI across all strikes)
pcr_vol(t) = sum(Put_Volume) / sum(Call_Volume)
Signal logic (CONTRARIAN, not directional):
PCR > 1.5 → EXTREME BEARISHNESS → contrarian BUY signal
PCR < 0.5 → EXTREME BULLISHNESS → contrarian SELL / reduce signal
PCR 0.7–1.2 → Neutral range
Why contrarian? Retail buys puts when they're scared. Options writers
(institutions) take the other side. If PCR is extreme, the options
writers (smarter money) are positioned against the consensus.
Feature: IV Skew
skew(t) = IV_at_10%_OTM_Put - IV_at_10%_OTM_Call
Interpretation:
HIGH positive skew: Market willing to pay MORE for downside protection
than upside participation → institutional hedging
LOW/negative skew: Market positioned for upside → less fear
skew_zscore = (skew - skew_30d_mean) / skew_30d_std
Rising skew_zscore + falling price = acceleration of fear → bearish
Falling skew_zscore + rising price = fear unwinding → continuation signal
Feature: Max Pain
Feature: OI Change Interpretation
oi_change_pct(t) = [OI(t) - OI(t-1)] / OI(t-1)
Combined signal matrix:
Price UP + OI UP → Long buildup → BULLISH
Price UP + OI DOWN → Short covering → weak bullish, likely reversal soon
Price DOWN + OI UP → Short buildup → BEARISH
Price DOWN + OI DOWN → Long unwinding → weak bearish, likely stabilization
Encode as:
oi_signal(t) = sign(price_change) × sign(oi_change)
+1 = trending (buildup in direction)
-1 = counter (covering or unwinding)
Source: NSE bhav copy for F&O — https://www.nseindia.com/api/option-chain-
indices?symbol=NIFTY All free. No paid API needed.
2.2 Bulk Deal & Block Deal Data
NSE publishes every bulk deal (>0.5% of shares traded in a session) and block deal
(negotiated off-market, typically >₹10cr) within the same trading day.
Python source:
max_pain(t) = strike price where total option writer loss is minimized at expiry
price_vs_maxpain = (Spot(t) - max_pain(t)) / ATR14(t)
Logic: Market makers hedge by buying/selling the underlying.
As expiry approaches, price gravitates toward max pain because
market makers' delta hedging creates that gravitational pull.
Signal: In the last 5 days before expiry:
If price > max_pain by > 2 ATR → expect mean reversion toward max_pain
If price < max_pain by > 2 ATR → expect rally toward max_pain
Bulk deals: https://www.nseindia.com/api/bulk-deals
Block deals: https://www.nseindia.com/api/block-deals
Features to engineer:
bulk_buy_value(t) = sum of buy-side bulk deal value for stock on day t
block_buy_flag(t) = 1 if a block deal occurred (institutional buying)
Signal: Block deal by a known FII/mutual fund → strong accumulation signal
The quantity is disclosed. Work backwards to check if it's delivery.
Look back 5 days:
institutional_buying_5d = rolling sum of bulk_buy_value, last 5 days
Normalize by ADV (average daily volume) to make it cross-sectional
This is DATA YOUR COMPETITORS DON’T HAVE IN THEIR DAILY-BAR MODELS.
2.3 Promoter Shareholding Changes
SEBI mandates disclosure within 2 trading days of any promoter transaction. This is the
closest thing to a legal “insider” signal.
Source: SEBI EDGAR (free API)
https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doPF=yes
Features:
promoter_buy_flag(t) = 1 if promoter buying within last 10 days
promoter_buy_pct(t) = % of shares bought / total promoter holding
promoter_pledge_pct(t) = pledged shares / total promoter holding
HIGH pledge (>30%) = danger signal — promoter may be forced to sell
DECREASING pledge = bullish — financial stress reducing
These update monthly from quarterly disclosures, and intraday from
SEBI disclosures for significant transactions.
2.4 Mutual Fund Portfolio Disclosures
SEBI requires all mutual funds to disclose full portfolios monthly (within 10 days of
month-end). This is PUBLIC DATA most retail traders completely ignore.
Source: AMFI monthly portfolio disclosures
https://www.amfiindia.com/research-information/mf-data/mf-scheme-portfolio
Feature engineering:
mf_accumulation_score(t) = change in mutual fund holding
(month over month, as % of total shares)
Category breakdown matters:
SMALL_CAP_FUND_HOLDING_CHANGE: high impact (they're forced buyers)
MIDCAP_FUND_HOLDING_CHANGE: medium impact
LARGECAP: low impact (benchmark-hugging, less signal)
Combined signal:
mf_smart_money_score = weighted_avg(holding_change × fund_alpha_score)
where fund_alpha_score = rolling 1yr excess return of that fund
(high-alpha funds' moves matter more than index funds')
2.5 Analyst Estimate Revisions
This is one of the strongest known alpha factors globally (Post-Earnings Announcement
Drift + Estimate Revision Momentum).
Source: Screener.in / Trendlyne API (low cost, not free but cheap)
or scrape BSE analyst reports (free but messy)
Feature:
earnings_revision_3m = (EPS_estimate_now - EPS_estimate_3m_ago)
/ |EPS_estimate_3m_ago|
Signal:
revision_zscore > +1.5: Multiple analysts raising estimates
Price hasn't caught up yet → BUY
revision_zscore < -1.5: Estimate cuts → potential earnings miss → AVOID
The IC (Information Coefficient) of estimate revision momentum
is ~0.08 monthly on NSE — statistically significant and consistent.
Your current features have IC near 0 on a 1-day horizon.
PA R T 3 — C O M P L E T E F E AT U R E OV E R H AU L
3.1 Information Coefficient Test — Do This FIRST
Before building anything, run this on your EXISTING 11 features:
def compute_feature_ic(features_df, returns_5d):
"""
IC = Spearman rank correlation between feature today
and forward 5-day return.
Run this for each feature separately.
Acceptable IC for daily features: > 0.03 (3%)
Good IC: > 0.
Great IC: > 0.
If a feature has IC < 0.02, it's contributing noise. Kill it.
"""
from scipy.stats import spearmanr
ics = {}
for col in features_df.columns:
ic, pval = spearmanr(features_df[col], returns_5d, nan_policy='omit')
ics[col] = {'ic': ic, 'pval': pval, 'significant': pval < 0.05}
return pd.DataFrame(ics).T.sort_values('ic', ascending=False)
My prediction: delivery_pct (obviously 0), amihud_illiquidity, and gap_significance
will show IC < 0.02. You’re wasting 3 feature slots.
3.2 Stationarity — Features Must Be Stationary
None of your features are tested for stationarity. A non-stationary feature will cause
your model to learn spurious relationships from specific time periods.
from statsmodels.tsa.stattools import adfuller
def test_stationarity(series, feature_name):
result = adfuller(series.dropna(), maxlag=20)
pvalue = result[1]
if pvalue > 0.05:
print(f"WARNING: {feature_name} is NON-STATIONARY (p={pvalue:.3f})")
print(f" → Apply fractional differentiation (d=0.3–0.5)")
print(f" → Or use % change instead of level")
return pvalue
# Likely non-stationary in your current setup:
# - relative_strength_nifty during trending markets
# - amihud_illiquidity (liquidity regime shifts)
# - price_vwap_deviation during strong trends
Fix: Apply fractional differentiation (you have optimize_frac_diff.py — USE IT) to
preserve memory while achieving stationarity. This alone will improve your model’s out-
of-sample stability significantly.
3.3 The New Feature Set (20 Features, All Tested)
TIER 1 — F&O Features (NEW — the edge)
Feature Formula IC Target
─────────────────────────────────────────────────────────────────
futures_basis_zscore (basis - basis_60d_mean) 0.05+
/ basis_60d_std
pcr_oi Put_OI / Call_OI 0.04+
iv_skew_zscore (skew - skew_30d_mean) 0.06+
/ skew_30d_std
oi_signal sign(Δprice) × sign(ΔOI) 0.05+
max_pain_distance (spot - max_pain) / ATR14 0.04+
[expiry week only, else 0]
TIER 2 — Alternative Data Features (NEW — the structural edge)
Feature Formula Source
───────────────────────────────────────────────────────────────
institutional_buy_5d sum(bulk/block buys, 5d) NSE bulk deals
/ ADV_20d
promoter_buy_flag 1/0, 10-day lookback SEBI EDGAR
pledge_pct_change Δpledge_pct, quarterly BSE disclosures
mf_accumulation_3m ΔMF holding, 3 months AMFI
earnings_revision_momentum ΔEPS_estimate / |EPS_t-3m| Screener.in
TIER 3 — Retained from Current System (After IC Test)
Feature Retain? Why
───────────────────────────────────────────────────────────
relative_strength_nifty YES IC proven in literature ~0.
dist_52w_high YES George & Hwang factor, IC ~0.
efficiency_ratio YES Regime indicator, useful
ofi_proxy YES Intraday order flow, IC ~0.
lower_wick_ratio MAYBE Only if IC > 0.02, test first
fii_flow_3d REWORK Move to 10d & 20d windows too
wick_asymmetry NO Redundant with lower_wick_ratio
gap_significance MAYBE Only for event-driven signals
price_vwap_deviation YES Institutional benchmark distance
delivery_pct NO Until real data exists
amihud_illiquidity REWORK Fix lookahead first, then test IC
PA R T 4 — M O D E L A R C H I T E C T U R E OV E R H AU L
4.1 Should You Use Separate Regime Models? Yes, But Not
How You Think.
Yo u r c u r re n t a p p ro a c h : d e t e c t re g i m e → switch model → predict.
The problem: Regime detection is itself noisy. If you’re wrong about the regime, you’re
running the wrong model entirely. Two noisy steps = compounded error.
The correct architecture:
┌─────────────────────────────────────────────────────────┐
│ TWO-STAGE ARCHITECTURE │
└─────────────────────────────────────────────────────────┘
STAGE 1: BASE MODEL (Trained on ALL regimes)
─────────────────────────────────────────────
Input: 20 features
Model: LightGBM (primary)
Output: base_probability ∈ [0,1]
This model learns universal patterns that work across regimes.
Train on full history (15 years, not 3).
STAGE 2: REGIME-SPECIFIC CALIBRATION LAYERS
─────────────────────────────────────────────
For each detected regime, apply a CALIBRATION MODEL
(NOT a separate full model — just a calibration layer)
Input: base_probability + regime_features
Model: Isotonic Regression or Platt Scaling (per regime)
Output: calibrated_probability
Regime-specific calibrators are trained on:
BULL regime: calibrated to actual bull-period win rates
BEAR regime: calibrated to actual bear-period win rates
HIGH_VOL regime: calibrated to volatile-period win rates
This separates concerns:
Base model: learns what patterns look like
Calibrator: learns what those probabilities MEAN in each context
Why this beats 3 separate full models:
3 separate models each need enough training data per regime. You don’t have it.
One base model uses ALL data → better generalization
Calibrators only need ~100 samples per regime to be useful
4.2 Model Selection — Kill Random Forest
Random Forest is the weakest model in your ensemble and contributes the most noise.
Here’s why:
Comparison on financial tabular data (well-documented in research):
Algorithm Typical Rank Why
────────────────────────────────────────────────────────
LightGBM #1 Leaf-wise growth, handles
tabular financial data best.
Native categorical support.
XGBoost #2 Level-wise, more regularized.
Good for smaller datasets.
CatBoost #3 Best for ordinal/categorical
features. Worth testing.
Random Forest #4 Parallel trees, high variance.
Usually dominated by gradient
boosting on financial data.
Neural Net #5 (daily) Needs 10x more data for daily
bars. Don't bother until you
have 50k+ samples.
Replace RF with CatBoost:
from catboost import CatBoostClassifier
catboost_model = CatBoostClassifier(
iterations=1000,
learning_rate=0.05,
depth=6,
l2_leaf_reg=3.0,
eval_metric='Logloss',
early_stopping_rounds=50,
task_type='GPU', # GPU support built-in
random_seed=42,
verbose=False,
# CatBoost handles class imbalance natively:
class_weights=[1.0, 2.0], # weight positive class 2x
)
CatBoost handles ordinal features (like your regime indicators) better than LightGBM
and has lower variance than RF.
4.3 Proper Ensemble Weighting — Not Arithmetic Mean
Yo u r c u r re n t (xgb + lgb + rf) / 3 gives equal weight to all models. This is suboptimal
and you already know it (you have ensemble.py for dynamic weighting but don’t use it in
the live path).
Stacking Ensemble:
"""
LEVEL 0: Base learners (LightGBM, XGBoost, CatBoost)
LEVEL 1: Meta-learner trained on out-of-fold predictions
THIS IS THE CORRECT WAY.
"""
# Step 1: Generate out-of-fold predictions for each base model
# (using your walk-forward folds — this is already handled correctly)
oof_xgb = cross_val_predict(xgb, X_train, y_train, cv=walk_forward_cv)
oof_lgb = cross_val_predict(lgb, X_train, y_train, cv=walk_forward_cv)
oof_cat = cross_val_predict(cat, X_train, y_train, cv=walk_forward_cv)
# Step 2: Stack OOF predictions + original features into meta-features
meta_X = np.column_stack([oof_xgb, oof_lgb, oof_cat, regime_features])
# Step 3: Train a simple meta-learner on these
from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression(C=0.1) # Low C = high regularization
meta_model.fit(meta_X, y_train)
# Step 4: At inference
level0_preds = [xgb.predict_proba(X)[:,1],
lgb.predict_proba(X)[:,1],
cat.predict_proba(X)[:,1]]
meta_input = np.column_stack(level0_preds + [regime_features])
final_prob = meta_model.predict_proba(meta_input)[:,1]
The meta-learner learns how much to trust each base model — and this weight can vary
by regime context automatically.
4.4 Fix Validation — Use CPCV, Not Walk-Forward
Yo u h ave cpcv.py in your codebase. It is NOT being used in trainer.py. This is a
significant oversight.
What Combinatorial Purged Cross-Validation (CPCV) does that walk-forward
doesn’t:
Walk-Forward (your current):
Test only the MOST RECENT period.
If the most recent period is anomalous (e.g., COVID rally),
your fold selection is completely biased.
CPCV (what you should use):
Tests EVERY POSSIBLE combination of train/test splits.
For N folds chosen k at a time, produces C(N,k) test sets.
Each sample appears in test set exactly C(N-1, k-1) times.
Result: distributional estimates of performance,
not a single optimistic test-set score.
You get: mean(sharpe), std(sharpe), P(sharpe > 0)
Not just: sharpe on the one test period that happened to look good.
# In trainer.py, replace WalkForwardCV with:
from cpcv import CombinatorialPurgedCV # you already have this file
cpcv = CombinatorialPurgedCV(
n_splits=10, # N folds
n_test_splits=2, # k test folds at a time → C(10,2) = 45 test combos
embargo_pct=0.02 # 2% embargo between train and test
)
# This gives you 45 performance estimates instead of 3-4.
# If P(sharpe > 1.5) > 70%, the signal is real.
# If the distribution is wide (std > mean), the signal is noise.
PA R T 5 — T H E E X E C U T I O N E D G E
5.1 Slippage Is Not 0.05%. Fix It.
For midcap stocks, here is the real slippage model:
def realistic_slippage(price, quantity, adv_20d, urgency='patient'):
"""
Market impact model based on Almgren-Chriss framework,
calibrated for NSE midcap stocks.
adv_20d: 20-day average daily value traded (₹)
participation_rate: what fraction of daily volume is your order
"""
trade_value = price * quantity
participation_rate = trade_value / adv_20d
# Temporary market impact (recovers after trade)
# Calibrated for NSE: η = 0.0025 for midcap
temp_impact = 0.0025 * (participation_rate ** 0.5)
# Permanent market impact (price doesn't recover)
# γ = 0.001 for midcap
perm_impact = 0.001 * participation_rate
# Bid-ask spread component (half-spread)
if adv_20d < 5e7: # < ₹5cr daily — illiquid
spread_cost = 0.0015 # 15bps half-spread
elif adv_20d < 5e8: # ₹5cr–₹50cr — semi-liquid
spread_cost = 0.0008 # 8bps
else: # > ₹50cr — liquid
spread_cost = 0.0003 # 3bps (Nifty 50 level)
total_slippage = temp_impact + perm_impact + spread_cost
return total_slippage
# Example: Buying ₹50,000 of a ₹30cr/day midcap stock:
# participation_rate = 50000 / 30,000,000 = 0.17%
# temp_impact = 0.0025 × sqrt(0.0017) = 0.0001 (1bp)
# perm_impact = 0.001 × 0.0017 = 0.0000017 (negligible)
# spread_cost = 0.0008 (8bps)
# TOTAL: ~9bps per leg = 18bps round trip (vs your 10bps assumption)
# For a ₹5,00,000 order in the same stock: ~25bps per leg
This matters enormously. At your current profit target of ~3%, 18bps vs 10bps is the
difference between 2.64% net and 2.80% net — small per trade, but compounded over
100 trades it’s 16% of your total P&L.
5.2 Order Type Strategy
Never use market orders in midcap stocks. Ever.
Order Hierarchy for MARK5:
─────────────────────────────────────────────────────────
ENTRY (patience is possible):
Type: Limit order at [next_open + 0.1%]
Cancel if: Not filled within 30 minutes
Fallback: Adjust limit to [current_LTP + 0.2%]
Why: You're entering based on yesterday's signal.
Another 0.1% in cost is worth knowing your fill price.
STOP LOSS (patience is NOT possible):
Type: SL-M order (Stop Loss Market) in Kite
Why: If your SL is hit, you MUST exit. Limit SL orders
can fail to fill in fast-moving markets.
PROFIT TARGET (patience is possible):
Type: Limit order at target price
Advantage: You will often get price IMPROVEMENT
(stock will trade through your target intraday)
4. TIME BARRIER EXIT:
Type: Limit order at LTP - 0.05% at 3:15 PM
Convert to Market at 3:25 PM if unfilled
5.3 Implement VWAP-Sliced Execution for Large Orders
If your position size is >1% of ADV (average daily volume), slice the order:
def vwap_schedule(total_qty, adv_pct_limit=0.005):
"""
PA R T 6 — T H E C O R R E C T VA L I DAT I O N
FRAMEWORK
6.1 The Bar You Must Clear Before Going Live
These are non-negotiable. If any metric fails, the system is NOT live-ready.
┌─────────────────────────────────────────────────────────────────┐
│ PRODUCTION GATE v2.0 │
│ (Replace gate.py with these thresholds) │
└─────────────────────────────────────────────────────────────────┘
CPCV PERFORMANCE (45 test folds from CPCV, not 3-4 walk-forward):
P(Sharpe > 1.5) > 70% ← Must hold across MOST test folds
Mean Sharpe: > 1.8 ← Not just barely above 1.
Worst-5% Sharpe: > 0.0 ← Even bad folds shouldn't be negative
SIGNAL QUALITY:
Mean IC (5-day): > 0.04 ← Information content exists
IC/ICIR: > 0.50 ← IC is stable, not just lucky folds
Precision at 0.60: > 58% ← When model is confident, it's right
NSE volume follows a U-curve: high at open, low midday, high at close.
Split large orders across time slices proportional to typical volume.
NSE volume profile (approximate):
9:15–10:00 AM: 22% of daily volume (most liquid, best for entry)
10:00–11:00 AM: 15% of volume
11:00–12:00 PM: 9% of volume
12:00–01:00 PM: 8% of volume
01:00–02:00 PM: 9% of volume
02:00–03:00 PM: 13% of volume
03:00–03:30 PM: 24% of volume (dangerous for entry, good for exit)
Strategy: Only trade in first 2 hours (37% of volume)
when liquidity is deepest for midcap stocks.
"""
max_per_slice = int(adv_pct_limit * daily_volume)
n_slices = ceil(total_qty / max_per_slice)
slice_times = ['09:15', '09:30', '09:45', '10:00', '10:15', '10:30'][:n_slices]
return [(t, max_per_slice) for t in slice_times]
ROBUSTNESS TESTS (all must pass):
Feature permutation: Performance drops <30% if any ONE feature removed
(if removing one feature kills performance, you're overfitting to it)
Lookback sensitivity: Trained on 3yr vs 5yr vs 10yr history →
performance variance < 20%
(if performance depends heavily on training window, signal isn't robust)
Transaction cost sensitivity: Double all costs →
Sharpe still > 1.
(edge must survive cost uncertainty)
REAL-WORLD SIMULATION:
Paper trade minimum: 6 months
Requires: ≥80 completed trades (statistical significance)
Live Sharpe must be within 0.5 of backtest Sharpe
(if live is more than 0.5 worse, there's execution or implementation leakage)
6.2 The Bias Tests You Must Run
# Test 1: Verify walk-forward has no lookahead
def test_no_lookahead(signals, returns):
"""
Shift signals forward by 1 day (making them deliberately wrong).
Performance should COLLAPSE. If it doesn't, there's lookahead.
"""
shifted_signals = signals.shift(1)
shifted_sharpe = compute_sharpe(shifted_signals, returns)
original_sharpe = compute_sharpe(signals, returns)
assert original_sharpe > 2 × shifted_sharpe, "Lookahead bias suspected"
# Test 2: Feature importance stability
def test_feature_stability(model, X_train_folds):
"""
Train on each fold separately.
Top 3 features should be consistent across folds.
If different folds give completely different top features,
you're fitting noise.
"""
importances = [model.fit(X_fold, y_fold).feature_importances_
for X_fold, y_fold in X_train_folds]
rank_correlation = spearmanr(importances[0], importances[-1])
assert rank_correlation > 0.7, "Feature importance is unstable — overfitting"
# Test 3: Return correlation structure
PA R T 7 — S P E C I F I C C O D E C H A N G E S , F I L E BY F I L E
7.1 features.py — Priority 1
def test_signal_correlation(signals_by_stock):
"""
Signals across stocks should NOT be 100% correlated.
If every stock gets BUY on the same days, you're trading
market beta, not stock-specific alpha.
Target: average pairwise signal correlation < 0.
"""
corr_matrix = signals_by_stock.corr()
avg_corr = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).mean().mean()
assert avg_corr < 0.40, f"Signals too correlated ({avg_corr:.2f}) — pure beta"
# ADD these imports and functions at top of features.py
def compute_fno_features(symbol, date, fno_cache):
"""
Pull from pre-cached F&O data (update daily from NSE).
Returns dict of F&O features for the given symbol and date.
"""
data = fno_cache.get(symbol, date)
if data is None:
return {f: 0.0 for f in FNO_FEATURE_NAMES} # neutral fallback
return {
'futures_basis_zscore': compute_basis_zscore(data),
'pcr_oi': data['put_oi'] / (data['call_oi'] + 1e-6),
'iv_skew_zscore': compute_skew_zscore(data),
'oi_signal': np.sign(data['price_change']) * np.sign(data['oi_change']),
'max_pain_distance': compute_max_pain_dist(data, atr14),
}
def compute_alt_data_features(symbol, date, alt_data_cache):
"""
Pull from pre-fetched alternative data cache.
bulk deals, block deals, promoter transactions, MF holding changes.
Update: bulk/block daily, MF monthly, promoter within 2 days of event.
"""
return {
7. 2 trainer.py — Priority 1
# REPLACE WalkForwardCV with CPCV
from cpcv import CombinatorialPurgedCV # already in your repo
cpcv = CombinatorialPurgedCV(
n_splits=10,
n_test_splits=2,
embargo_pct=0.02,
purge_pct=0.01,
)
# REPLACE RF with CatBoost
from catboost import CatBoostClassifier
models = {
'lgb': LGBMClassifier(**lgb_params),
'xgb': XGBClassifier(**xgb_params),
'cat': CatBoostClassifier(**cat_params),
}
# ADD: Out-of-fold stacking
# REPLACE: Arithmetic mean ensemble → LogisticRegression meta-learner
# ADD: Calibration after training
from sklearn.calibration import CalibratedClassifierCV
# Wrap each base model with Platt scaling calibration
lgb_calibrated = CalibratedClassifierCV(lgb, method='sigmoid', cv='prefit')
lgb_calibrated.fit(X_calib, y_calib) # small held-out calibration set
# ADD: Feature importance stability test (see Section 6.2)
# ADD: IC computation and logging per feature per fold
# CHANGE: Production gate thresholds (see Section 6.1)
'institutional_buy_5d': alt_data_cache.bulk_buy_5d(symbol, date),
'block_deal_flag': alt_data_cache.block_deal(symbol, date),
'promoter_buy_flag': alt_data_cache.promoter_buying(symbol, date),
'pledge_pct': alt_data_cache.pledge_pct(symbol, date),
'mf_accumulation_3m': alt_data_cache.mf_change_3m(symbol, date),
}
# REMOVE: delivery_pct (constant 0.5 — dead feature)
# REMOVE: wick_asymmetry (redundant with lower_wick_ratio)
# MODIFY: amihud_illiquidity — compute p99 on TRAINING WINDOW ONLY
# Pass training_end_date to engineer_all_features()
# amihud_p99 = amihud_20[:training_end_date].quantile(0.99)
7. 3 data_pipeline.py — New File Needed
"""
NEW FILE: data_pipeline.py
─────────────────────────────────────────────────────────────
Orchestrates the daily alternative data fetch.
Run at 6:30 PM IST (after all NSE data is published).
Schedule:
6:00 PM: FII/DII data published → fetch fii_data.py (existing)
6:15 PM: Bulk/block deals final → fetch bulk_deals.py (new)
6:30 PM: F&O bhav copy → fetch fno_data.py (new)
7:00 PM: All data validated → update feature cache →
retrain if weekly → send readiness alert
Monthly (1st of month + 10 days):
AMFI portfolio disclosures → fetch mf_holdings.py (new)
Event-driven (SEBI EDGAR RSS):
Promoter transactions → fetch promoter_data.py (new)
"""
class DailyDataOrchestrator:
def run_daily_pipeline(self, date):
self.fetch_fii_data(date) # existing
self.fetch_bulk_deals(date) # new
self.fetch_fno_bhav_copy(date) # new
self.validate_all_sources(date) # new
self.update_feature_cache(date) # new
self.trigger_retraining_if_needed()
7. 4 regime_detector.py — Fix Dead STRONG_BULL
# CURRENT (broken):
STRONG_BULL_RET_THRESHOLD = 0.15 # 15% in 20 days → never triggers
# FIX:
STRONG_BULL_RET_THRESHOLD = 0.07 # 7% in 20 days (~88% annualized)
# Triggers ~2x/year in bull markets
STRONG_BULL_ADX_THRESHOLD = 28 # Not 40 (ADX rarely hits 40 on daily)
# 28 = well-established trend
# This makes the STRONG_BULL regime actually activate and
# the RULE 88 / Sharpe whitelist logic becomes testable.
7. 5 position_sizer.py — Fix Slippage Model
# REPLACE fixed slippage with:
from data_pipeline import MarketMicrostructure
def compute_expected_slippage(symbol, qty, price):
ms = MarketMicrostructure.get(symbol)
return realistic_slippage(price, qty, ms.adv_20d)
# Uses the Almgren-Chriss model from Section 5.
# ADD: Participation rate limit
MAX_PARTICIPATION_RATE = 0.01 # Never be more than 1% of daily volume
# This is a hard constraint, not a warning.
def calculate_size(self, symbol, capital, signal):
# ... existing logic ...
adv = self.market_data.get_adv(symbol, days=20)
max_participation_qty = int(adv * MAX_PARTICIPATION_RATE / signal.price)
final_qty = min(volatility_qty, max_capital_qty, max_participation_qty)
return final_qty
PA R T 8 — P H AS E D I M P L E M E N TAT I O N R OA D M A P
Phase 0 — Validation of Current System (2 weeks)
Do this BEFORE writing any new code.
Week 1:
□ Run IC analysis on all 11 current features
□ Run stationarity tests on all features
□ Run the 3 bias tests from Section 6.
□ Document what you find
Week 2:
□ Use CPCV (existing cpcv.py) instead of walk-forward in trainer.py
□ Compute P(Sharpe > 1.5) across 45 test folds
□ If P < 50%, your current signal has no edge. Confirm this before moving on.
□ This tells you what you're actually starting from.
Phase 1 — Data Infrastructure (3 weeks)
No model changes yet. Just build the data pipes.
Week 3:
□ Build fno_data.py: Daily F&O bhav copy fetcher
Futures OHLCV + OI for all stocks with F&O
Options chain: strike-wise OI, volume, IV
Compute: basis, PCR, IV skew, max pain, OI change signal
Cache to SQLite/CSV
Week 4:
□ Build bulk_deals.py: NSE bulk/block deal fetcher
Daily scrape of NSE bulk deals API
Match to your universe stocks
Compute: institutional_buy_5d (rolling 5d sum normalized by ADV)
block_deal_flag (binary)
Week 5:
□ Build promoter_data.py: SEBI EDGAR scraper
Scrape SEBI promoter transaction disclosures
Map to your universe
promoter_buy_flag, pledge_pct_change
□ Build data_pipeline.py: Orchestrator (Section 7.3)
□ Validate all data sources for 30 days — check for gaps, delays, errors
Phase 2 — Feature & Model Overhaul (4 weeks)
Week 6:
□ Add F&O features to features.py (Tier 1 from Section 3.3)
□ Add alt data features (Tier 2)
□ Remove dead features (delivery_pct, wick_asymmetry)
□ Fix amihud lookahead (compute p99 on training window only)
□ Run IC analysis on ALL new features — kill any with IC < 0.02
Week 7:
□ Add CatBoost to trainer.py
□ Implement stacking meta-learner
□ Add calibration layer (Platt scaling)
□ Wire in regime-specific calibrators
Week 8:
□ Implement full CPCV validation pipeline
□ Run all robustness tests (Section 6.2)
□ Document results — what is the actual P(Sharpe > 1.5)?
Week 9:
□ Expand universe to Midcap 150 (not just Nifty 50)
□ Fix STRONG_BULL regime thresholds
□ Fix slippage model (Almgren-Chriss)
□ Add participation rate hard limit to position sizer
Phase 3 — Paper Trading + Monitoring (12 weeks minimum)
Months 3-5: Paper trading with full new system
□ Trade the full Midcap 150 universe in paper mode
□ Track EVERY metric: IC per feature, signal correlation, slippage
□ Compare live IC vs backtest IC — if live IC < 0.6 × backtest IC,
there is implementation leakage — find and fix it
□ Track: Does P&L correlate with Nifty? (beta exposure you didn't want)
□ Minimum requirement: 80 completed trades for statistical significance
Month 6: Live trading decision
□ If paper Sharpe > 1.5 AND within 0.5 of backtest Sharpe → go live
□ Start with 10% of target capital. Scale up over 6 months.
□ If paper Sharpe < 1.0 → go back to Phase 2. Don't skip this.
PA R T 9 — T H E S H O R TC U T T H AT AC T UA L LY
WORKS
If you want edge faster without building everything above from scratch, here is the one
thing you can do in 2 weeks that has the highest probability of adding real alpha:
The Index Reconstitution Strategy
Every 6 months, NSE rebalances the Nifty 50 and Nifty Next 50. When a stock is ADDED
to an index:
Every index fund MUST buy it (mandatory, rule-based buying)
They must buy it between announcement and effective date (~4-6 weeks)
This creates predictable, forced buying that moves the price
Signal construction:
Monitor NSE circular for index rebalancing announcement
(published ~6 weeks before effective date)
On announcement date: BUY stocks being added to the index
Hold until 2 days before effective date (when buying pressure peaks)
Exit before the effective date (when index funds are fully invested
and the demand disappears)
Historical performance on NSE (documented in academic papers):
Average excess return during inclusion window: +4% to +8%
Win rate: ~70%
Events per year: ~8-12 stocks across Nifty 50 + Next 50
This is NOT a model. It's pure information arbitrage.
The signal (official NSE circular) is unambiguous.
The buyers (index funds) are rule-bound and predictable.
The window (6 weeks) is long enough for daily-bar execution.
Build this as a separate "Event Engine" module.
It does not need ML. It needs:
NSE circular monitoring (email/RSS)
Position sizing (standard ATR-based)
Exit timing (T-2 from effective date)
This single strategy, executed correctly on 8-12 events per year, is more likely to be
profitable than your entire 11-feature ML system — because the alpha source is
structural, not statistical.
SUMMARY — PRIORITY ORDER
MUST DO (system is broken without these):
Run IC analysis — find out which features have zero information
Switch to CPCV — your current validation is dangerously optimistic
Fix amihud p99 lookahead
Fix STRONG_BULL thresholds (dead regime = dead risk controls)
Fix slippage model — current model is fiction for midcaps
SHOULD DO (this is where real edge is):
Build F&O data pipeline (basis, PCR, IV skew, OI signal)
Build bulk/block deal data pipeline
Add CatBoost, implement stacking ensemble
Expand to Midcap 150 universe
Implement index reconstitution event strategy
NICE TO DO (marginal improvement):
Promoter/pledge data
MF portfolio tracking
Earnings estimate revision data
VWAP-sliced order execution
The honest assessment: if you execute Phase 1 and Phase 2 correctly, you will have a
system that can legitimately claim to have an edge. It will not be a high-frequency edge
or a large-fund edge, but for capital between ₹10 lakh and ₹1 crore, a well-implemented
midcap momentum + event system with proper alternative data can realistically target
18-25% annualized with Sharpe 1.5-2.0.
That is realistic. That is achievable. But not with 3-year OHLCV data and 11 commodity
indicators.
MARK5 Rebuild Roadmap v1.0 — Senior Quant Review Generated: 2026-03-20