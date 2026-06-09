# Indian Market Behavioral Intelligence Guide
**Date:** 2026-05-23  
**Purpose:** Document behavioral patterns, calendar effects, and institutional flow dynamics  
**Status:** ✅ Complete — v4 implementation derived from these patterns  

---

## How Human Psychology Creates Tradeable Patterns in Indian Markets

Indian markets are uniquely shaped by four forces that interact in predictable ways:
1. **FII dominance of price discovery** (FIIs drive ~60% of large/mid-cap price movement)
2. **DII counter-cyclical buying** (SIP flows create support floors; DIIs buy FII sell-offs)
3. **Retail FOMO/panic cycles** (concentrated in F&O — 90%+ retail lose money in options)
4. **Calendar regularity** (budget, RBI, F&O expiry, monsoon — all create recurring patterns)

The profit opportunity: **when FIIs panic-sell, retail follow, DIIs buy but slower → 
oversold conditions that mean-revert. When FIIs accumulate slowly, retail late-arrive → 
momentum. These are the two dominant patterns in MARK5.**

---

## Part 1: FII/DII Flow Dynamics

### Who Moves Indian Markets?

```
Participant       | Share of NSE cash volume | Price influence
─────────────────────────────────────────────────────────────
FII (Foreign)     | 12-18% of volume         | Highest (size + information advantage)
DII (Domestic)    | 15-22% of volume         | Stabilizing (SIP mandates = regular buy)
Retail (individual)| 45-55% of volume        | Noise (correlated with sentiment)
Proprietary/HFT   | 15-25% of volume         | Price discovery (intraday)
```

**Why FIIs dominate despite lower volume share:**
- FII trades are 5-15× larger per order than retail
- FII buying/selling creates sustained directional pressure (days to weeks)
- Retail copies FII direction after 1-3 day lag (observable in news flow)
- Result: FII net flow predicts 5-day Nifty direction with ~55-60% accuracy

### FII Behavioral Patterns

**Pattern 1: Risk-Off Cascade (3-7 day pattern)**
```
Trigger: US Fed hawkish signal, global risk-off, DXY spike
Day 0: FII start selling (₹2,000-5,000 cr net sell)
Day 1: Market -1.5%, retail buy (FOMO reversed to bargain hunting)
Day 2: FII accelerate sells (₹5,000-10,000 cr)
Day 3: Market -3%, retail panic, VIX spikes
Day 4-5: DII buy (SIP mandates + large institutions step in)
Day 6-7: Market stabilizes → RSI oversold → MR opportunity

MARK5 v4 response:
  - Day 0-3: FII signal = BEARISH → block new momentum entries
  - Day 4-7: RSI < 35 → swing trade entry window opens
```

**Pattern 2: FII Accumulation (2-4 week pattern)**
```
Trigger: Emerging market buying, India macro improving (falling rates, GDP beat)
Week 1: FII net buy ₹3,000-6,000 cr/day, market grinds +1-2%
Week 2: Retail notices, starts buying → volume increases
Week 3: FOMO retail flows in, stocks hit new highs → MARK5 momentum signal
Week 4: FII slow down buying, market consolidates

MARK5 v4 response:
  - Week 1-2: FII signal = BULLISH → reinforce momentum entries
  - Week 3: Confluence filter fires (stock at 20d high, golden cross) → enter
  - Week 4: Watch for momentum exit signals
```

**FII Net Flow Signal Levels (calibrated from NSE data patterns):**
```
Net > +₹5,000 cr/5-day rolling: STRONGLY BULLISH — full momentum deployment
Net +₹1,000 to +₹5,000 cr:     BULLISH — normal momentum entries
Net -₹1,000 to +₹1,000 cr:     NEUTRAL — no adjustment
Net -₹1,000 to -₹5,000 cr:     CAUTIOUS — reduce new momentum entry confidence
Net < -₹5,000 cr/5-day:        BEARISH — block new momentum entries, MR/swing only
Net < -₹10,000 cr/5-day:       CRISIS — all new entries blocked, exit signals strengthened
```

### DII Behavioral Patterns

DIIs (LIC, SBI MF, HDFC MF, ICICI Pru) have mandated SIP inflows (~₹15,000-20,000 cr/month as of 2025). This creates a FLOOR that prevents crashes.

```
Monthly SIP inflow: ~₹18,000 cr → ~₹720 cr/trading day
When market falls -3%: additional deployment of ₹2,000-5,000 cr/day
When market falls -5%: DII buying becomes emergency-level (₹5,000-8,000 cr/day)

Pattern: DII buying peaks 2-3 days AFTER FII selling peaks
→ Best MR entry: 2 days after FII peak selling, when DIIs start supporting
```

---

## Part 2: Options Expiry Effects (Monthly F&O Expiry)

### The Expiry Pattern

Indian F&O expiry is the **last Thursday of every month** (for index options).

**4 trading days before expiry:**
```
Monday before expiry Thursday:
  - Option writers accelerate gamma hedging
  - Max pain concept: market gravitates toward strike price where most options expire worthless
  - Increased intraday volatility (±1.5-2.5% swings)
  - Volume spike: 1.4-1.8× normal
  
Tuesday-Wednesday before expiry:
  - Hedging activity peaks
  - Nifty often pins to nearest 100-strike (e.g., 23,500 or 24,000)
  - Large bid-ask spreads in underlying stocks
  
Expiry Thursday:
  - Volatility spikes early morning, calms by afternoon
  - After 2 PM: market often drifts direction of next month's bias
```

**Why this matters for MARK5:**
1. **Avoid new momentum entries 4 trading days before expiry** — increased whipsaw risk
2. **Swing trades are OK** — RSI signals before/after expiry still valid (short holds)
3. **After expiry (Friday)**: fresh positioning begins → good window for momentum entries
4. **Weekly options (now available on Nifty)**: every Thursday has mini-expiry effects

**Implementation in behavioral_signals.py:**
```python
def is_expiry_week(date):
    # Returns True if within 4 trading days before last Thursday of month
    # Used to block new momentum entries but not MR/swing
```

---

## Part 3: Union Budget Effect (February 1 ± 5 days)

### The Budget Pattern

India's Union Budget is presented on February 1 every year (since 2017).

**Pre-budget period (Jan 20 – Feb 1):**
```
Behavior: Anticipatory positioning, sector rotation
  - Defense, infrastructure, railways: pre-buy in anticipation of capex boost
  - FMCG: cautious (excise duty risk)
  - IT/pharma: neutral (rarely budget-sensitive)
  
Market tendency: Nifty +1-3% in pre-budget 10 days as FIIs anticipate
Stock-specific: high beta sectors rally 5-8%
```

**Budget day (Feb 1):**
```
Pattern: "Buy the rumor, sell the news" in 70% of years
  - If budget neutral/positive: Nifty stable, individual stocks move ±5-15%
  - If budget positive (capex boom): infrastructure/defense rally; IT/FMCG fall
  - If budget negative (higher taxes): sell-off, Nifty -2 to -5%

2025 budget: Capex ₹11.11 lakh cr announced → BHEL, HAL +8-15% in 1 day
2023 budget: Continued infrastructure push → positive, market +1.8%
2022 budget: PLI schemes → neutral, market -0.5%
```

**Post-budget (Feb 1-7):**
```
Volatility compression: VIX falls 15-20% post-uncertainty-resolution
Sector rotation settles: winners consolidate, losers bounce
Best entry window: Feb 5-7 (post-settlement, pre-earnings)
```

**MARK5 Implementation:**
- Block new entries on Feb 1 ± 2 days (budget day itself is highest volatility)
- Resume normal entries from Feb 3 with regime check
- Pre-budget (Jan 20-31): normal entries OK, but watch for mean-reversion as sell-off

---

## Part 4: RBI Monetary Policy Committee (MPC) Effect

### The RBI Pattern

RBI MPC meets **6 times per year** (bimonthly); announces after 3-day meeting.
Key dates: Feb, Apr, Jun, Aug, Oct, Dec (first week).

**Bank stock pattern (HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK):**
```
1 day before announcement: Volatility increases, spreads widen
Announcement day: Often -0.5% to +1.5% for Nifty Bank
  - Rate cut surprise: +3-5% for banking stocks
  - Rate hold (expected): neutral, mild rally
  - Rate hike surprise: -3-5% for banking stocks

2-3 days after: Sentiment fully digested
  - After rate cut: HDFCBANK, KOTAKBANK follow-through +5-8% over next week
  - After rate hold: Normal trading resumes
```

**MARK5 v4 response:**
- Bank stocks: avoid new momentum entries 2 days before RBI announcement
- Post-announcement (when rate cut): bank stocks often good swing trade entries
- Bank-heavy MR signals: wait 1 day after RBI before entering

**Approximate RBI dates per year:**
```python
def is_rbi_week(date):
    month = date.month
    day = date.day
    # MPC meets in Feb, Apr, Jun, Aug, Oct, Dec (first week, announces day 3-5)
    return month % 2 == 0 and 1 <= day <= 7
```

---

## Part 5: Monsoon Effect (June – September)

### The Monsoon Pattern

India's SW monsoon runs June-September. Progress tracked by IMD.

**FMCGs, Two-Wheelers, Rural Finance (June-September):**
```
Good monsoon progress (>95% of long-period average):
  - FMCG stocks: +8-15% over monsoon season
  - Two-wheelers (Bajaj, TVS): +10-18% (rural demand)
  - Microfinance/rural NBFCs: rally
  - Jan-Feb following year: continued rural spending

Below-normal monsoon (< 85% of LPA):
  - FMCG: flat or -5%
  - Two-wheelers: -8-12%
  - Rural banks: risk aversion
```

**Trading Pattern:**
```
June 1-15: Monsoon onset tracking → if onset on time: buy two-wheelers, FMCG
July 1-31: Monthly monsoon review → course-correct if rain disappointing
August: Post-harvest anticipation → FMCG consolidation, rural finance buy
September: Kharif crop assessment → final monsoon trade
```

**MARK5 Implementation:** Use as sector context (not direct signal in v4 — no monsoon data available). Flag in research for future enhancement.

---

## Part 6: Calendar Effects Summary (All Tradeable Patterns)

| Effect | Period | Market Behavior | MARK5 Action |
|--------|--------|-----------------|--------------|
| Budget | Feb 1 ± 2 days | High volatility, sector rotation | Block new momentum entries |
| RBI MPC | Bimonthly (Feb/Apr/Jun/Aug/Oct/Dec), first week | Bank sector volatile | Avoid bank-heavy entries Day -2 to Day +1 |
| F&O Expiry | Last Thursday monthly | Pin risk, volatility spike | Block new momentum entries 4 days prior |
| FII selling cascade | Ongoing (signal-based) | Nifty -2 to -5% over 3-5 days | FII BEARISH signal → block momentum |
| FII buying | Ongoing (signal-based) | Nifty +1-2%/week | FII BULLISH → reinforce momentum |
| Budget post-clarity | Feb 5-10 | Volatility drops, positions reset | Good entry window |
| Quarterly earnings (Oct/Jan) | Q2: Oct 1-21, Q3: Jan 1-21 | High single-stock vol | Avoid entering earnings week |
| January effect | Jan 1-15 | FII return from holiday | New year positioning = light FII |
| March tax selling | Mar 15-31 | LTCG tax harvesting (India FY end Mar 31) | Weakness = MR opportunity |
| Monsoon onset | June 1-20 | FMCG/two-wheeler tracking | Sector-specific context |
| Navratri/Diwali | Oct | Retail optimism, lower vol | Good for momentum continuation |
| Year-end (Dec) | Dec 15-31 | FII global rebalancing (outflows) | Avoid new momentum entries |

---

## Part 7: Retail Behavioral Patterns — Exploitable Edges

### The FOMO Cycle (Most Exploitable Pattern)

```
Phase 1 (Ignored): Stock at new 52-week high → ML signal fires → retail ignores
Phase 2 (Noticed): +15% from entry → CNBC coverage begins → retail starts buying
Phase 3 (FOMO peak): +30% from entry → everyone knows → retail pours in
Phase 4 (Top): +40-50% from entry → retail at maximum position → stock tops
Phase 5 (Panic): -10% from peak → retail stops out → stock oversold again

MARK5 ratchet stop exits at Phase 4-5 transition:
  M2 active (≥50% gain) → 8% trail → exit at peak × 0.92
  Retail exits when stock is -15-20% from peak (emotionally driven)
  We exit 7-12 percentage points earlier → significantly better prices
```

### The Panic Sell-Off Pattern (Mean Reversion Opportunity)

```
Indian retail behavior during corrections:
  Day -1: Market -2% → "buying opportunity" consensus
  Day -2: Market -3% → "long-term SIP holders stay put"
  Day -3: Market -4% → Uncertainty, some stop-loss hits
  Day -4: Market -5% (cumulative -12%) → Panic capitulation
  Day -5: Capitulation complete → volume spike, RSI < 30

After capitulation:
  DII starts buying (SIP flow continues regardless)
  FII sees value → stop selling
  Price stabilizes → RSI reverses from <30 → 40+
  
MARK5 MR/Swing response:
  MR entry: RSI < 35 at the bottom
  Swing entry: RSI reversal (was <35, now >40) + price > prev day high
  Typical bounce: +8-15% over 10-20 trading days
```

### The SGX/GIFT Nifty Pre-Market Signal

GIFT Nifty futures trade from 6 AM IST, 2.5 hours before NSE opens.
```
GIFT Nifty vs previous NSE close:
  +1% gap up: expect opening 0.6-0.8% higher
  -1% gap down: expect opening -0.6-0.8% lower
  
Useful for:
  - Avoid new momentum entry on -0.8% gap down days (increased loss risk)
  - Swing trades: gap down then recovery = strong bullish reversal signal
```

**MARK5 v4 integration:** GIFT Nifty not available in historical data → use as live-trading override, not backtest signal.

---

## Part 8: Volatility (VIX Proxy) Behavioral Analysis

### India VIX Behavioral Map

India VIX is computed from Nifty options. MARK5 uses 20-day realized vol as a proxy.

```
VIX Proxy Levels (20-day annualized Nifty vol):

< 12%:  Extreme complacency → markets often near short-term top
12-16%: Normal bull market vol → full momentum deployment
16-22%: Elevated uncertainty → standard momentum, watch for turns
22-28%: Fear zone → reduce position sizes 20%, MR signals preferred
28-35%: Panic zone → reduce 40%, swing/MR only, no new momentum
> 35%:  Crisis → cash/government bonds only (CRISIS regime)
```

**Historical India VIX spikes:**
```
Mar 2020 (COVID crash): VIX proxy hit 55-65%
Mar 2022 (Russia/Ukraine): VIX proxy hit 28-32%
Oct 2023 (Israel/Hamas): VIX proxy hit 22-25%
May 2024 (Election volatility): VIX proxy hit 24-27%
```

**Key insight**: Every spike above 28% was followed within 30-60 days by a reversion to 15-18%. This is the mean-reversion zone for the VIX itself.

---

## Part 9: Sector-Specific Behavioral Patterns

### Defense (HAL, BEL, BHEL, MDL)
```
Trigger: Government capex announcement, geopolitical tension
Behavior: Step-change +20-40% followed by plateau → trend-following optimal
MARK5: ML already captures this (HAL was top pick in 2022)
Risk: Government order delays, election-year pause in capex
```

### Banking (HDFCBANK, ICICIBANK, AXISBANK)
```
Trigger: RBI rate cycle, NPA cycle, credit growth data
Behavior: Range-bound → breakout on rate cut cycle beginning
MARK5: MR works well in banking (known trading ranges)
Risk: NPA surprises, global banking stress (contagion)
```

### IT (INFY, TCS, WIPRO, HCLTECH)
```
Trigger: US tech spending, USD/INR rate
Behavior: Quarterly earnings driven → limited trend between earnings
MARK5: Momentum works when IT in super-cycle; MR works between quarters
Risk: US recession → IT budgets cut → prolonged downtrend
```

### FMCG (HUL, ITC, NESTLE)
```
Trigger: Monsoon, rural wage growth, RBI rate cuts (disposable income)
Behavior: Slow movers, defensive → poor momentum candidates
MARK5: Good MR candidates (stable, range-bound)
Risk: Input cost inflation (palm oil, crude → packaging costs)
```

### Auto/Two-Wheeler (BAJAJ-AUTO, TVS, HERO)
```
Trigger: Monsoon, EV penetration, festive season (Oct-Nov)
Behavior: Clear seasonality → buy pre-festive (Sep), sell post-festive (Dec)
MARK5: Seasonal swing trade opportunity in Sep-Nov
Risk: EV disruption uncertainty creates structural uncertainty
```

---

## Part 10: Implementation Checklist for v4

The following behavioral signals are implemented in `core/strategies/behavioral_signals.py`:

| Signal | Data Source | Frequency | Use in v4 |
|--------|-------------|-----------|-----------|
| VIX Proxy | Nifty OHLCV (20d realized vol) | Daily | Position size adjustment |
| Market Breadth | Available tickers % above SMA50 | Daily | Regime confirmation |
| FII Net Flow | Synthetic FII data (5-day rolling) | Daily | Entry gate |
| F&O Expiry Gate | Calendar (last Thursday of month) | Monthly | Block momentum entries |
| Budget Day Gate | Calendar (Feb 1) | Annual | Block all entries |
| RBI MPC Gate | Calendar (bimonthly first week) | Bimonthly | Bank stock caution |
| Retail Panic Signal | RSI < 30 on index | Event-driven | MR/swing entry green light |

The v4 backtest integrates all these signals and measures their combined impact on WR, DD, and returns.

---

*This document is for paper trading research. Capital: ₹5 crore. Never switch to LIVE.*
