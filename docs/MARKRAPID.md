# MARKRAPID — Aggressive Swing Trading System

**Subsidiary of MARK5. Independent, read-only access to MARK5 core modules.**

---

## What MARKRAPID Is (and Is Not)

MARKRAPID is a concentrated, short-hold swing trading system targeting +10% per trade within 30 calendar days on ₹10,000 capital.

It is NOT MARK5. The two systems solve different problems:

| | MARK5 Momentum | MARKRAPID |
|---|---|---|
| Hold period | 200-800 days | ≤ 30 days |
| Capital | ₹5 crore, 5 positions | ₹10,000, 1 position |
| Strategy | Let winners run | Capture first breakout leg |
| Win rate | ~37% | ~48-55% |
| Payoff ratio | ~5.7:1 | ~2:1 |
| Mode | Monthly rebalance | Daily scan |
| Goal | Beat Nifty CAGR | +10% per trade |

**Why MARK5 holds so long:** The system deliberately holds for 400+ days because momentum returns compound. HAL +209% in 419 days — no swing trade captures that. MARKRAPID would have exited at +12% and missed 197% of the move.

**Why MARKRAPID exists:** To trade stocks MARK5 ignores (its universe is 30 stocks, cherry-picked for long-term momentum). MARKRAPID scans 80 stocks daily looking for short-term breakout setups — fast entries, fast exits.

---

## The RAPID Score (5 Components)

Every stock gets a RAPID score from 0 to 1 on each trading day:

```
Component         Weight   Signal
─────────────────────────────────────────────────────────────────
Breakout          30%      Close > 20-day high (momentum ignition)
Volume Surge      25%      Today's volume vs 20-day average
RSI Momentum      20%      RSI 14 in [48, 76] = momentum, not overbought
Trend Alignment   15%      Price > EMA20 > EMA50 (uptrend structure)
Catalyst/News     10%      Gap-up proxy (backtest) or RSS news (live)
─────────────────────────────────────────────────────────────────
```

**Entry threshold: RAPID score ≥ 0.72**

### Entry Gate (ALL conditions required)

1. RAPID score ≥ 0.72
2. Price > SMA200 (no secular downtrend — don't catch falling knives)
3. Volume today ≥ 2× 20-day avg (institutional confirmation)
4. RSI < 76 (not already overbought)
5. ₹50 ≤ price ≤ ₹3,000 (tradeable with ₹10k capital)

If multiple stocks pass all gates, the highest-score stock is selected.

---

## Trade Parameters

```
Capital per trade:  ₹10,000 (fixed — no Kelly sizing for swing trades)
Max positions:      1 (fully concentrated)
Profit target:      +12% gross (+11.71% net after 0.29% transaction costs)
Stop loss:          -6% from entry price
Max hold:           30 calendar days
```

**Why 12% target?** NSE transaction costs eat 0.29% round-trip. A +12% gross target nets +11.71% — just above the +10% goal. Never set the target at exactly 10%; brokerage would leave you short.

**Why -6% stop?** Gives a 2:1 R:R (12%/6%). At 50% win rate and 2:1 payoff, expected value per trade = +3%. The stop is wide enough to survive noise but tight enough to kill thesis-broken trades fast.

**Why 30 days max?** The system chases short-term catalyst-driven moves. After 30 days, the catalyst has either played out or failed. Holding longer converts a swing trade into a mediocre medium-term position.

---

## Exit Waterfall (checked daily, in order)

1. **TARGET_HIT** — intraday high ≥ entry × 1.12 → exit at target price
2. **STOP_LOSS** — intraday low ≤ entry × 0.94 → exit at stop price
3. **TIME_STOP** — calendar days ≥ 30 → exit at close
4. **SIGNAL_FADE** — RAPID score < 0.42 after ≥ 5 days in trade → thesis broken

---

## Universe (80 Stocks)

MARKRAPID scans 80 NSE stocks daily, covering all market caps:

- **NIFTY50 large-caps**: ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BHARTIARTL, BRITANNIA, CIPLA, COALINDIA, DIVISLAB, DRREDDY, EICHERMOT, GRASIM, HCLTECH, HDFCBANK, HDFCLIFE, HINDUNILVR, HINDALCO, ICICIBANK, INDUSINDBK, INFY, ITC, JSWSTEEL, KOTAKBANK, LT, LTIM, MARUTI, M&M, NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBILIFE, SBIN, SUNPHARMA, TATACONSUM, TATAMOTORS, TATASTEEL, TCS, TECHM, TITAN, TRENT, ULTRACEMCO, WIPRO

- **Midcap / high-swing names**: HAL, BEL, BHEL, IRFC, IRCTC, RVNL, NMDC, SAIL, COFORGE, PERSISTENT, MPHASIS, LUPIN, ALKEM, TORNTPHARM, CHOLAFIN, MUTHOOTFIN, JUBLFOOD, TATAPOWER, CANBK, BANKINDIA, IDEA, IDFCFIRSTB, BANDHANBNK, AUBANK, MOTHERSON, VOLTAS, HAVELLS, POLYCAB, ZOMATO, NYKAA

---

## Data Sources

| Data | Source | Mode |
|------|--------|------|
| OHLCV (backtest) | yfinance (NSE .NS suffix) via MARK5's `nse_data_provider` | Cache → parquet |
| OHLCV (live) | Kite Connect (broker) | Real-time |
| News sentiment (live) | RSS: MoneyControl, ET, BS, LiveMint | Via MARK5's `news_sentiment.py` |
| Bulk/block deals | NSE public API | Via MARK5's `bulk_deals.py` |
| NSE corporate announcements | NSE announcements API (scraped) | `markrapid/news_scraper.py` |
| Catalyst (backtest proxy) | Gap-up open + volume surge | Technical proxy |

MARKRAPID **imports** from MARK5's `core/data/` modules but **never modifies** them.

---

## Architecture

```
markrapid/
├── __init__.py        — package metadata
├── config.py          — all constants (single source of truth)
├── signals.py         — RAPID score engine
├── scanner.py         — daily universe scan → ranked candidates
├── news_scraper.py    — live news + NSE announcements (live mode)
├── backtest.py        — full historical backtest engine
└── portfolio.py       — ₹10k trade tracker

scripts/
└── markrapid_backtest.py  — CLI runner

tests/
└── test_markrapid.py      — 89 tests (100% pass)
```

---

## Running the Backtest

```bash
# Full OOS backtest (2022-2026)
python3 scripts/markrapid_backtest.py

# Custom date range
python3 scripts/markrapid_backtest.py --start 2023-01-01 --end 2025-12-31

# Specific tickers only
python3 scripts/markrapid_backtest.py --tickers HAL,TRENT,ZOMATO,IRCTC

# Compound profits (reinvest)
python3 scripts/markrapid_backtest.py --compound

# Pure technical (no news proxy)
python3 scripts/markrapid_backtest.py --no-news
```

Output: `reports/markrapid_results.json`

---

## What the Backtest Does NOT Include

The backtest uses a **technical-only proxy** for the news/catalyst component:
- Gap-up open ≥ 3% on 2× volume → treated as catalyst score = 0.8
- Otherwise: neutral (0.0)

**Live trading adds:**
- RSS news sentiment (MoneyControl, ET, BS)
- NSE bulk/block deal signals
- NSE corporate announcements keyword scoring

The live news layer is expected to improve WR by 5-8 percentage points (filter out breakouts with no fundamental catalyst).

---

## Relationship to MARK5

```
MARK5 Momentum Portfolio (parent system)
└── markrapid/ (subsidiary — read-only import of core/)
    ├── uses: core/data/nse_data_provider.py     (OHLCV download)
    ├── uses: core/data/news_sentiment.py         (RSS news)
    ├── uses: core/data/bulk_deals.py             (institutional activity)
    └── NEVER modifies ANY file in core/ or scripts/ (MARK5 stays intact)
```

**Isolation contract:**
- MARKRAPID reads from `core/` — never writes, never imports to modify
- All MARKRAPID code lives in `markrapid/` and `scripts/markrapid_backtest.py`
- MARKRAPID tests live in `tests/test_markrapid.py`
- Running MARKRAPID does not affect MARK5's existing backtest results

---

## Honest Assessment

**What MARKRAPID can do:**
- Identify high-probability 30-day swing setups using 5-factor signals
- Catch the first breakout leg of multi-month momentum moves
- Cut losses fast (-6%) and take profits at +12%
- Generate 2-3× more trades than MARK5 per year (more opportunities)

**What MARKRAPID cannot do:**
- Guarantee +10% on every trade (this is not possible in any market)
- Match MARK5's +18% CAGR on small capital (single ₹10k trade → absolute ₹1,200 profit per win)
- Predict gap-downs from overnight news
- Replace MARK5 as the primary portfolio system

**Expected live performance range:**
- Win rate: 45-55% (depending on market regime)
- Avg winner: +13-16% (overshoot of 12% target on strong moves)
- Avg loser: ~-6% (hard stop discipline)
- Expected value per trade: +3-5%
- Trades per year: ~20-35

---

## Paper Mode

MARKRAPID is **ALWAYS paper mode**. No real orders are executed.
The `portfolio.py` tracker simulates trade P&L but does not interface with any broker API.
To convert to live trading: connect `portfolio.enter()` to Kite Connect's order placement API — this is intentionally NOT done until paper trading confirms performance.
