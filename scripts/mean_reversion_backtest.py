"""
MARK5 — Mean Reversion Strategy Backtest  (v2 — quality universe)
=================================================================
Standalone OOS backtest (2022-2026) targeting oversold bounces in
HIGH-QUALITY NSE stocks only.

Design rationale:
  - RSI < 30 + strict price > SMA200: ensures we're catching corrections
    in uptrending quality stocks, NOT catching falling knives in downtrends.
  - Universe restricted to 8 stocks with institutional (DII/SIP) buying floors:
    Banking (HDFCBANK, ICICIBANK, KOTAKBANK), Pharma (SUNPHARMA, LUPIN),
    IT (TCS, INFY), NBFC (BAJFINANCE).
    These have the most reliable bounce patterns due to structured institutional
    demand at dip levels (documented in MARK5 behavioral research).
  - ATR-based stop loss: adapts to each stock's realized volatility rather than
    a fixed 8% that gets hit by routine NSE intraday noise.
  - Complementary to the momentum portfolio: MR targets DIFFERENT stocks in
    DIFFERENT regimes (price below momentum threshold, RSI oversold).

Key changes vs v1:
  - Universe: 33 → 8 quality stocks
  - RSI threshold: < 35 → < 30 (confirmed oversold)
  - SMA200 filter: "within 30% below" → strict "price > SMA200"
  - Stop loss: fixed 8% → ATR-based (2×ATR14, min 4%, max 10%)
  - Take profit: +12% → +10% (faster lock-in; R:R still 1.5:1+ with ATR stop)
  - Position size: 10% → 15% of MR capital (higher conviction per quality stock)
  - Max concurrent: 3 → 4 positions

PAPER MODE ONLY — never switch to LIVE.
"""
import os, sys, json, logging, warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MARK5.MRBacktest")

# ── Quality universe (institutional floor stocks only) ───────────────────────
# Rationale for each:
#   Banking: DII/SIP buying floor is strongest here. FII cascade → DII absorbs.
#   Pharma:  FDA event-driven dips recover reliably once overhang clears.
#   IT:      Rate-sensitive but earnings-floor prevents prolonged declines.
#   NBFC:    BAJFINANCE has the cleanest bounce patterns due to brand premium.
MR_UNIVERSE = [
    "HDFCBANK",   # Banking — largest SIP absorber
    "ICICIBANK",  # Banking — strong NIM recovery
    "KOTAKBANK",  # Banking — quality book, premium valuation floor
    "SUNPHARMA",  # Pharma — FDA bounce, domestic rx resilience
    "LUPIN",      # Pharma — generic recovery after approval
    "TCS",        # IT — buy-on-dips due to dividend yield floor
    "INFY",       # IT — ADR parity limits excessive downside
    "BAJFINANCE", # NBFC — brand premium, retail AUM floor
]

# ── Config ────────────────────────────────────────────────────────────────────
MR_CAPITAL        = 1_00_00_000.0   # ₹1 crore dedicated MR pool
MAX_MR_POSITIONS  = 4               # max 4 concurrent positions (quality focus)
MR_POSITION_PCT   = 0.15            # 15% of MR capital per trade

TAKE_PROFIT_PCT   = 0.10            # +10% take profit (achievable R:R target)
ATR_STOP_MULT     = 2.0             # stop = 2 × ATR14 from entry
MIN_STOP_PCT      = 0.04            # minimum stop distance 4% (prevent whipsaws)
MAX_STOP_PCT      = 0.10            # maximum stop distance 10% (risk cap)
TIME_STOP_BARS    = 25              # exit after 25 bars if no target hit
RSI_EXIT_LEVEL    = 65.0            # overbought exit (earlier capture)
ENTRY_COOLDOWN    = 15              # bars between MR entries on same ticker

# Entry filters — strict quality criteria
RSI_ENTRY_MAX      = 30.0   # RSI must be < 30 (confirmed oversold)
FALL_FROM_52W_MIN  = 0.10   # min 10% fall from 52-week high
FALL_FROM_52W_MAX  = 0.45   # max 45% fall (beyond = structural breakdown)
VOLUME_CONFIRM_MIN = 1.2    # entry day volume ≥ 1.2× 20-day average

COST_PCT      = 0.0029
SLIPPAGE_PCT  = 0.001
LTCG_RATE     = 0.125
STCG_RATE     = 0.200
LTCG_EXEMPT   = 125_000

OOS_START = "2022-01-01"
OOS_END   = "2026-05-21"
CACHE_DIR = os.path.join(_ROOT, "data", "cache")
REPORTS_DIR = os.path.join(_ROOT, "reports")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_cache(ticker: str) -> Optional[pd.DataFrame]:
    for suffix in ["_daily.parquet", "_NS_1d.parquet"]:
        path = os.path.join(CACHE_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                from zoneinfo import ZoneInfo
                df.index = df.index.tz_convert(ZoneInfo("Asia/Kolkata")).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df
    return None


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised RSI series."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr14(df: pd.DataFrame) -> pd.Series:
    """14-day Average True Range series (as fraction of close)."""
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    cl = df["close"].astype(float)
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)
    cl_safe = cl.replace(0, float("nan"))
    return (tr.ewm(span=14, adjust=False).mean() / cl_safe).fillna(0.02)


def compute_tax(trades: List[Dict]) -> float:
    by_year: Dict[int, Dict[str, float]] = {}
    for t in trades:
        if t["net_pnl"] <= 0:
            continue
        yr = t["exit_date"].year
        if yr not in by_year:
            by_year[yr] = {"ltcg": 0.0, "stcg": 0.0}
        if t["hold_days"] > 365:
            by_year[yr]["ltcg"] += t["net_pnl"]
        else:
            by_year[yr]["stcg"] += t["net_pnl"]
    total_tax = 0.0
    for yr, gains in by_year.items():
        ltcg_gain  = max(0.0, gains["ltcg"] - LTCG_EXEMPT)
        total_tax += ltcg_gain * LTCG_RATE + gains["stcg"] * STCG_RATE
    return total_tax


# ── Portfolio ─────────────────────────────────────────────────────────────────
class MRPortfolio:
    def __init__(self):
        self.cash = MR_CAPITAL
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []

    def get_equity(self, prices: Dict[str, float]) -> float:
        pos_val = sum(
            p["shares"] * prices.get(tk, p["entry_price"])
            for tk, p in self.positions.items()
        )
        return self.cash + pos_val

    def enter(self, ticker: str, price: float, date: pd.Timestamp,
              bar_idx: int, atr_pct: float):
        if ticker in self.positions or len(self.positions) >= MAX_MR_POSITIONS:
            return
        alloc = min(self.cash * MR_POSITION_PCT, self.cash * 0.95)
        if alloc < 5_000:
            return
        fill   = price * (1 + SLIPPAGE_PCT)
        shares = int(alloc / fill)
        if shares < 1:
            return
        spent  = shares * fill * (1 + COST_PCT)
        if spent > self.cash:
            return
        # ATR-based stop price
        stop_dist = max(MIN_STOP_PCT, min(MAX_STOP_PCT, ATR_STOP_MULT * atr_pct))
        self.cash -= spent
        self.positions[ticker] = {
            "shares":      shares,
            "entry_price": fill,
            "entry_date":  date,
            "entry_bar":   bar_idx,
            "entry_total": spent,
            "stop_price":  fill * (1.0 - stop_dist),
        }
        logger.info(
            f"MR ENTER {ticker} @₹{fill:.0f} ×{shares} "
            f"stop={stop_dist:.1%} on {date.date()}"
        )

    def exit(self, ticker: str, price: float, date: pd.Timestamp, reason: str):
        if ticker not in self.positions:
            return
        pos = self.positions.pop(ticker)
        fill     = price * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * fill * (1 - COST_PCT)
        net_pnl  = proceeds - pos["entry_total"]
        self.cash += proceeds
        hold_days = (date - pos["entry_date"]).days
        self.trades.append({
            "ticker":       ticker,
            "entry_date":   pos["entry_date"], "exit_date": date,
            "entry_price":  pos["entry_price"], "exit_price": fill,
            "shares":       pos["shares"], "net_pnl": net_pnl,
            "pnl_pct":      net_pnl / pos["entry_total"] * 100,
            "hold_days":    hold_days, "reason": reason,
        })
        logger.info(f"MR EXIT {ticker} @₹{fill:.0f} ({reason}) pnl={net_pnl/1e5:.2f}L")


# ── Main simulation ───────────────────────────────────────────────────────────
def run_mr_backtest():
    print(f"\n{'═'*80}")
    print(f"  MEAN REVERSION BACKTEST v2 — OOS {OOS_START} → {OOS_END}")
    print(f"  Universe: {len(MR_UNIVERSE)} quality stocks (banking/pharma/IT/NBFC)")
    print(f"  Capital: ₹{MR_CAPITAL/1e7:.0f}cr  |  Max positions: {MAX_MR_POSITIONS}")
    print(f"  Entry: RSI<{RSI_ENTRY_MAX} + price>SMA200 + fall {FALL_FROM_52W_MIN:.0%}-{FALL_FROM_52W_MAX:.0%}"
          f" + vol≥{VOLUME_CONFIRM_MIN}×avg")
    print(f"  Exit:  +{TAKE_PROFIT_PCT:.0%} target | {ATR_STOP_MULT}×ATR14 stop "
          f"({MIN_STOP_PCT:.0%}–{MAX_STOP_PCT:.0%}) | {TIME_STOP_BARS}bar max")
    print(f"{'═'*80}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    all_data: Dict[str, pd.DataFrame] = {}
    for tk in MR_UNIVERSE:
        df = load_cache(tk)
        if df is not None:
            all_data[tk] = df
        else:
            print(f"  {tk}: no data (skipped)")

    active_tickers = [tk for tk in MR_UNIVERSE if tk in all_data]
    print(f"  {len(active_tickers)} tickers loaded: {', '.join(active_tickers)}")

    # ── Pre-compute technical indicators ──────────────────────────────────
    print("Pre-computing RSI, ATR, SMA200, 52W-high, volume avg...")
    rsi_series:    Dict[str, pd.Series] = {}
    atr_series:    Dict[str, pd.Series] = {}
    sma200_series: Dict[str, pd.Series] = {}
    high52w_series: Dict[str, pd.Series] = {}
    vol_avg_series: Dict[str, pd.Series] = {}

    for tk in active_tickers:
        df    = all_data[tk]
        close = df["close"].astype(float)
        vol   = df["volume"].astype(float)
        rsi_series[tk]     = rsi(close)
        atr_series[tk]     = atr14(df)
        sma200_series[tk]  = close.rolling(200, min_periods=60).mean()
        high52w_series[tk] = close.rolling(252, min_periods=50).max()
        vol_avg_series[tk] = vol.rolling(20, min_periods=5).mean()

    # ── Build unified calendar ─────────────────────────────────────────────
    _ref = all_data.get("HDFCBANK")
    if _ref is None:
        _ref = all_data[active_tickers[0]]
    all_dates = _ref.loc[OOS_START:OOS_END].index

    portfolio = MRPortfolio()
    _mr_cooldown: Dict[str, int] = {}

    def _get_val(series_dict: Dict, tk: str, date: pd.Timestamp,
                 default: float = 0.0) -> float:
        s = series_dict.get(tk)
        if s is None:
            return default
        try:
            idx = s.index.searchsorted(date, side="right") - 1
            if 0 <= idx < len(s):
                v = s.iloc[idx]
                return float(v) if pd.notna(v) else default
        except Exception:
            pass
        return default

    # ── Main simulation loop ───────────────────────────────────────────────
    for bar_idx, date in enumerate(all_dates):
        # Current prices
        prices: Dict[str, float] = {}
        for tk in active_tickers:
            try:
                prices[tk] = float(all_data[tk].loc[date, "close"])
            except (KeyError, ValueError):
                pass

        # ── Exit checks (daily) ────────────────────────────────────────────
        for tk in list(portfolio.positions.keys()):
            if tk not in prices:
                continue
            pos   = portfolio.positions[tk]
            curr  = prices[tk]
            entry = pos["entry_price"]
            stop  = pos["stop_price"]
            hold_bars = bar_idx - pos["entry_bar"]

            gain_pct = (curr - entry) / entry

            # 1. Take profit
            if gain_pct >= TAKE_PROFIT_PCT:
                portfolio.exit(tk, curr, date, f"TAKE_PROFIT({gain_pct:+.1%})")
                _mr_cooldown[tk] = bar_idx + ENTRY_COOLDOWN
                continue

            # 2. ATR-based stop loss
            if curr <= stop:
                loss_pct = (entry - curr) / entry
                portfolio.exit(tk, curr, date, f"STOP_LOSS({-loss_pct:.1%})")
                _mr_cooldown[tk] = bar_idx + ENTRY_COOLDOWN
                continue

            # 3. Time stop
            if hold_bars >= TIME_STOP_BARS:
                portfolio.exit(tk, curr, date, f"TIME_STOP({hold_bars}bars)")
                _mr_cooldown[tk] = bar_idx + ENTRY_COOLDOWN // 2
                continue

            # 4. RSI overbought exit
            current_rsi = _get_val(rsi_series, tk, date, 50.0)
            if current_rsi >= RSI_EXIT_LEVEL:
                portfolio.exit(tk, curr, date, f"RSI_OB({current_rsi:.0f})")
                _mr_cooldown[tk] = bar_idx + ENTRY_COOLDOWN // 2
                continue

        # ── Equity tracking ────────────────────────────────────────────────
        equity = portfolio.get_equity(prices)
        portfolio.equity_history.append({"date": date, "equity": equity})

        # ── Entry scan (daily — MR can be triggered any day) ──────────────
        if len(portfolio.positions) >= MAX_MR_POSITIONS:
            continue

        mr_candidates = []
        for tk in active_tickers:
            if tk in portfolio.positions:
                continue
            if tk not in prices or prices[tk] <= 0:
                continue
            if bar_idx < _mr_cooldown.get(tk, 0):
                continue

            curr = prices[tk]

            # Condition 1: RSI < 30 (confirmed oversold)
            current_rsi = _get_val(rsi_series, tk, date, 50.0)
            if current_rsi >= RSI_ENTRY_MAX:
                continue

            # Condition 2: Price STRICTLY above SMA(200)
            # This is the critical quality filter — no catching falling knives
            sma200 = _get_val(sma200_series, tk, date, 0.0)
            if sma200 <= 0 or curr <= sma200:
                continue

            # Condition 3: Fallen 10-45% from 52-week high
            # (correction in uptrend, not crash or structural breakdown)
            h52w = _get_val(high52w_series, tk, date, 0.0)
            if h52w <= 0:
                continue
            fall_pct = (h52w - curr) / h52w
            if not (FALL_FROM_52W_MIN <= fall_pct <= FALL_FROM_52W_MAX):
                continue

            # Condition 4: Volume ≥ 1.2× 20-day average (capitulation confirmation)
            try:
                vol_today = float(all_data[tk].loc[date, "volume"])
                vol_avg   = _get_val(vol_avg_series, tk, date, 0.0)
                if vol_avg > 0 and vol_today < VOLUME_CONFIRM_MIN * vol_avg:
                    continue
            except Exception:
                pass  # If no volume data, skip the volume check gracefully

            # All conditions met — collect with RSI as sort key
            atr_pct = _get_val(atr_series, tk, date, 0.02)
            mr_candidates.append((tk, current_rsi, fall_pct, atr_pct))

        # Sort by most oversold (lowest RSI first → strongest bounce signal)
        mr_candidates.sort(key=lambda x: x[1])
        for tk, rsi_val, fall_val, atr_val in mr_candidates:
            if len(portfolio.positions) >= MAX_MR_POSITIONS:
                break
            curr = prices[tk]
            logger.info(
                f"MR SIGNAL {tk} on {date.date()}: "
                f"RSI={rsi_val:.1f} fall={fall_val:.1%} ATR={atr_val:.2%}"
            )
            portfolio.enter(tk, curr, date, bar_idx, atr_val)

    # ── Close open positions at end ────────────────────────────────────────
    last_date = all_dates[-1]
    for tk in list(portfolio.positions.keys()):
        try:
            lp = float(all_data[tk].loc[last_date, "close"])
        except Exception:
            lp = portfolio.positions[tk]["entry_price"]
        portfolio.exit(tk, lp, last_date, "END_OF_SIM")

    final_equity = portfolio.get_equity({})

    # ── Performance ────────────────────────────────────────────────────────
    eq_df  = pd.DataFrame(portfolio.equity_history).set_index("date")
    years  = (all_dates[-1] - all_dates[0]).days / 365.25
    gross_ret  = (final_equity / MR_CAPITAL - 1)
    gross_cagr = (1 + gross_ret) ** (1 / years) - 1 if years > 0 else 0.0

    total_tax = compute_tax(portfolio.trades)
    net_final = final_equity - total_tax
    net_ret   = (net_final / MR_CAPITAL - 1)
    net_cagr  = (1 + net_ret) ** (1 / years) - 1 if years > 0 else 0.0

    eq_curve     = eq_df["equity"]
    rolling_max  = eq_curve.cummax()
    dd_curve     = eq_curve / rolling_max - 1
    max_dd       = float(dd_curve.min())
    daily_ret_s  = eq_curve.pct_change().dropna()
    sharpe       = (float(daily_ret_s.mean()) / float(daily_ret_s.std()) * np.sqrt(252)
                    if float(daily_ret_s.std()) > 0 else 0.0)

    wins  = [t for t in portfolio.trades if t["net_pnl"] > 0]
    losses= [t for t in portfolio.trades if t["net_pnl"] <= 0]
    wr    = len(wins) / len(portfolio.trades) * 100 if portfolio.trades else 0
    avg_hold = np.mean([t["hold_days"] for t in portfolio.trades]) if portfolio.trades else 0

    # Annual breakdown
    annual: Dict[int, float] = {}
    for yr in range(2022, 2027):
        yr_mask = eq_df.index.year == yr
        yr_data = eq_curve[yr_mask]
        if len(yr_data) == 0:
            continue
        before_yr = eq_curve[eq_df.index < yr_data.index[0]]
        y_start = float(before_yr.iloc[-1]) if len(before_yr) > 0 else MR_CAPITAL
        y_end   = float(yr_data.iloc[-1])
        annual[yr] = y_end / y_start - 1

    # ── Print results ──────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  MR v2 RESULTS — ₹{MR_CAPITAL/1e7:.0f}cr pool | {OOS_START} to {OOS_END}")
    print(f"{'═'*80}")
    print(f"  ₹{MR_CAPITAL/1e7:.0f}cr → ₹{net_final/1e7:.2f}cr  (net {net_ret*100:+.1f}%)")
    print(f"  Gross CAGR: {gross_cagr*100:+.2f}%  |  Net CAGR: {net_cagr*100:+.2f}%")
    print(f"  Max DD: {max_dd*100:.1f}%  |  Sharpe: {sharpe:.2f}")
    print(f"  Tax: ₹{total_tax/1e5:.1f}L")
    print(f"  Trades: {len(portfolio.trades)}  W:{len(wins)} L:{len(losses)}  WR:{wr:.0f}%  avg hold {avg_hold:.0f}d")
    print(f"\n  Annual returns:")
    for yr, ret in sorted(annual.items()):
        bar = "▓" * int(abs(ret) * 100) if abs(ret) * 100 < 40 else "▓" * 40
        sign = "+" if ret >= 0 else ""
        print(f"    {yr}: {sign}{ret*100:.1f}%  {bar}")

    # ── Exit reason breakdown ──────────────────────────────────────────────
    print(f"\n  Exit reasons:")
    by_reason: Dict[str, Dict] = {}
    for t in portfolio.trades:
        rkey = t["reason"].split("(")[0]
        if rkey not in by_reason:
            by_reason[rkey] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        by_reason[rkey]["count"] += 1
        if t["net_pnl"] > 0:
            by_reason[rkey]["wins"] += 1
        by_reason[rkey]["total_pnl"] += t["net_pnl"]
    for reason, stats in sorted(by_reason.items(), key=lambda x: -x[1]["count"]):
        w_rate = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"    {reason:<20} {stats['count']:3d} trades  WR:{w_rate:.0f}%  "
              f"₹{stats['total_pnl']/1e5:+.1f}L")

    # ── Top 10 trades ──────────────────────────────────────────────────────
    if portfolio.trades:
        print(f"\n  Top 10 trades by PnL:")
        top10 = sorted(portfolio.trades, key=lambda t: t["net_pnl"], reverse=True)[:10]
        for t in top10:
            pnl_l = t["net_pnl"] / 1e5
            print(f"    {t['ticker']:<12} {t['entry_date'].strftime('%Y-%m-%d')}→"
                  f"{t['exit_date'].strftime('%Y-%m-%d')} "
                  f"({t['hold_days']:3d}d) {t['pnl_pct']:>+7.1f}%  "
                  f"₹{pnl_l:>+6.1f}L  [{t['reason']}]")

    # ── Per-ticker summary ─────────────────────────────────────────────────
    print(f"\n  Per-ticker summary:")
    by_tk: Dict[str, Dict] = {}
    for t in portfolio.trades:
        if t["ticker"] not in by_tk:
            by_tk[t["ticker"]] = {"count": 0, "wins": 0, "pnl": 0.0}
        by_tk[t["ticker"]]["count"] += 1
        if t["net_pnl"] > 0:
            by_tk[t["ticker"]]["wins"] += 1
        by_tk[t["ticker"]]["pnl"] += t["net_pnl"]
    for tk in MR_UNIVERSE:
        if tk not in by_tk:
            print(f"    {tk:<12} 0 trades")
            continue
        s = by_tk[tk]
        wr_tk = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        print(f"    {tk:<12} {s['count']:2d} trades  WR:{wr_tk:.0f}%  ₹{s['pnl']/1e5:+.1f}L")

    # Save results
    results = {
        "strategy":    "mean_reversion_v2",
        "universe":    MR_UNIVERSE,
        "gross_cagr":  round(gross_cagr * 100, 2),
        "net_cagr":    round(net_cagr * 100, 2),
        "max_dd":      round(max_dd * 100, 2),
        "sharpe":      round(sharpe, 3),
        "win_rate":    round(wr, 1),
        "n_trades":    len(portfolio.trades),
        "annual":      {str(yr): round(ret * 100, 2) for yr, ret in annual.items()},
        "trades":      [
            {k: (v.isoformat() if hasattr(v, "isoformat") else v)
             for k, v in t.items()}
            for t in portfolio.trades
        ],
    }
    out_path = os.path.join(REPORTS_DIR, "mr_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"{'═'*80}")

    return results


if __name__ == "__main__":
    run_mr_backtest()
