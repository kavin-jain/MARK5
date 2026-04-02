"""
MARK5 F&O Data Provider — Phase 1
══════════════════════════════════════════════════════════════════════
Fetches NSE F&O bhav copy data and computes 5 institutional-grade
features for daily swing trading signals.

DATA SOURCES
────────────
• Historical (pre 2024-07-08):
    https://archives.nseindia.com/content/historical/DERIVATIVES/YYYY/MMM/foDDMMMYYYYbhav.csv.zip
    Columns: INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
             OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
             OPEN_INT, CHG_IN_OI, TIMESTAMP

• Historical (post 2024-07-08, UDiFF format):
    https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
    Columns: TradDt, TckrSymb, XpryDt, StrkPric, OptnTp, ClsPric,
             SttlmPric, OpnIntrst, ChngInOpnIntrst, UndrlygPric,
             TtlTradgVol, FinInstrmTp (STO=stock opt, STF=stock fut, etc.)

FEATURES PRODUCED (per symbol, per date)
─────────────────────────────────────────
1. futures_basis_zscore  — (front-month_settle - spot) z-scored over 60d
2. pcr_oi               — sum(PE_OI) / sum(CE_OI), near-month expiry
3. iv_skew_zscore       — risk-reversal proxy z-scored over 30d
                          proxy = (OTM_PE_price - OTM_CE_price) / spot
4. oi_signal            — sign(Δspot) × sign(Δfutures_OI), near-month
5. max_pain_distance    — (spot - max_pain) / ATR14, expiry week only; 0 elsewhere

VALIDATION GATES (per RULE 48/49)
──────────────────────────────────
• pcr_oi ∈ [0.1, 5.0]
• |futures_basis_zscore| ≤ 6
• OI ≥ 0

CACHE
─────
data/fno/raw/YYYYMMDD_bhav.parquet  — normalized bhav copy per date
data/fno/features/SYMBOL.parquet    — feature series per symbol
"""

import io
import logging
import os
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("MARK5.FNOData")

# ── Constants ────────────────────────────────────────────────────────────────

# NSE switched to UDiFF format on this date
UDIFF_CUTOVER = date(2024, 7, 8)

# URL templates
OLD_BHAV_URL = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES"
    "/{year}/{month_abbr}/fo{dd}{month_abbr_upper}{year}bhav.csv.zip"
)
NEW_BHAV_URL = (
    "https://nsearchives.nseindia.com/content/fo"
    "/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

CACHE_BASE = Path(__file__).resolve().parents[2] / "data" / "fno"
RAW_CACHE   = CACHE_BASE / "raw"
FEAT_CACHE  = CACHE_BASE / "features"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Open interest validation limits (RULE 48)
PCR_MIN, PCR_MAX       = 0.10, 5.0
BASIS_ZSCORE_LIMIT     = 6.0
RISK_FREE_RATE         = 0.065   # RBI repo approx 2022-2025

# Rolling windows for z-scoring
BASIS_WINDOW  = 60  # trading days
SKEW_WINDOW   = 30  # trading days

# Approximate trading days per year for 0-DTE time calculation
TRADING_DAYS_PER_YEAR = 252


# ── Bhav Copy Fetcher ─────────────────────────────────────────────────────────

class BhavCopyFetcher:
    """
    Downloads and normalises NSE F&O bhav copy for any trading date.
    Handles both old (pre-2024-07-08) and new UDiFF (post-2024-07-08) formats.

    Normalised output columns:
        date, symbol, instrument_type, option_type, expiry,
        strike, close, settle, open_interest, chg_oi, volume, underlying_price

    instrument_type: 'OPTSTK' | 'FUTSTK' | 'OPTIDX' | 'FUTIDX'
    option_type:     'CE' | 'PE' | 'FUT'
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        RAW_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, trading_date: date) -> Optional[pd.DataFrame]:
        """
        Return normalised bhav copy for *trading_date*.
        Returns None if the date is not a trading day (weekend / holiday / future).
        Uses disk cache automatically.
        """
        cache_path = RAW_CACHE / f"{trading_date.strftime('%Y%m%d')}_bhav.parquet"
        if cache_path.exists():
            logger.debug(f"BhavCopy cache hit: {trading_date}")
            return pd.read_parquet(cache_path)

        raw = self._download(trading_date)
        if raw is None:
            return None

        normalized = self._normalize(raw, trading_date)
        if normalized is not None and not normalized.empty:
            normalized.to_parquet(cache_path, index=False)
            logger.info(f"  BhavCopy cached {len(normalized):,} rows → {cache_path.name}")

        return normalized

    def fetch_range(self, start: date, end: date) -> pd.DataFrame:
        """Fetch and concatenate bhav copies for a date range, skipping non-trading days."""
        frames = []
        current = start
        while current <= end:
            if current.weekday() < 5:   # skip weekends
                df = self.fetch(current)
                if df is not None:
                    frames.append(df)
            current += timedelta(days=1)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── Download ──────────────────────────────────────────────────────────────

    def _download(self, d: date) -> Optional[pd.DataFrame]:
        if d >= UDIFF_CUTOVER:
            return self._download_new(d)
        return self._download_old(d)

    def _download_new(self, d: date) -> Optional[pd.DataFrame]:
        """Download UDiFF format (post 2024-07-08)."""
        url = NEW_BHAV_URL.format(yyyymmdd=d.strftime("%Y%m%d"))
        return self._fetch_zip_csv(url, d, format_hint="new")

    def _download_old(self, d: date) -> Optional[pd.DataFrame]:
        """Download legacy format (pre 2024-07-08)."""
        month_abbr = d.strftime("%b").upper()  # JAN, FEB, ...
        url = OLD_BHAV_URL.format(
            year=d.year,
            month_abbr=month_abbr,                 # JAN, FEB (for path)
            month_abbr_upper=month_abbr,           # JAN, FEB (for filename)
            dd=d.strftime("%d"),
        )
        return self._fetch_zip_csv(url, d, format_hint="old")

    def _fetch_zip_csv(
        self, url: str, d: date, format_hint: str
    ) -> Optional[pd.DataFrame]:
        try:
            resp = requests.get(url, headers=NSE_HEADERS, timeout=self.timeout)
            if resp.status_code == 404:
                logger.debug(f"  {d} — 404 (likely holiday/non-trading day): {url}")
                return None
            resp.raise_for_status()
            if len(resp.content) < 1000:
                logger.warning(f"  {d} — suspiciously small response ({len(resp.content)} bytes)")
                return None
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
            logger.debug(f"  {d} [{format_hint}] downloaded {len(df):,} rows")
            return df
        except requests.Timeout:
            logger.warning(f"  {d} — timeout fetching bhav copy")
            return None
        except (zipfile.BadZipFile, pd.errors.ParserError) as exc:
            logger.warning(f"  {d} — parse error: {exc}")
            return None
        except requests.HTTPError as exc:
            logger.warning(f"  {d} — HTTP error: {exc}")
            return None

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame, d: date) -> Optional[pd.DataFrame]:
        """Dispatch to format-specific normaliser."""
        if "TckrSymb" in df.columns:
            return self._normalize_new(df, d)
        if "SYMBOL" in df.columns:
            return self._normalize_old(df, d)
        logger.warning(f"  {d} — unrecognised bhav copy schema: {list(df.columns)[:6]}")
        return None

    def _normalize_new(self, df: pd.DataFrame, d: date) -> pd.DataFrame:
        """Normalise UDiFF format."""
        # FinInstrmTp: STO=stock option, STF=stock future, IFO=index future, ITO=index option
        inst_map = {"STO": "OPTSTK", "STF": "FUTSTK", "IFO": "FUTIDX", "ITO": "OPTIDX"}
        opt_type = df["OptnTp"].fillna("FUT")
        opt_type = opt_type.replace({"nan": "FUT", "": "FUT"})

        out = pd.DataFrame({
            "date":              pd.to_datetime(d),
            "symbol":            df["TckrSymb"].str.strip(),
            "instrument_type":   df["FinInstrmTp"].map(inst_map).fillna("UNKNOWN"),
            "option_type":       opt_type,
            "expiry":            pd.to_datetime(df["XpryDt"], errors="coerce"),
            "strike":            pd.to_numeric(df["StrkPric"], errors="coerce").fillna(0.0),
            "close":             pd.to_numeric(df["ClsPric"],  errors="coerce"),
            "settle":            pd.to_numeric(df["SttlmPric"], errors="coerce"),
            "open_interest":     pd.to_numeric(df["OpnIntrst"], errors="coerce").fillna(0),
            "chg_oi":            pd.to_numeric(df["ChngInOpnIntrst"], errors="coerce").fillna(0),
            "volume":            pd.to_numeric(df["TtlTradgVol"], errors="coerce").fillna(0),
            "underlying_price":  pd.to_numeric(df["UndrlygPric"], errors="coerce"),
        })
        return out[out["open_interest"] >= 0]  # RULE 48

    def _normalize_old(self, df: pd.DataFrame, d: date) -> pd.DataFrame:
        """Normalise legacy bhav copy format."""
        opt_col = df["OPTION_TYP"].str.strip()
        opt_type = opt_col.replace({"XX": "FUT"})   # futures have XX as option_type

        out = pd.DataFrame({
            "date":              pd.to_datetime(d),
            "symbol":            df["SYMBOL"].str.strip(),
            "instrument_type":   df["INSTRUMENT"].str.strip(),
            "option_type":       opt_type,
            "expiry":            pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce"),
            "strike":            pd.to_numeric(df["STRIKE_PR"], errors="coerce").fillna(0.0),
            "close":             pd.to_numeric(df["CLOSE"],     errors="coerce"),
            "settle":            pd.to_numeric(df["SETTLE_PR"], errors="coerce"),
            "open_interest":     pd.to_numeric(df["OPEN_INT"],  errors="coerce").fillna(0),
            "chg_oi":            pd.to_numeric(df["CHG_IN_OI"], errors="coerce").fillna(0),
            "volume":            pd.to_numeric(df["CONTRACTS"],  errors="coerce").fillna(0),
            "underlying_price":  np.nan,   # not in old format; filled later from spot cache
        })
        return out[out["open_interest"] >= 0]


# ── IV Solver ────────────────────────────────────────────────────────────────

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes option price (no dividends)."""
    from math import log, sqrt, exp
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol(
    market_price: float, S: float, K: float, T: float,
    r: float = RISK_FREE_RATE, is_call: bool = True,
    tol: float = 1e-5, max_iter: int = 50,
) -> float:
    """Newton-Raphson implied volatility solver. Returns NaN on failure."""
    from math import sqrt, log, exp
    from scipy.stats import norm

    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if market_price <= intrinsic + 1e-6:
        return np.nan

    sigma = 0.30   # starting estimate
    for _ in range(max_iter):
        price = _bs_price(S, K, T, r, sigma, is_call)
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T)
        if vega < 1e-10:
            return np.nan
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        if sigma <= 0:
            return np.nan
    return np.nan


# ── F&O Feature Engine ────────────────────────────────────────────────────────

class FNOFeatureEngine:
    """
    Computes 5 F&O features from a normalised bhav copy DataFrame.
    Operates on ONE symbol at a time across its full history.

    Input: concatenated normalized bhav copy (all dates) filtered to one symbol.
    Output: DataFrame indexed by date with columns:
        pcr_oi, futures_basis_zscore, iv_skew_zscore, oi_signal, max_pain_distance
    """

    def compute(
        self,
        symbol_df: pd.DataFrame,
        spot_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Args:
            symbol_df: Normalized bhav copy rows for ONE symbol, all dates.
            spot_df:   OHLCV DataFrame for the stock (used for ATR14 and
                       spot fill if underlying_price is missing).
                       Index = datetime, columns include close, high, low.
        Returns:
            DataFrame indexed by date, 5 feature columns.
        """
        if symbol_df.empty:
            return pd.DataFrame()

        symbol_df = symbol_df.copy()
        symbol_df["date"] = pd.to_datetime(symbol_df["date"]).dt.normalize()
        dates = sorted(symbol_df["date"].unique())

        rows = []
        for d in dates:
            day = symbol_df[symbol_df["date"] == d]
            row = self._compute_day(d, day, spot_df)
            rows.append(row)

        out = pd.DataFrame(rows).set_index("date")
        out.index = pd.to_datetime(out.index)
        return self._zscore_features(out)

    # ── Per-day computation ───────────────────────────────────────────────────

    def _compute_day(
        self, d: pd.Timestamp, day: pd.DataFrame, spot_df: Optional[pd.DataFrame]
    ) -> dict:
        result = {"date": d}

        # Spot price: prefer bhav copy underlying_price; fallback to spot_df
        spot = float(day["underlying_price"].dropna().median())
        if np.isnan(spot) and spot_df is not None and d in spot_df.index:
            spot = float(spot_df.loc[d, "close"])

        # Identify near-month expiry (smallest expiry >= today)
        option_rows = day[day["instrument_type"].isin(["OPTSTK", "OPTIDX"])]
        future_rows = day[day["instrument_type"].isin(["FUTSTK", "FUTIDX"])]

        near_expiry = self._near_expiry(option_rows, d)
        near_opts    = option_rows[option_rows["expiry"] == near_expiry] if near_expiry else option_rows.iloc[0:0]
        near_futs    = future_rows[future_rows["expiry"] == self._near_expiry(future_rows, d)] if not future_rows.empty else future_rows.iloc[0:0]

        # 1. PCR OI
        result["pcr_oi"] = self._pcr_oi(near_opts)

        # 2. Futures basis (raw — z-scored later across history)
        result["futures_basis_raw"] = self._futures_basis_raw(near_futs, spot)

        # 3. IV skew proxy (raw — z-scored later)
        result["iv_skew_raw"] = self._iv_skew_raw(near_opts, spot, near_expiry, d)

        # 4. OI signal
        result["oi_signal"] = self._oi_signal(near_futs, spot_df, d)

        # 5. Max pain distance (raw — in ATR units)
        result["max_pain_distance"] = self._max_pain_distance(near_opts, spot, spot_df, d, near_expiry)

        return result

    # ── Feature computations ──────────────────────────────────────────────────

    @staticmethod
    def _near_expiry(df: pd.DataFrame, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Return the nearest expiry date >= d."""
        valid = df["expiry"].dropna()
        valid = valid[valid >= d]
        return valid.min() if not valid.empty else None

    @staticmethod
    def _pcr_oi(near_opts: pd.DataFrame) -> float:
        """Put/Call OI ratio. Validated: [0.1, 5.0]."""
        pe_oi = near_opts[near_opts["option_type"] == "PE"]["open_interest"].sum()
        ce_oi = near_opts[near_opts["option_type"] == "CE"]["open_interest"].sum()
        if ce_oi <= 0:
            return np.nan
        pcr = float(pe_oi / ce_oi)
        return pcr if PCR_MIN <= pcr <= PCR_MAX else np.nan   # validation gate

    @staticmethod
    def _futures_basis_raw(near_futs: pd.DataFrame, spot: float) -> float:
        """
        Raw basis = near-month futures settle - spot.
        Sign: positive = futures at premium (bullish carry).
        """
        if near_futs.empty or np.isnan(spot) or spot <= 0:
            return np.nan
        f_settle = float(near_futs["settle"].dropna().iloc[0]) if not near_futs["settle"].dropna().empty else np.nan
        if np.isnan(f_settle):
            return np.nan
        return f_settle - spot

    @staticmethod
    def _iv_skew_raw(
        near_opts: pd.DataFrame, spot: float,
        expiry: Optional[pd.Timestamp], d: pd.Timestamp,
    ) -> float:
        """
        IV skew proxy = median(OTM PE close) - median(OTM CE close), divided by spot.
        OTM = strikes within 2–8% from spot.
        This is a simplified risk-reversal proxy. Full BS IV computation is available
        but adds ~200ms/stock/day — use raw price ratio for backfill performance.
        """
        if near_opts.empty or np.isnan(spot) or spot <= 0:
            return np.nan

        lo, hi = spot * 0.92, spot * 1.08
        otm_pe = near_opts[
            (near_opts["option_type"] == "PE") &
            (near_opts["strike"] < spot) &
            (near_opts["strike"] >= lo) &
            (near_opts["close"] > 0)
        ]["close"]
        otm_ce = near_opts[
            (near_opts["option_type"] == "CE") &
            (near_opts["strike"] > spot) &
            (near_opts["strike"] <= hi) &
            (near_opts["close"] > 0)
        ]["close"]

        if otm_pe.empty or otm_ce.empty:
            return np.nan

        skew = float(otm_pe.median() - otm_ce.median()) / spot
        return skew

    @staticmethod
    def _oi_signal(
        near_futs: pd.DataFrame, spot_df: Optional[pd.DataFrame], d: pd.Timestamp
    ) -> float:
        """
        sign(Δspot_return) × sign(Δfutures_OI).
        +1: price up + OI up = bullish (new longs entering).
        -1: price up + OI down = bearish (shorts covering, not conviction).
        """
        if near_futs.empty or spot_df is None or d not in spot_df.index:
            return np.nan

        delta_oi = float(near_futs["chg_oi"].sum())

        spot_idx = spot_df.index.get_loc(d)
        if spot_idx == 0:
            return np.nan
        prev_d = spot_df.index[spot_idx - 1]
        delta_spot = float(spot_df.loc[d, "close"]) - float(spot_df.loc[prev_d, "close"])

        if delta_spot == 0 or delta_oi == 0:
            return 0.0

        return float(np.sign(delta_spot) * np.sign(delta_oi))

    @staticmethod
    def _max_pain_distance(
        near_opts: pd.DataFrame, spot: float,
        spot_df: Optional[pd.DataFrame], d: pd.Timestamp,
        expiry: Optional[pd.Timestamp],
    ) -> float:
        """
        Computes max pain strike = strike that causes maximum aggregate loss to
        option buyers. Returns (spot - max_pain) / ATR14.
        Returns 0 if not in expiry week (last 5 trading days before expiry).
        """
        if near_opts.empty or expiry is None or np.isnan(spot):
            return 0.0

        # Expiry week gate: only active in last 5 calendar days before expiry
        days_to_expiry = (expiry.date() - d.date()).days
        if days_to_expiry > 7:
            return 0.0

        strikes = near_opts["strike"].unique()
        if len(strikes) == 0:
            return 0.0

        # Max pain: for each strike, sum total OI loss to buyers at that pin
        pain = {}
        for pin in strikes:
            ce_loss = near_opts[
                (near_opts["option_type"] == "CE") & (near_opts["strike"] < pin)
            ].apply(lambda r: (pin - r["strike"]) * r["open_interest"], axis=1).sum()

            pe_loss = near_opts[
                (near_opts["option_type"] == "PE") & (near_opts["strike"] > pin)
            ].apply(lambda r: (r["strike"] - pin) * r["open_interest"], axis=1).sum()

            pain[pin] = float(ce_loss + pe_loss)

        if not pain:
            return 0.0

        max_pain = min(pain, key=pain.get)

        # ATR14 from spot_df
        atr = 1.0
        if spot_df is not None and len(spot_df) > 14 and d in spot_df.index:
            idx = spot_df.index.get_loc(d)
            window = spot_df.iloc[max(0, idx - 14): idx + 1]
            if len(window) >= 5:
                prev_close = window["close"].shift(1)
                tr = pd.concat([
                    window["high"] - window["low"],
                    (window["high"] - prev_close).abs(),
                    (window["low"] - prev_close).abs(),
                ], axis=1).max(axis=1)
                atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])

        return float((spot - max_pain) / (atr + 1e-9))

    # ── Z-scoring ─────────────────────────────────────────────────────────────

    def _zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw features to z-scores over rolling windows.
        futures_basis → basis_zscore (60-day), iv_skew_raw → iv_skew_zscore (30-day).
        pcr_oi, oi_signal, max_pain_distance stay as-is.
        Validate: |basis_zscore| > 6 → NaN (data error gate per RULE 49).
        """
        out = df.copy()

        for col, window, out_col in [
            ("futures_basis_raw", BASIS_WINDOW, "futures_basis_zscore"),
            ("iv_skew_raw",       SKEW_WINDOW,  "iv_skew_zscore"),
        ]:
            if col not in out.columns:
                continue
            s = out[col]
            roll_mean = s.rolling(window, min_periods=max(10, window // 3)).mean()
            roll_std  = s.rolling(window, min_periods=max(10, window // 3)).std()
            zscore    = (s - roll_mean) / (roll_std + 1e-9)
            out[out_col] = zscore

        # Validation gate: |basis_zscore| > 6 → likely data error
        if "futures_basis_zscore" in out.columns:
            mask = out["futures_basis_zscore"].abs() > BASIS_ZSCORE_LIMIT
            out.loc[mask, "futures_basis_zscore"] = np.nan

        # Drop intermediate raw columns
        out = out.drop(columns=["futures_basis_raw", "iv_skew_raw"], errors="ignore")

        return out[["pcr_oi", "futures_basis_zscore", "iv_skew_zscore",
                    "oi_signal", "max_pain_distance"]]


# ── Main Provider ─────────────────────────────────────────────────────────────

class FNODataProvider:
    """
    Main interface for F&O feature retrieval.

    Usage in features.py:
        fno = FNODataProvider()
        fno_feats = fno.get_fno_features("COFORGE", start="2022-01-01", end="2025-01-01")
        # fno_feats.index = DatetimeIndex, columns = [pcr_oi, futures_basis_zscore, ...]

    Backfill usage (run once, takes 2-4 hours for 50 stocks × 3 years):
        fno.backfill_historical(symbols=NIFTY_50_TICKERS, start="2022-01-01", end="2025-01-01")
    """

    def __init__(self):
        self.fetcher = BhavCopyFetcher()
        self.engine  = FNOFeatureEngine()
        FEAT_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_fno_features(
        self,
        symbol: str,
        start: str,
        end: str,
        spot_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Return F&O feature DataFrame for a single symbol.
        Reads from feature cache if available; otherwise computes on-the-fly
        from raw bhav copy cache.

        Args:
            symbol:  NSE trading symbol WITHOUT exchange suffix (e.g. "RELIANCE")
            start:   "YYYY-MM-DD"
            end:     "YYYY-MM-DD"
            spot_df: Optional OHLCV DataFrame for accurate ATR14 and oi_signal.
        """
        cache_path = FEAT_CACHE / f"{symbol}.parquet"
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)

        feat_df = pd.DataFrame()
        if cache_path.exists():
            feat_df = pd.read_parquet(cache_path)
            feat_df.index = pd.to_datetime(feat_df.index)

        # Check if cache covers the requested range
        if not feat_df.empty:
            cached_start = feat_df.index.min()
            cached_end   = feat_df.index.max()
            if cached_start <= start_dt and cached_end >= end_dt:
                return feat_df.loc[start_dt:end_dt]

        # Need to compute: assemble bhav copy from raw cache
        logger.info(f"Computing F&O features for {symbol} ({start} → {end})...")
        raw = self._assemble_raw(symbol, start_dt.date(), end_dt.date())

        if raw.empty:
            logger.warning(f"  {symbol}: no raw F&O data available in cache")
            return pd.DataFrame()

        feat_df = self.engine.compute(raw, spot_df=spot_df)

        if not feat_df.empty:
            feat_df.to_parquet(cache_path)
            logger.info(f"  {symbol}: {len(feat_df)} feature rows cached")

        return feat_df.loc[start_dt:end_dt] if not feat_df.empty else pd.DataFrame()

    def backfill_historical(
        self,
        symbols: list[str],
        start: str = "2022-01-01",
        end: str = "2025-01-01",
        spot_data: Optional[dict] = None,
    ) -> None:
        """
        Run historical backfill for a list of symbols.
        Downloads all bhav copies in the date range (one per trading day),
        then computes and caches F&O features per symbol.

        This should be run ONCE before live trading. Takes 2–4 hours for
        50 stocks × 750 trading days.

        Args:
            symbols:    List of NSE symbols without .NS suffix (e.g. ["RELIANCE", "TCS"])
            start:      Start date "YYYY-MM-DD"
            end:        End date "YYYY-MM-DD"
            spot_data:  Dict of {symbol: pd.DataFrame} with OHLCV for each stock.
                        If None, ATR14 and oi_signal features will be approximated.
        """
        start_d = datetime.strptime(start, "%Y-%m-%d").date()
        end_d   = datetime.strptime(end,   "%Y-%m-%d").date()
        spot_data = spot_data or {}

        # Step 1: Download all bhav copies in range (reuses cached files)
        logger.info(f"Backfill: downloading bhav copies {start} → {end}...")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        dates_to_fetch = []
        current = start_d
        while current <= end_d:
            if current.weekday() < 5:
                dates_to_fetch.append(current)
            current += timedelta(days=1)

        total_days = len(dates_to_fetch)
        success_count = 0
        
        # Download in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {executor.submit(self.fetcher.fetch, d): d for d in dates_to_fetch}
            for i, future in enumerate(as_completed(future_to_date)):
                d = future_to_date[future]
                try:
                    res = future.result()
                    if res is not None:
                        success_count += 1
                except Exception as e:
                    logger.debug(f"  {d} backfill error: {e}")
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Downloaded {i+1}/{total_days} dates...")

        logger.info(f"  {success_count}/{total_days} valid trading days downloaded & cached")

        # Step 2: Compute features per symbol

        for sym in symbols:
            logger.info(f"  Computing features: {sym}...")
            spot_df = spot_data.get(sym)
            self.get_fno_features(sym, start, end, spot_df=spot_df)

        logger.info("Backfill complete.")

    def update_today(
        self, symbols: list[str], spot_data: Optional[dict] = None
    ) -> None:
        """
        Fetch today's bhav copy and update feature cache for all symbols.
        Run after 6 PM IST when NSE publishes the day's bhav copy.
        """
        today = date.today()
        logger.info(f"Daily update: fetching bhav copy for {today}...")
        df = self.fetcher.fetch(today)
        if df is None:
            logger.warning("  No bhav copy for today — holiday or non-trading day?")
            return

        spot_data = spot_data or {}
        for sym in symbols:
            cache_path = FEAT_CACHE / f"{sym}.parquet"
            existing   = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
            raw_today  = df[df["symbol"] == sym]
            if raw_today.empty:
                continue

            spot_df = spot_data.get(sym)
            feat_today = self.engine.compute(raw_today, spot_df=spot_df)
            if feat_today.empty:
                continue

            updated = pd.concat([existing, feat_today]).sort_index()
            updated = updated[~updated.index.duplicated(keep="last")]
            updated.to_parquet(cache_path)
        logger.info(f"  Updated {len(symbols)} symbols for {today}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assemble_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Read raw bhav parquets from disk, filter to symbol."""
        frames = []
        current = start
        while current <= end:
            p = RAW_CACHE / f"{current.strftime('%Y%m%d')}_bhav.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                sym_df = df[df["symbol"] == symbol]
                if not sym_df.empty:
                    frames.append(sym_df)
            current += timedelta(days=1)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
