"""
MARK5 — Foundation Model Signal Component
==========================================
Wraps Kronos (AAAI 2026, arXiv:2508.02739) and Amazon Chronos (ICML 2024)
to generate forward-looking ML signal scores compatible with MomentumSignalEngine.

Both components produce a score in [0, 1] representing the foundation model's
directional conviction for the next `horizon` bars:
  > 0.55 → model expects upward price move
  < 0.45 → model expects downward move
  ~0.50  → neutral / uncertain

SAFETY INVARIANTS (never violate these):
  1. Fail-open:   all exceptions return NEUTRAL (0.5) — never crash the portfolio
  2. No lookahead: strictly uses df[df.index <= date] — no future data leaks
  3. Lazy-load:    model weights downloaded on first use, not at import time
  4. Additive:     existing MomentumSignalEngine output unchanged when disabled
  5. Disk-cached:  predictions cached to avoid recomputation across backtest runs

INTEGRATION (in momentum_portfolio.py):
  from core.models.foundation_signal import build_foundation_signal
  fs = build_foundation_signal(model="auto")
  # Precompute for all rebalance dates (cached):
  fd_scores = fs.precompute_rebalance_scores(ticker, df, rebal_dates)
  # At each rebalance, blend with momentum score (10% weight):
  blended_rank = 0.90 * momentum_score + 0.10 * fd_scores.get(date, 0.5)

INSTALL:
  Kronos:  pip install git+https://github.com/shiyu-coder/Kronos.git
  Chronos: pip install chronos-forecasting

PAPER MODE ONLY — never executes real orders.
"""
from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MARK5.FoundationSignal")

_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR = _ROOT / "data" / "cache" / "foundation_scores"

NEUTRAL = 0.5                  # fallback score on any failure
RETURN_SCALE = 0.15            # sigmoid scale: 15% return → score ~0.73
MIN_CONTEXT_BARS = 60          # minimum bars to form a valid prediction
MAX_CHRONOS_CONTEXT = 512      # Chronos hard limit (token budget)
MAX_KRONOS_MINI_CONTEXT = 2048 # Kronos-mini context window


# ── Sigmoid helper ────────────────────────────────────────────────────────────

def _sig(x: float, scale: float) -> float:
    """Map ℝ → (0, 1) via sigmoid with the given scale."""
    try:
        return 1.0 / (1.0 + math.exp(-x / scale))
    except (OverflowError, ZeroDivisionError):
        return 0.0 if x < 0 else 1.0


def _return_to_score(predicted_return: float, scale: float = RETURN_SCALE) -> float:
    """Convert a predicted forward return to a [0, 1] directional score."""
    return float(np.clip(_sig(predicted_return, scale), 0.0, 1.0))


# ── CUDA helper ───────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Abstract base ─────────────────────────────────────────────────────────────

class FoundationSignalBase(abc.ABC):
    """
    Abstract base for all foundation-model signal components.

    Subclasses must implement:
      _load_model()             → bool (True if successful)
      _predict_forward_return() → float (predicted return, e.g. 0.05 = +5%)
    """

    def __init__(self, model_size: str = "mini", device: Optional[str] = None):
        self.model_size = model_size
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self._model = None
        self._available: Optional[bool] = None  # None = not yet attempted

    @abc.abstractmethod
    def _load_model(self) -> bool:
        """Load model. Returns True on success, False on ImportError / failure."""

    @abc.abstractmethod
    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        """
        Predict `horizon`-bar forward return using `df` as context.
        `df` is already sliced to available history — no further slicing needed.
        Returns e.g. 0.05 for +5% expected return.
        """

    # ── Public API ────────────────────────────────────────────────────────────

    def score_at(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        horizon: int = 21,
    ) -> float:
        """
        Score in [0, 1] for the given ticker at `date`.
        Uses ONLY data up to and including `date` (zero lookahead).
        Returns NEUTRAL (0.5) on any error or if model unavailable.
        """
        try:
            df_slice = df.loc[df.index <= date]
            if len(df_slice) < MIN_CONTEXT_BARS:
                return NEUTRAL

            if self._available is None:
                self._available = self._load_model()

            if not self._available:
                return NEUTRAL

            fwd = self._predict_forward_return(df_slice, horizon)
            return _return_to_score(fwd)

        except Exception as exc:
            logger.debug(f"{self.__class__.__name__}.score_at({date}): {exc}")
            return NEUTRAL

    def precompute_rebalance_scores(
        self,
        ticker: str,
        df: pd.DataFrame,
        dates: pd.DatetimeIndex,
        horizon: int = 21,
    ) -> Dict[pd.Timestamp, float]:
        """
        Compute scores for all `dates` (typically the ~80 rebalance dates in a
        4-year OOS run). Disk-caches results so subsequent runs are instant.

        Args:
            ticker:  NSE ticker symbol (used as cache key component)
            df:      Full OHLCV DataFrame for this ticker (all history)
            dates:   DatetimeIndex of dates to evaluate at
            horizon: Forward prediction horizon in bars (default 21 = ~1 month)

        Returns:
            Dict mapping each date → score in [0.0, 1.0]
        """
        results: Dict[pd.Timestamp, float] = {}
        cache_key = self._cache_key(ticker, horizon)
        cached = self._load_cache(cache_key)
        new_entries = 0

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            if date_str in cached:
                results[date] = float(cached[date_str])
            else:
                score = self.score_at(df, date, horizon)
                results[date] = score
                cached[date_str] = score
                new_entries += 1

        if new_entries > 0:
            self._save_cache(cache_key, cached)
            logger.debug(f"{ticker}: cached {new_entries} new foundation scores")

        return results

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._load_model()
        return bool(self._available)

    def clear_cache(self, ticker: Optional[str] = None, horizon: int = 21) -> None:
        """Remove cached scores. Pass ticker=None to wipe all cache for this model."""
        if ticker is not None:
            p = self._cache_path(self._cache_key(ticker, horizon))
            if p.exists():
                p.unlink()
        else:
            for p in _CACHE_DIR.glob("*.json"):
                p.unlink()

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, ticker: str, horizon: int) -> str:
        model_id = f"{self.__class__.__name__}_{self.model_size}_h{horizon}"
        h = hashlib.md5(model_id.encode()).hexdigest()[:8]
        return f"{h}_{ticker}"

    def _cache_path(self, cache_key: str) -> Path:
        return _CACHE_DIR / f"{cache_key}.json"

    def _load_cache(self, cache_key: str) -> Dict[str, float]:
        p = self._cache_path(cache_key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache_key: str, data: Dict[str, float]) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._cache_path(cache_key).write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug(f"Cache save failed ({cache_key}): {exc}")


# ── Kronos Component ──────────────────────────────────────────────────────────

class KronosSignalComponent(FoundationSignalBase):
    """
    Foundation model signal using Kronos (AAAI 2026, arXiv:2508.02739).

    Kronos was pretrained on 45+ global exchanges on OHLCV (K-line) sequences.
    It uses hierarchical discrete tokenization + decoder-only Transformer.
    This makes it the most appropriate model for MARK5's OHLCV data.

    Why Kronos over generic time-series models:
      - Trained specifically on financial K-line data (not electricity/weather)
      - OHLCV-aware tokenization handles multi-dimensional candle structure
      - Kronos-mini has 2048-bar context (8+ years of daily data)
      - Cross-market pretraining transfers knowledge to NSE (solves data scarcity)

    The model/ directory must be present at core/models/kronos_model/.
    It is bundled with MARK5 (copied from the Kronos GitHub repo).

    Model sizes and HuggingFace IDs:
      mini  (4.1M,  2048-bar context): NeoQuasar/Kronos-mini + NeoQuasar/Kronos-Tokenizer-2k
      small (24.7M, 512-bar context):  NeoQuasar/Kronos-small + NeoQuasar/Kronos-Tokenizer-base
      base  (102.3M, 512-bar context): NeoQuasar/Kronos-base  + NeoQuasar/Kronos-Tokenizer-base
    """

    _MODEL_IDS: Dict[str, str] = {
        "mini":  "NeoQuasar/Kronos-mini",
        "small": "NeoQuasar/Kronos-small",
        "base":  "NeoQuasar/Kronos-base",
    }
    _TOKENIZER_IDS: Dict[str, str] = {
        "mini":  "NeoQuasar/Kronos-Tokenizer-2k",
        "small": "NeoQuasar/Kronos-Tokenizer-base",
        "base":  "NeoQuasar/Kronos-Tokenizer-base",
    }
    _CONTEXT: Dict[str, int] = {
        "mini": MAX_KRONOS_MINI_CONTEXT,
        "small": 512,
        "base": 512,
    }

    def _load_model(self) -> bool:
        try:
            from core.models.kronos_model import KronosTokenizer, Kronos, KronosPredictor
            model_id = self._MODEL_IDS.get(self.model_size, self._MODEL_IDS["mini"])
            tok_id   = self._TOKENIZER_IDS.get(self.model_size, self._TOKENIZER_IDS["mini"])
            ctx      = self._CONTEXT.get(self.model_size, 512)
            tokenizer = KronosTokenizer.from_pretrained(tok_id)
            model     = Kronos.from_pretrained(model_id)
            self._model = KronosPredictor(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                max_context=ctx,
            )
            self._KronosPredictor = KronosPredictor
            logger.info(f"Kronos-{self.model_size} loaded ({model_id})")
            return True
        except ImportError as exc:
            logger.warning(f"KronosSignalComponent: kronos_model import failed — {exc}")
            return False
        except Exception as exc:
            logger.warning(f"KronosSignalComponent._load_model failed: {exc}")
            return False

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        required = ["open", "high", "low", "close"]
        if any(c not in df.columns for c in required):
            return 0.0

        ctx_len = self._CONTEXT.get(self.model_size, 512)
        df_ctx  = df.tail(ctx_len).copy()

        # Kronos needs volume and amount columns
        if "volume" not in df_ctx.columns:
            df_ctx["volume"] = 0.0
        if "amount" not in df_ctx.columns:
            df_ctx["amount"] = df_ctx["volume"] * df_ctx["close"]

        x_timestamp = df_ctx.index
        # Build future business-day timestamps for horizon bars
        last_date   = pd.Timestamp(df_ctx.index[-1])
        y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

        pred_df = self._model.predict(
            df=df_ctx,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=0.8,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        current_close = float(df["close"].iloc[-1])
        if current_close <= 0 or pred_df is None or len(pred_df) == 0:
            return 0.0

        pred_close = float(pred_df["close"].iloc[-1])
        return (pred_close / current_close) - 1.0


# ── Chronos Component ─────────────────────────────────────────────────────────

class ChronosSignalComponent(FoundationSignalBase):
    """
    Foundation model signal using Amazon Chronos (ICML 2024).

    Chronos is a language-model-based univariate time series forecaster.
    Pretrained on a diverse corpus including financial time series.
    Works directly on close prices — no OHLCV structure needed.

    Why Chronos is a good fallback for MARK5:
      - pip install chronos-forecasting (much simpler than Kronos)
      - Chronos-Bolt is 250x faster than original Chronos
      - Provides calibrated uncertainty (quantiles) — not just point estimates
      - State-of-the-art on GIFT-Eval financial benchmark

    Install:
      pip install chronos-forecasting

    Recommended sizes:
      mini  → alias for bolt-small (48M): fast, good accuracy
      small → bolt-small (48M): recommended default
      base  → bolt-base (205M): higher accuracy, slower
      v2    → chronos-2 (120M): latest generation, best zero-shot
    """

    _HF_IDS: Dict[str, str] = {
        "tiny":     "amazon/chronos-bolt-tiny",
        "mini":     "amazon/chronos-bolt-small",
        "small":    "amazon/chronos-bolt-small",
        "base":     "amazon/chronos-bolt-base",
        "large":    "amazon/chronos-t5-large",
        "v2":       "amazon/chronos-2",
        "v2-small": "amazon/chronos-2-small",
    }

    def _load_model(self) -> bool:
        try:
            import torch
            model_id = self._HF_IDS.get(self.model_size, self._HF_IDS["small"])
            dtype = torch.float32  # bfloat16 only for GPU

            # Pipeline selection priority:
            # 1. ChronosBoltPipeline — for "bolt" models (fastest, quantile output)
            # 2. Chronos2Pipeline    — for "chronos-2" models
            # 3. ChronosPipeline     — for "chronos-t5" models (sample output)
            loaded = False

            if "bolt" in model_id:
                try:
                    from chronos import ChronosBoltPipeline
                    self._model = ChronosBoltPipeline.from_pretrained(
                        model_id, device_map=self.device, dtype=dtype
                    )
                    self._is_bolt = True
                    loaded = True
                    logger.info(f"ChronosBoltPipeline loaded: {model_id}")
                except Exception as exc:
                    logger.debug(f"ChronosBoltPipeline failed: {exc}")

            if not loaded and "chronos-2" in model_id:
                try:
                    from chronos import Chronos2Pipeline
                    self._model = Chronos2Pipeline.from_pretrained(
                        model_id, device_map=self.device, dtype=dtype
                    )
                    self._is_bolt = False
                    loaded = True
                    logger.info(f"Chronos2Pipeline loaded: {model_id}")
                except Exception as exc:
                    logger.debug(f"Chronos2Pipeline failed: {exc}")

            if not loaded:
                from chronos import ChronosPipeline
                self._model = ChronosPipeline.from_pretrained(
                    model_id, device_map=self.device, dtype=dtype
                )
                self._is_bolt = False
                logger.info(f"ChronosPipeline loaded: {model_id}")

            self._torch = torch
            return True

        except ImportError:
            logger.warning(
                "chronos-forecasting not installed. "
                "Install with: pip install chronos-forecasting"
            )
            return False
        except Exception as exc:
            logger.warning(f"ChronosSignalComponent._load_model failed: {exc}")
            return False

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        import torch

        close = df["close"].astype(float).values
        ctx   = close[-MAX_CHRONOS_CONTEXT:]
        context = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        current_close = float(close[-1])
        if current_close <= 0:
            return 0.0

        is_bolt = getattr(self, "_is_bolt", False)

        if is_bolt:
            # Bolt output: [num_series=1, num_quantiles=9, horizon]
            # Quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # Median = index 4 (0.5 quantile)
            forecast = self._model.predict(context, prediction_length=horizon)
            # forecast.shape: [1, 9, horizon]
            num_q = forecast.shape[1]
            median_idx = num_q // 2  # middle quantile ≈ 0.5
            pred_close = float(forecast[0, median_idx, -1].item())
        else:
            # Sample-based: [num_samples, num_series=1, horizon]
            forecast = self._model.predict(
                context, prediction_length=horizon, num_samples=20
            )
            # forecast.shape: [20, 1, horizon]
            pred_median = forecast.median(dim=0).values  # [1, horizon]
            pred_close  = float(pred_median[0, -1].item())

        return (pred_close / current_close) - 1.0


# ── Auto-selector ─────────────────────────────────────────────────────────────

class FoundationSignalAuto(FoundationSignalBase):
    """
    Tries Kronos first, then Chronos. Uses whichever is available.

    If neither is installed, all calls return NEUTRAL (0.5) silently.
    The portfolio backtest continues unchanged.

    Priority:
      1. Kronos-mini   — financial-specific OHLCV model, 2048-bar context
      2. Chronos-bolt-small — general time-series, easy install, fast

    Override preference with prefer="chronos" to try Chronos first.
    """

    def __init__(self, prefer: str = "kronos", device: Optional[str] = None):
        super().__init__(device=device)
        self._prefer = prefer
        self._delegate: Optional[FoundationSignalBase] = None

    def _load_model(self) -> bool:
        candidates: List[type] = (
            [KronosSignalComponent, ChronosSignalComponent]
            if self._prefer == "kronos"
            else [ChronosSignalComponent, KronosSignalComponent]
        )
        for cls in candidates:
            inst = cls(model_size="mini", device=self.device)
            if inst._load_model():
                inst._available = True
                self._delegate = inst
                self._model = inst._model
                logger.info(f"FoundationSignalAuto → {cls.__name__} (size=mini)")
                return True

        logger.warning(
            "No foundation model available. "
            "Install with: pip install chronos-forecasting  (recommended)\n"
            "              OR pip install git+https://github.com/shiyu-coder/Kronos.git"
        )
        return False

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        if self._delegate is None:
            return 0.0
        return self._delegate._predict_forward_return(df, horizon)

    def score_at(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        horizon: int = 21,
    ) -> float:
        if self._available is None:
            self._available = self._load_model()
        if not self._available or self._delegate is None:
            return NEUTRAL
        return self._delegate.score_at(df, date, horizon)

    def precompute_rebalance_scores(
        self,
        ticker: str,
        df: pd.DataFrame,
        dates: pd.DatetimeIndex,
        horizon: int = 21,
    ) -> Dict[pd.Timestamp, float]:
        if self._available is None:
            self._available = self._load_model()
        if not self._available or self._delegate is None:
            return {d: NEUTRAL for d in dates}
        return self._delegate.precompute_rebalance_scores(ticker, df, dates, horizon)

    @property
    def backend_name(self) -> str:
        if self._delegate is None:
            return "none"
        return self._delegate.__class__.__name__


# ── Factory ───────────────────────────────────────────────────────────────────

def build_foundation_signal(
    model: str = "auto",
    size: str = "mini",
    device: Optional[str] = None,
) -> FoundationSignalBase:
    """
    Build a foundation signal component.

    Args:
        model:  "auto"    → try Kronos, then Chronos (recommended)
                "kronos"  → Kronos only (OHLCV-specific, install required)
                "chronos" → Chronos only (close-price, easy install)
        size:   model size variant (see each class for options)
        device: "cpu", "cuda", or None (auto-detect)

    Returns:
        FoundationSignalBase instance (always safe to use — fail-open)

    Example:
        fs = build_foundation_signal("auto")
        score = fs.score_at(df, date, horizon=21)  # → float in [0, 1]
    """
    if model == "kronos":
        return KronosSignalComponent(model_size=size, device=device)
    elif model == "chronos":
        return ChronosSignalComponent(model_size=size, device=device)
    else:
        prefer = "kronos" if model in ("auto", "best") else "chronos"
        return FoundationSignalAuto(prefer=prefer, device=device)


# ── Utility: score blend ──────────────────────────────────────────────────────

def blend_with_momentum(
    momentum_score: float,
    foundation_score: float,
    foundation_weight: float = 0.10,
) -> float:
    """
    Blend a momentum score (dominant) with a foundation model score (augmenting).

    Default 10% weight for foundation: conservative enough to not overrule
    a strong momentum signal, but enough to improve ranking when both agree.

    Args:
        momentum_score:    Score from MomentumSignalEngine (0–1)
        foundation_score:  Score from a FoundationSignalBase component (0–1)
        foundation_weight: Fraction allocated to foundation model (default 0.10)

    Returns:
        Blended score in [0, 1]
    """
    w = float(np.clip(foundation_weight, 0.0, 1.0))
    blended = (1.0 - w) * momentum_score + w * foundation_score
    return float(np.clip(blended, 0.0, 1.0))
