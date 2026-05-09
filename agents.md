# MARK5 — Agent Environment Guide

> **For Jules and other AI coding agents:** This file tells you how to set up and run MARK5.
> If you have a setup script (`setup.sh`), run it. Otherwise use the hints below.

---

## Project Overview

MARK5 is a production-grade, ML-driven algorithmic trading system for NSE Indian equities.
- **Language:** Python 3.12+
- **Mode:** PAPER (simulation only — never switch to LIVE)
- **Exchange:** NSE — Midcap 150 / NIFTY100
- **Capital:** ₹5,00,00,000 paper pool
- **DB stack:** SQLite (main) + Redis (cache) — TimescaleDB optional
- **ML stack:** XGBoost + LightGBM ensemble (two-layer: weekly rank → daily ML entry)

---

## Quick-Start for Agents

```bash
# Run the full environment setup
bash setup.sh

# Or step-by-step:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create required directories
mkdir -p data logs reports database/main database/backups models models_oos

# Copy env template and fill placeholders
cp .env.example .env   # if available, else use existing .env

# Verify core imports work
python3 -c "import pandas, numpy, sklearn, xgboost, lightgbm, optuna; print('Core OK')"

# Run tests
pytest tests/ -v --tb=short
```

---

## Environment Variables Required

| Variable | Purpose | Required |
|---|---|---|
| `ENVIRONMENT` | `simulation` or `live` | **Yes** |
| `ISE_API_KEY` | Indian Stock Exchange API | Yes for live data |
| `KITE_API_KEY` | Zerodha Kite API key | Yes for live trading |
| `KITE_API_SECRET` | Zerodha Kite secret | Yes for live trading |
| `KITE_ACCESS_TOKEN` | Zerodha session token | Refreshed daily |
| `MARK5_EMAIL_SENDER` | Alert emails sender | No (alerts disabled) |
| `MARK5_EMAIL_PASSWORD` | Email SMTP password | No (alerts disabled) |

> All secrets must come from environment variables — **never hardcode** into source files.

---

## Python Version

Requires **Python 3.12+**.  
Jules VMs ship with Python 3.12.11 — no version management needed.

---

## System Dependencies (apt)

These C-level libraries must be installed before `pip install`:

```bash
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    libgomp1 \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    redis-server \
    sqlite3
```

> **TA-Lib** is intentionally excluded from `requirements.txt` (requires C lib compilation).
> If you need TA-Lib: `sudo apt-get install -y libta-lib-dev && pip install TA-Lib`

---

## Required Directory Structure

MARK5 expects these directories to exist at runtime:

```
data/               # Market data cache (CSV/Parquet)
logs/               # System and trade logs
reports/            # Backtest and analytics reports
database/main/      # SQLite databases
database/backups/   # Database snapshots
models/             # Trained model artifacts (.pkl, .json)
models_oos/         # Out-of-sample model evaluation artifacts
```

Create them all at once:

```bash
mkdir -p data logs reports database/main database/backups models models_oos
```

---

## Key Entry Points

| Script | Purpose |
|---|---|
| `apply.py` | Run the full signal-generation and paper-trading pipeline |
| `backtest_150.py` | Backtest across the NSE Midcap 150 universe |
| `dashboard.py` | Launch the Textual TUI monitoring dashboard |
| `prime_cache.py` | Pre-warm market data cache before trading session |
| `diagnostic_ic.py` | Check feature Information Coefficient |
| `diagnostic_separation.py` | Check model discriminative power |

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v --tb=short

# Single test file
pytest tests/test_universe_optimizer.py -v

# With coverage
pytest tests/ --cov=core --cov-report=term-missing
```

Test configuration is in `pytest.ini`.  
Tests live in `tests/` only — do not add test files to the project root.

---

## ML Model Notes

- **Approved feature set (8 features):** `fii_flow_3d`, `relative_strength_nifty`, `dist_52w_high`,
  `amihud_illiquidity`, `gap_significance`, `sector_rel_strength`, `volume_confirmation`, `atr_regime`
- **Never use `yfinance`** — all data comes from `nsepython`, Kite Connect, or the Indian Stock API.
- **Model retraining:** Delete `models/<SYMBOL>/` before retraining to avoid stale artifact regression.
- **Calibration:** Requires ≥500 samples; use raw probabilities below that threshold.

---

## Config

Main config lives in `config.yaml`. Key sections:
- `backtesting.initial_capital` — ₹1,00,000 default (override for ₹5cr paper pool)
- `production.enable_paper_trading` — **always `true`** in CI/agent environments
- `monitoring.log_level` — set to `DEBUG` in dev, `INFO` in CI

---

## Safety Rules for Agents

1. **Never flip `enable_paper_trading` to `false`** — that enables real-money trading.
2. **Never bypass the 4-layer risk architecture** (pre-trade, real-time, portfolio, circuit-breaker).
3. **No `print()` statements** — use `logging.getLogger("MARK5.<ModuleName>")`.
4. **No `time.sleep()` in trading paths** — use event-driven patterns.
5. **All financial math in `Decimal`** — never `float` for P&L or capital.
6. **No new config fields** without first adding them to `core/config/validators.py`.
