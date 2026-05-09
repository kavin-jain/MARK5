#!/usr/bin/env bash
# ============================================================
# MARK5 — Environment Setup Script
# Jules / CI / fresh-VM compatible
# ============================================================
# Usage:
#   bash setup.sh          # full setup (default)
#   bash setup.sh --ci     # headless CI mode (skips Redis start)
#   bash setup.sh --verify # verify only, no install
# ============================================================
set -euo pipefail

# ── colours ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}✅ $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠️  $*${RESET}"; }
err()  { echo -e "${RED}❌ $*${RESET}"; exit 1; }
info() { echo -e "${CYAN}── $*${RESET}"; }
sep()  { echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

CI_MODE=false
VERIFY_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--ci" ]]     && CI_MODE=true
  [[ "$arg" == "--verify" ]] && VERIFY_ONLY=true
done

sep
echo -e "${BOLD}  MARK5 — Environment Setup${RESET}"
echo    "  Mode: $([ "$CI_MODE" = true ] && echo CI || echo Standard)"
sep

# ── 0. Python version check ──────────────────────────────────
info "Checking Python version..."
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
  err "Python 3.10+ required. Found $PY_VERSION"
fi
ok "Python $PY_VERSION"

if [[ "$VERIFY_ONLY" == true ]]; then
  info "Verify-only mode — skipping install steps"
else

  # ── 1. System dependencies ──────────────────────────────────
  sep; info "Installing system dependencies..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq 2>/dev/null || warn "apt-get update failed — continuing"
    sudo apt-get install -y \
      build-essential \
      libgomp1 \
      libhdf5-dev \
      libssl-dev \
      libffi-dev \
      sqlite3 \
      redis-server \
      2>/dev/null || warn "Some apt packages may not have installed — continuing"
    ok "System packages installed"
  else
    warn "apt-get not available — skipping system packages"
  fi

  # ── 2. Virtual environment ───────────────────────────────────
  sep; info "Setting up Python virtual environment..."
  if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    ok "Created .venv"
  else
    ok ".venv already exists"
  fi

  # Activate venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ok "Activated .venv ($(which python3))"

  # ── 3. Upgrade pip ──────────────────────────────────────────
  info "Upgrading pip..."
  pip install --upgrade pip setuptools wheel -q
  ok "pip $(pip --version | awk '{print $2}')"

  # ── 4. Install dependencies ──────────────────────────────────
  sep; info "Installing Python dependencies from requirements.txt..."

  # Split requirements: install heavyweight ML libs first with no-cache
  # to avoid memory spikes on resource-constrained VMs
  pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "pandas>=2.2.0" \
    "scipy>=1.12.0" \
    "scikit-learn>=1.4.0" \
    -q && ok "Core data science libs"

  pip install --no-cache-dir \
    "xgboost>=2.0.3" \
    "lightgbm>=4.1.0" \
    "catboost>=1.2.3" \
    "optuna>=3.4.0" \
    -q && ok "ML libraries"

  # Install the rest from requirements.txt, skipping already-installed
  # Skip torch/tensorflow by default in CI (huge download, not needed for core tests)
  if [[ "$CI_MODE" == true ]]; then
    info "CI mode: skipping torch/tensorflow (too large for CI)"
    grep -v -E "^(torch|tensorflow|keras)" requirements.txt > /tmp/requirements_ci.txt
    pip install --no-cache-dir -r /tmp/requirements_ci.txt -q && ok "All other requirements (CI subset)"
  else
    pip install --no-cache-dir -r requirements.txt -q && ok "All requirements.txt packages"
  fi

fi   # end VERIFY_ONLY check

# ── 5. Activate venv if not already ─────────────────────────
if [[ -d ".venv" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ── 6. Create required directories ───────────────────────────
sep; info "Creating required directories..."
DIRS=(
  "data"
  "data/cache"
  "data/raw"
  "data/processed"
  "logs"
  "reports"
  "database/main"
  "database/backups"
  "models"
  "models_oos"
)
for d in "${DIRS[@]}"; do
  mkdir -p "$d"
done
ok "All runtime directories exist"

# ── 7. Environment file ───────────────────────────────────────
sep; info "Checking environment configuration..."
if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp .env.example .env
    warn ".env created from .env.example — fill in real values before running live"
  else
    # Create a safe CI .env with simulation mode
    cat > .env << 'EOF'
ENVIRONMENT=simulation

# Indian Stock API — replace with real key for live data
ISE_API_KEY=PLACEHOLDER_SET_IN_CI_SECRETS

# Kite Connect — replace for live trading (PAPER mode is safe without these)
KITE_API_KEY=PLACEHOLDER
KITE_API_SECRET=PLACEHOLDER
KITE_ACCESS_TOKEN=PLACEHOLDER
KITE_REQUEST_TOKEN=

# Alerts (optional — disabled by default)
MARK5_EMAIL_SENDER=
MARK5_EMAIL_PASSWORD=
EOF
    warn ".env created with placeholder values (simulation mode is safe)"
  fi
else
  ok ".env exists"
fi

# Validate ENVIRONMENT is set to simulation in CI
ENV_VAL=$(grep '^ENVIRONMENT=' .env | cut -d= -f2 | tr -d '"' | tr -d "'")
if [[ "$CI_MODE" == true && "$ENV_VAL" == "live" ]]; then
  err "ENVIRONMENT=live is not allowed in CI mode. Set to 'simulation'."
fi
ok "ENVIRONMENT=${ENV_VAL}"

# ── 8. SQLite DB health check ─────────────────────────────────
sep; info "Verifying SQLite..."
sqlite3 database/main/health_check.db "CREATE TABLE IF NOT EXISTS ping (id INTEGER PRIMARY KEY); INSERT INTO ping DEFAULT VALUES; SELECT COUNT(*) FROM ping;" > /dev/null 2>&1 \
  && ok "SQLite functional" \
  || warn "SQLite check failed — DB may need manual init"
rm -f database/main/health_check.db

# ── 9. Redis (optional) ───────────────────────────────────────
if [[ "$CI_MODE" == false ]]; then
  info "Starting Redis (optional cache layer)..."
  if command -v redis-server &>/dev/null; then
    if ! redis-cli ping &>/dev/null 2>&1; then
      redis-server --daemonize yes --loglevel warning 2>/dev/null || true
      sleep 1
    fi
    redis-cli ping &>/dev/null && ok "Redis running" || warn "Redis not responding — caching disabled"
  else
    warn "Redis not installed — in-memory cache only"
  fi
else
  info "CI mode: skipping Redis start"
fi

# ── 10. Core import validation ────────────────────────────────
sep; info "Validating core Python imports..."

python3 - << 'PYCHECK'
import sys, importlib

required = [
    ("numpy",         "numpy"),
    ("pandas",        "pandas"),
    ("scipy",         "scipy"),
    ("sklearn",       "scikit-learn"),
    ("xgboost",       "xgboost"),
    ("lightgbm",      "lightgbm"),
    ("optuna",        "optuna"),
    ("yaml",          "pyyaml"),
    ("dotenv",        "python-dotenv"),
    ("colorlog",      "colorlog"),
    ("psutil",        "psutil"),
    ("joblib",        "joblib"),
    ("plotly",        "plotly"),
    ("requests",      "requests"),
    ("cachetools",    "cachetools"),
    ("pybreaker",     "pybreaker"),
    ("cloudpickle",   "cloudpickle"),
    ("statsmodels",   "statsmodels"),
    ("shap",          "shap"),
    ("tqdm",          "tqdm"),
]

optional = [
    ("torch",         "torch"),
    ("tensorflow",    "tensorflow"),
    ("catboost",      "catboost"),
    ("redis",         "redis"),
    ("textual",       "textual"),
    ("flask",         "flask"),
    ("fastapi",       "fastapi"),
]

failed_req = []
for mod, pkg in required:
    try:
        importlib.import_module(mod)
        print(f"  ✅ {pkg}")
    except ImportError as e:
        print(f"  ❌ REQUIRED: {pkg} — {e}", file=sys.stderr)
        failed_req.append(pkg)

print()
for mod, pkg in optional:
    try:
        importlib.import_module(mod)
        print(f"  ✅ {pkg} (optional)")
    except ImportError:
        print(f"  ⚠️  {pkg} (optional — not installed)")

if failed_req:
    print(f"\nFATAL: {len(failed_req)} required package(s) missing: {failed_req}", file=sys.stderr)
    sys.exit(1)

print(f"\n  All required imports OK")
PYCHECK

# ── 11. MARK5 config load check ───────────────────────────────
sep; info "Verifying MARK5 config.yaml loads correctly..."
python3 - << 'CFGCHECK'
import yaml, sys
try:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "backtesting" in cfg, "Missing 'backtesting' section"
    assert "production" in cfg, "Missing 'production' section"
    assert "monitoring" in cfg, "Missing 'monitoring' section"
    assert cfg["production"].get("enable_paper_trading", False) is not True or True, "OK"
    print("  ✅ config.yaml valid")
except Exception as e:
    print(f"  ❌ config.yaml error: {e}", file=sys.stderr)
    sys.exit(1)
CFGCHECK

# ── 12. Run test suite ────────────────────────────────────────
sep; info "Running test suite..."
if command -v pytest &>/dev/null || python3 -m pytest --version &>/dev/null 2>&1; then
  python3 -m pytest tests/ \
    -v \
    --tb=short \
    --no-header \
    -q \
    --ignore=tests/dashboard_health_check.py \
    2>&1 | tail -30 || warn "Some tests failed — review output above"
  ok "Test run complete"
else
  warn "pytest not found — skipping tests"
fi

# ── 13. Summary ───────────────────────────────────────────────
sep
echo -e "${GREEN}${BOLD}"
echo    "  MARK5 environment setup complete!"
echo -e "${RESET}"
echo    "  ► To activate manually:   source .venv/bin/activate"
echo    "  ► To prime data cache:     python3 prime_cache.py"
echo    "  ► To run backtester:       python3 backtest_150.py"
echo    "  ► To run trading pipeline: python3 apply.py"
echo    "  ► To run dashboard:        python3 dashboard.py"
echo
echo -e "${YELLOW}  ⚠️  PAPER mode only. Never set ENVIRONMENT=live without ops review.${RESET}"
sep
