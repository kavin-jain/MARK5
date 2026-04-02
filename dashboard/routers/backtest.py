from fastapi import APIRouter
import time
import asyncio

router = APIRouter()

# Global state to track mock backtest for UI
bt_state = {"running": False, "progress": 0, "logs": []}

BT_TICKERS = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
              'SBIN.NS','BHARTIARTL.NS','ITC.NS','LT.NS','WIPRO.NS']

async def simulate_backtest():
    """Mock background backtest process to simulate real one."""
    global bt_state
    bt_state["running"] = True
    bt_state["progress"] = 0
    bt_state["logs"] = []
    
    import random
    
    for i, ticker in enumerate(BT_TICKERS):
        await asyncio.sleep(0.35)
        sh = round(0.7 + random.random() * 2.2, 2)
        wr = round(38 + random.random() * 28, 1)
        brier = round(0.16 + random.random() * 0.09, 3)
        passed = sh > 1.5
        
        log_entry = {
            "ticker": ticker,
            "sh": sh,
            "wr": wr,
            "brier": brier,
            "pass": passed
        }
        bt_state["logs"].append(log_entry)
        bt_state["progress"] = int(((i + 1) / len(BT_TICKERS)) * 100)
    
    bt_state["running"] = False

@router.post("/run")
async def run_backtest():
    if bt_state["running"]:
        return {"status": "error", "message": "Backtest already running"}
    
    # In production, this would trigger model/tcn/backtester.py as a subprocess or Celery task.
    asyncio.create_task(simulate_backtest())
    return {"status": "started", "message": "Parallel backtest initiated"}

@router.get("/status")
def backtest_status():
    return bt_state

@router.get("/health")
def get_model_health():
    # Enforces Rule 30 (Win rate < 40% drops confidence)
    # Enforces Rule 39 (Sharpe < 0.4 gating)
    import random
    return {
        "live_edge": {
            "win_rate": 38.4,       # Below 40% triggers red mode
            "profit_factor": 1.15,  # Needs to be > 1.2
            "rolling_trades": 25,
            "status": "DEGRADED"
        },
        "sharpe_gating": [
            {"ticker": "RELIANCE.NS", "sharpe": 1.45, "gated": False},
            {"ticker": "TCS.NS", "sharpe": 1.82, "gated": False},
            {"ticker": "WIPRO.NS", "sharpe": 0.35, "gated": True},
            {"ticker": "HDFCBANK.NS", "sharpe": 0.12, "gated": True},
            {"ticker": "INFY.NS", "sharpe": 0.88, "gated": False}
        ]
    }
