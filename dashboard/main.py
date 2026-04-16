from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dashboard.routers import system, backtest, data
import uvicorn
import os
from dotenv import load_dotenv

# Load all env vars from MARK5 root .env
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
from contextlib import asynccontextmanager

from core.data.adapters.kite_adapter import KiteFeedAdapter
from core.infrastructure.database_manager import MARK5DatabaseManager
from core.data.adapters.ise_adapter import ISEAdapter

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup global adapters
    app.state.db = MARK5DatabaseManager()
    
    # Init Kite Feed
    app.state.kite = KiteFeedAdapter({})
    app.state.kite.connect()
    
    # Init ISE
    app.state.ise = ISEAdapter()
    
    yield
    
    # Shutdown
    app.state.kite.disconnect()
    app.state.db.close()

app = FastAPI(title="MARK5 Dashboard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/system", tags=["System Controls"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest Framework"])
app.include_router(data.router, prefix="/api/data", tags=["Market Data & PnL"])

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "MARK5"}

if __name__ == "__main__":
    uvicorn.run("dashboard.main:app", host="0.0.0.0", port=8000, reload=True)
