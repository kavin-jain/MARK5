import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from core.api.websocket_manager import manager
import random
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mark5_api")

app = FastAPI(title="MARK5 API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "system": "MARK5 Trading Engine"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages (subscriptions, orders, etc.)
            message = json.loads(data)
            logger.info(f"Received: {message}")
            
            if message.get("type") == "subscribe":
                # Mock subscription handling
                await manager.send_personal_message(json.dumps({"type": "status", "msg": "Subscribed"}), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Mock Data Generator (Background Task)
async def mock_market_data_stream():
    """Simulates real-time market data for UI development."""
    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY_50"]
    while True:
        update = {
            "type": "market_update",
            "data": {
                ticker: {
                    "price": round(random.uniform(1000, 3000), 2),
                    "change": round(random.uniform(-2, 2), 2),
                    "volume": random.randint(1000, 50000)
                } for ticker in tickers
            },
            "timestamp": "2025-11-22T10:00:00Z" # Mock timestamp
        }
        await manager.broadcast(update)
        await asyncio.sleep(1) # 1 second update rate

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(mock_market_data_stream())
