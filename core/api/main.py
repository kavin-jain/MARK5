import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from core.api.websocket_manager import manager
import random
from core.utils.fast_io import fast_dumps, current_time_ns

# Configure logging (Only for startup/errors, not per-tick)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mark5_api")

app = FastAPI(title="MARK5 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # wait for messages (subscriptions)
            data = await websocket.receive_text()
            # Process strictly necessary logic only
            # No logging here in production!
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WS Error: {e}")
        manager.disconnect(websocket)

async def mock_market_data_stream():
    """
    Optimized Mock Generator.
    Uses fast_dumps and minimizes GC overhead.
    """
    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY_50"]
    
    # Pre-structure the dictionary to avoid re-hashing keys
    base_structure = {
        "type": "market_update",
        "data": {},
        "timestamp": 0
    }
    
    logger.info("Starting High-Speed Mock Stream...")
    
    while True:
        # Update timestamp
        base_structure["timestamp"] = current_time_ns()
        
        # Update data (Simulate fast movement)
        # Note: In real system, this comes from Redis/SHM, not random
        base_structure["data"] = {
            ticker: {
                "price": round(random.uniform(1000, 3000), 2),
                "vol": random.randint(100, 500)
            } for ticker in tickers
        }
        
        # Serialize with Rust-based serializer (orjson)
        payload = fast_dumps(base_structure)
        
        # Fire and forget
        # Note: manager.broadcast_non_blocking in websocket_manager.py needs to be async
        # We await it here because the implementation in websocket_manager is simple loop
        await manager.broadcast_non_blocking(payload)
        
        # Throttle to simulate 10 ticks/sec (100ms)
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    # Run mock stream as a background task
    asyncio.create_task(mock_market_data_stream())
