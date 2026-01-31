import asyncio
import logging
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import redis.asyncio as redis # Use async redis driver
from core.utils.fast_io import fast_dumps

logger = logging.getLogger("Mark5.Gateway")

class ConnectionManager:
    def __init__(self):
        # Set for O(1) removals vs List O(N)
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client Connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    def broadcast_non_blocking(self, message: str):
        """
        Fire-and-forget broadcast. 
        We do not await the send inside the loop to avoid Head-of-Line blocking.
        """
        # Snapshot the set to avoid modification during iteration errors
        clients = list(self.active_connections) 
        if not clients:
            return

        # Create background tasks for sending. 
        # In extreme load (10k+ users), use a dedicated broadcasting microservice/Redis PubSub.
        # For <1000 users, asyncio.gather is highly efficient.
        asyncio.create_task(self._broadcast_task(clients, message))

    async def _broadcast_task(self, clients: List[WebSocket], message: str):
        # Parallel send
        await asyncio.gather(
            *[self._safe_send(ws, message) for ws in clients], 
            return_exceptions=True
        )

    async def _safe_send(self, ws: WebSocket, message: str):
        try:
            await ws.send_text(message)
        except Exception:
            # If send fails, assume disconnect or network error
            self.disconnect(ws)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Redis Bridge
    task = asyncio.create_task(redis_bridge())
    yield
    # Cleanup
    task.cancel()

app = FastAPI(title="MARK5 HFT Interface", lifespan=lifespan)

@app.websocket("/ws/market")
async def websocket_market_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open, ignore incoming data (unidirectional stream)
            # await websocket.receive_text() would block until client speaks
            # simpler is to just sleep deeply or wait for a specific 'ping'
            await asyncio.sleep(60) 
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def redis_bridge():
    """
    High-Performance Redis -> WebSocket Bridge.
    Uses Blocking Reads (XREAD) to eliminate sleep-polling latency.
    """
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        # Ensure connection
        await r.ping()
        logger.info("Redis Bridge Connected")
        
        last_id = '$'
        
        while True:
            try:
                # BLOCK=0 means wait indefinitely until new data arrives.
                # BLOCK=100 means wait 100ms.
                # using 100ms allows us to check for shutdown signals periodically
                streams = await r.xread(streams={"market_ticks": last_id}, count=50, block=100)
                
                if not streams:
                    continue

                for stream_name, messages in streams:
                    for message_id, data in messages:
                        last_id = message_id
                        
                        # Use fast_dumps for serialization
                        payload = fast_dumps({
                            "t": "tick",
                            "d": data,
                            "ts": message_id # Use Redis ID as High-Res Timestamp
                        })
                        
                        manager.broadcast_non_blocking(payload)
                        
            except asyncio.CancelledError:
                logger.info("Bridge stopping...")
                break
            except Exception as e:
                logger.error(f"Bridge connection error: {e}")
                await asyncio.sleep(1) # Backoff only on error
                
    finally:
        await r.aclose()
