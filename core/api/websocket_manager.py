import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger("Mark5.WSManager")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_non_blocking(self, message: str):
        # This is a simplified version. 
        # The Architect's version in rest_api.py is superior (asyncio.gather).
        # This file is kept for compatibility if main.py imports it.
        # Ideally, we should unify with rest_api.py's manager.
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()
