"""
MARK5 TIME SYNCHRONIZATION
--------------------------
NTP-Correction for millisecond-accurate logging and trade execution.
"""

import time
import ntplib
import logging
from datetime import datetime, timezone

logger = logging.getLogger("MARK5.Time")

class TimeSyncManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimeSyncManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.offset = 0.0
        self.last_sync = 0
        self._initialized = True

    def sync(self, server="pool.ntp.org") -> bool:
        try:
            client = ntplib.NTPClient()
            response = client.request(server, version=3)
            self.offset = response.offset
            self.last_sync = time.time()
            logger.info(f"✅ NTP Synced. Offset: {self.offset:.6f}s")
            return True
        except Exception as e:
            logger.warning(f"⚠️ NTP Sync Failed: {e}. Using Local Time.")
            return False

    def now(self) -> float:
        """Get corrected unix timestamp"""
        return time.time() + self.offset

    def iso_now(self) -> str:
        """Get corrected ISO string"""
        ts = self.now()
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
