import logging
import time
import threading

logger = logging.getLogger("MARK5.FeedManager")

class FeedManager:
    """
    Manages data feeds (Kite, etc.)
    """
    def __init__(self, stream_producer=None):
        self.stream_producer = stream_producer
        self.running = False
        self.thread = None

    def start(self):
        logger.info("Feed Manager Starting...")
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            time.sleep(1)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Feed Manager Stopped.")
