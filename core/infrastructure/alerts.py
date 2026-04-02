"""
MARK5 ALERT MANAGER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification
- [Previous] v2.0: Async architecture with retry mechanism

TRADING ROLE: Non-blocking system-wide alerting (Email/Slack)
SAFETY LEVEL: CRITICAL - Main thread latency protection

FEATURES:
✅ Producer-Consumer Queue (Zero main-thread blocking)
✅ Retry mechanism with exponential backoff
✅ Thread-safe Singleton pattern
"""

import logging
import json
import requests
import threading
import queue
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum


# ── Enums required by autonomous.py ──────────────────────────────────────────
class AlertLevel(Enum):
    INFO     = "INFO"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    SYSTEM_ERROR        = "SYSTEM_ERROR"
    TRADE_EXECUTED      = "TRADE_EXECUTED"
    RISK_BREACH         = "RISK_BREACH"
    HIGH_RISK_WARNING   = "HIGH_RISK_WARNING"
    MODEL_DEGRADATION   = "MODEL_DEGRADATION"
    DATA_QUALITY        = "DATA_QUALITY"
    CIRCUIT_BREAKER     = "CIRCUIT_BREAKER"
    WEEKLY_LIMIT        = "WEEKLY_LIMIT"
    MONTHLY_LIMIT       = "MONTHLY_LIMIT"
    DRAWDOWN_STOP       = "DRAWDOWN_STOP"
    GENERAL             = "GENERAL"

# Configure logging
logger = logging.getLogger("MARK5_Alerts")

class AlertManager:
    """
    MARK5 High-Performance Asynchronous Alert System.
    Architecture: Producer-Consumer via Thread-Safe Queue.
    Guarantee: Main thread never blocks on I/O.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Thread-safe Singleton Pattern using Double-Checked Locking"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(AlertManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.alert_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.config = {}
        self.enabled = False
        
        # State flags
        self.email_enabled = False
        self.slack_enabled = False
        
        self._initialized = True
        self._start_worker()
        logger.info("AlertManager V2 (Async) initialized.")

    def _start_worker(self):
        """Start the background consumer thread."""
        self.worker_thread = threading.Thread(target=self._worker_loop, name="AlertWorker", daemon=True)
        self.worker_thread.start()

    def configure(self, config: Dict):
        """
        Hot-reloadable configuration.
        """
        with self._lock:
            self.config = config.get('alerts', {})
            self.enabled = self.config.get('enabled', False)
            
            # Email Config
            self.email_config = self.config.get('email', {})
            self.email_enabled = self.email_config.get('enabled', False)
            self.smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            self.smtp_port = self.email_config.get('smtp_port', 587)
            self.sender_email = self.email_config.get('sender_email', '')
            self.sender_password = os.getenv('SMTP_PASSWORD', self.email_config.get('sender_password', ''))
            self.recipient_emails = self.email_config.get('recipients', [])

            # Slack Config
            self.slack_config = self.config.get('slack', {})
            self.slack_enabled = self.slack_config.get('enabled', False)
            self.slack_webhook_url = os.getenv('SLACK_WEBHOOK', self.slack_config.get('webhook_url', ''))

            logger.info(f"AlertManager configured. Mode: {'ACTIVE' if self.enabled else 'SILENT'}")

    def create_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        title: str,
        message: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Structured alert creation — accepts enum values from autonomous.py.
        Converts to string keys for the existing queue-based pipeline.
        """
        level_str = level.value if isinstance(level, AlertLevel) else str(level)
        type_str  = alert_type.value if isinstance(alert_type, AlertType) else str(alert_type)
        full_msg  = f"[{type_str}] {message}"
        if metadata:
            full_msg += f"\nMetadata: {metadata}"
        self.send_alert(level=level_str, title=title, message=full_msg)

    def send_alert(self, level: str, title: str, message: str):
        """
        Non-blocking alert dispatch.
        Puts payload into queue and returns control to trading loop immediately.
        """
        if not self.enabled:
            return

        payload = {
            'level': level,
            'title': title,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.alert_queue.put_nowait(payload)
        except queue.Full:
            logger.error("CRITICAL: Alert queue full! Dropping alert to preserve core latency.")

    def _worker_loop(self):
        """
        Consumer loop. Processes alerts one by one to prevent network congestion.
        """
        while not self.stop_event.is_set():
            try:
                # Block for 1s to allow checking stop_event periodically
                payload = self.alert_queue.get(timeout=1.0)
                self._process_alert(payload)
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in alert worker: {e}")

    def _process_alert(self, payload: Dict):
        """Actual I/O operations happen here, isolated from main thread."""
        level = payload['level']
        title = payload['title']
        message = payload['message']
        timestamp = payload['timestamp']
        
        full_message = f"[{level}] {title}\nTime: {timestamp}\n\n{message}"
        
        # 1. Internal Logging (Always safe)
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"ALERT DISPATCH: {title}")

        # 2. External Dispatch (Retried safely)
        if self.email_enabled:
            self._retry_operation(self._send_email, level, title, full_message)
            
        if self.slack_enabled:
            self._retry_operation(self._send_slack, level, title, message)

    def _retry_operation(self, func, *args, max_retries=2):
        """Exponential backoff for network stability."""
        for attempt in range(max_retries + 1):
            try:
                func(*args)
                return
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Failed to send alert after {max_retries} retries: {e}")
                else:
                    time.sleep(0.5 * (2 ** attempt)) # 0.5s, 1s wait

    def _send_email(self, level: str, title: str, body: str):
        if not self.sender_email or not self.recipient_emails:
            return

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(self.recipient_emails)
        msg['Subject'] = f"[MARK5] [{level}] {title}"
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)

    def _send_slack(self, level: str, title: str, message: str):
        if not self.slack_webhook_url:
            return

        colors = {'INFO': '#36a64f', 'WARNING': '#ffcc00', 'ERROR': '#ff0000', 'CRITICAL': '#7b0000'}
        payload = {
            "attachments": [{
                "color": colors.get(level, '#cccccc'),
                "title": f"[{level}] {title}",
                "text": message,
                "footer": "MARK5 Algo Core",
                "ts": int(time.time())
            }]
        }
        requests.post(self.slack_webhook_url, json=payload, timeout=5)

    def shutdown(self):
        """Graceful shutdown ensuring all pending alerts are sent."""
        logger.info("Shutting down AlertManager...")
        self.stop_event.set()
        
        # Process remaining items if needed
        if not self.alert_queue.empty():
            logger.warning(f"Flushing {self.alert_queue.qsize()} pending alerts...")
            while not self.alert_queue.empty():
                try:
                    payload = self.alert_queue.get_nowait()
                    self._process_alert(payload)
                except queue.Empty:
                    break
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

# Global Access Point
alert_manager = AlertManager()
