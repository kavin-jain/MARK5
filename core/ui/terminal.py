"""
MARK5 HFT TERMINAL (READ-ONLY VIEW)
-----------------------------------
Renders system state from Redis/Memory.
Does NOT execute trading logic or fetch data.
"""

import time
import logging
import threading
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.align import Align
from rich import box

from core.system.container import container

class MARK5Terminal:
    def __init__(self, container_ref):
        self.container = container_ref
        self.console = Console()
        self.running = False
        self.redis = container.redis
        
        # Config
        self.watchlist = container.config.ui.default_watchlist

    def run(self):
        """Main Event Loop"""
        self.running = True
        self.console.clear()
        
        with Live(self._make_layout(), refresh_per_second=4, screen=True) as live:
            while self.running:
                try:
                    live.update(self._make_layout())
                    time.sleep(0.25)
                except KeyboardInterrupt:
                    self.running = False

    def _make_layout(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="market_data", ratio=2),
            Layout(name="system_status", ratio=1)
        )
        
        layout["header"].update(self._render_header())
        layout["market_data"].update(self._render_market_table())
        layout["system_status"].update(self._render_status_panel())
        layout["footer"].update(self._render_footer())
        
        return layout

    def _render_header(self):
        # Fetch System State from Redis
        state = self.redis.get("system:state") or "UNKNOWN"
        color = "green" if state == "TRADING" else "yellow"
        
        return Panel(
            Align.center(f"[bold white]MARK5 HFT ENGINE[/] | STATE: [{color}]{state}[/] | {datetime.now().strftime('%H:%M:%S')}"),
            style=f"white on {color}" if state == "TRADING" else "white on blue"
        )

    def _render_market_table(self):
        table = Table(expand=True, box=box.SIMPLE_HEAD)
        table.add_column("Ticker", style="cyan")
        table.add_column("LTP", justify="right")
        table.add_column("Signal", justify="center")
        table.add_column("Conf", justify="right")
        table.add_column("Position", justify="right")

        for ticker in self.watchlist:
            # 1. Get Price (Hot Cache)
            price_data = self.redis.get_market_data(ticker)
            ltp = float(price_data.get('last_price', 0.0)) if price_data else 0.0
            
            # 2. Get Signal (Latest Prediction)
            # Assuming Predictor writes to Redis key 'pred:{ticker}'
            pred = self.redis.get(f"pred:{ticker}") or {}
            signal = pred.get('signal', '-')
            conf = float(pred.get('confidence', 0.0))
            
            # 3. Get Position
            # Assuming OMS writes position summary
            pos = "FLAT" # Fetch from OMS in real impl
            
            # Styling
            sig_style = "green" if signal == "BUY" else "red" if signal == "SELL" else "dim"
            
            table.add_row(
                ticker,
                f"{ltp:.2f}",
                f"[{sig_style}]{signal}[/]",
                f"{conf:.1f}%",
                pos
            )
            
        return Panel(table, title="Live Market Data")

    def _render_status_panel(self):
        # System Health from Redis Heartbeats
        workers = ["DataIngestion", "FeatureEngine", "Inference"]
        status_str = ""
        
        for w in workers:
            # Check heartbeat (should be updated every 1s)
            last_beat = float(self.redis.get(f"heartbeat:{w}") or 0)
            lag = time.time() - last_beat
            
            color = "green" if lag < 2.0 else "red"
            status = "OK" if lag < 2.0 else "STALLED"
            status_str += f"[{color}]●[/] {w}: {status} ({lag:.1f}s)\n"

        return Panel(status_str, title="Process Health")

    def _render_footer(self):
        return Panel("Press Ctrl+C to Stop System safely.", style="dim")
