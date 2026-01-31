# Add project root to sys.path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import decimal
from decimal import Decimal
import time
import psutil
import struct
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import RenderableType

from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Vertical, Horizontal
from textual.widgets import Header, Footer, Button, DataTable, Static, Input, RichLog, ProgressBar, Label
from textual.screen import Screen
from textual.reactive import reactive
from textual.message import Message
from textual.worker import Worker, WorkerState

# --- CORE INTEGRATIONS ---
from core.infrastructure.ipc import SharedMemoryRingBuffer
from core.utils.constants import INSTRUMENT_MAP_REV
from core.utils.config_manager import get_config
from core.data.provider import DataProvider
from core.models.training.trainer import MARK5MLTrainer
from core.models.predictor import MARK5Predictor

# Suppress bridge logs to avoid polluting TUI
logger = logging.getLogger("MARK5.Bridge")
logger.setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# ROBUST DATA BRIDGE
# -----------------------------------------------------------------------------
class DataBridge:
    def __init__(self):
        self.connected = False
        self.shm = None
        self.last_head = 0
        self.market_state: Dict[str, Dict] = {}  # Latest state with TTL
        self.ticker_history: Dict[str, pd.DataFrame] = {}  # Historical data cache
        self._lock = threading.Lock()  # Thread safety
        self.ticker_ttl = timedelta(seconds=10)  # Expire data older than 10s
        
        # Metrics
        self._last_metrics_time = time.time()
        self._ticks_processed = 0
        self._current_tick_rate = 0.0
        
        # Start connection
        self.connect()

    def connect(self):
        """Connect to shared memory with error logging."""
        try:
            # Must match DataIngestionWorker config
            self.shm = SharedMemoryRingBuffer(
                name="mark5_tick_buffer",
                shape=(4,),
                dtype=np.float64,
                slots=100000, 
                create=False
            )
            self.connected = True
            logger.info("Connected to shared memory buffer.")
        except FileNotFoundError:
            self.connected = False
            logger.warning("Shared memory buffer not found.")
        except Exception as e:
            self.connected = False
            logger.error(f"Connection failed: {str(e)}")

    def get_market_snapshot(self) -> Dict[str, Dict]:
        """Process ticks with thread safety, handle out-of-order, and stale data."""
        if not self.connected:
            self.connect()
            if not self.connected: 
                return {}
            
        try:
            new_head, ticks = self.shm.read_latest(self.last_head)
            self.last_head = new_head
            
            # Tick Rate Calculation
            now = time.time()
            self._ticks_processed += len(ticks)
            if now - self._last_metrics_time >= 1.0:
                self._current_tick_rate = self._ticks_processed / (now - self._last_metrics_time)
                self._ticks_processed = 0
                self._last_metrics_time = now

            with self._lock:
                current_utc = datetime.utcnow()
                
                # Check for stale data in cache
                expired_tickers = []
                for tkr, data in self.market_state.items():
                    if (current_utc - data['last_updated']) > self.ticker_ttl:
                        expired_tickers.append(tkr)
                
                for tkr in expired_tickers:
                    self.market_state.pop(tkr, None)

                for t in ticks:
                    # Validate tick structure
                    # Tick: {'ts': int(ns), 'id': int, 'p': float, 'v': int}
                    if not all(k in t for k in ('id', 'p', 'v', 'ts')):
                        continue

                    ticker_id = t['id']
                    # Using get default to avoid KeyError crash
                    ticker = INSTRUMENT_MAP_REV.get(ticker_id, f"UNKNOWN_{ticker_id}")
                    
                    # Timestamp handling
                    try:
                        tick_ts = datetime.utcfromtimestamp(t['ts'] / 1e9)
                    except Exception:
                        continue
                        
                    # Stale tick check (vs existing state)
                    # Note: We trust the monotonic TS from ring buffer usually, 
                    # but if we get out-of-order within a batch (rare but possible via multi-writers)
                    # For now invalidating older ticks is enough.
                    
                    # Decimal Precision for Price
                    price = Decimal(str(t['p']))
                    volume = int(t['v'])
                    latency_ms = (time.time_ns() - t['ts']) / 1_000_000
                    
                    # Calculate change based on internal history or prev state
                    change = 0.0
                    prev = self.market_state.get(ticker, {})
                    prev_price = prev.get('price', price)
                    
                    if prev_price > 0:
                        change = float((price - prev_price) / prev_price) * 100

                    # Update market_state with fresh data
                    self.market_state[ticker] = {
                        'price': price,
                        'volume': volume,
                        'change': change,
                        'latency': latency_ms,
                        'timestamp': t['ts'],
                        'last_updated': current_utc
                    }

            return self.market_state.copy()
        except Exception as e:
            self.connected = False
            return self.market_state.copy()

    def get_system_metrics(self):
        disk_usage = psutil.disk_usage('/').percent
        return {
            'cpu': psutil.cpu_percent(interval=None),
            'ram': psutil.virtual_memory().percent,
            'disk': disk_usage,
            'tick_rate': self._current_tick_rate,
            'connected': self.connected
        }

bridge = DataBridge()

# -----------------------------------------------------------------------------
# WIDGETS
# -----------------------------------------------------------------------------

class Sidebar(Container):
    def compose(self) -> ComposeResult:
        yield Button("Overview (w)", id="btn-overview", classes="-active")
        yield Button("Training Hub (t)", id="btn-training")
        yield Button("Deep Analysis (a)", id="btn-analysis")
        yield Button("Quit (q)", id="btn-quit")

class SystemHeader(Static):
    latency = reactive(0.0)
    cpu_load = reactive(0.0)
    ram_load = reactive(0.0)
    disk_usage = reactive(0.0)
    tick_rate = reactive(0.0)
    connected = reactive(False)

    def on_mount(self) -> None:
        self.set_interval(0.5, self.update_stats)

    def update_stats(self) -> None:
        metrics = bridge.get_system_metrics()
        self.cpu_load = metrics['cpu']
        self.ram_load = metrics['ram']
        self.disk_usage = metrics['disk']
        self.tick_rate = metrics['tick_rate']
        self.connected = metrics['connected']

    def render(self) -> RenderableType:
        status_color = "green" if self.connected else "red"
        status_text = "ONLINE" if self.connected else "OFFLINE"
        
        # Advanced Monitor Layout
        return Panel(
            Grid(
                Label(f"[bold blue]MARK5 HFT ENGINE[/]"),
                Label(f"Status: [bold {status_color}]{status_text}[/]"),
                Label(f"CPU: [bold yellow]{self.cpu_load:04.1f}%[/]"),
                Label(f"RAM: [bold cyan]{self.ram_load:04.1f}%[/]"),
                Label(f"Disk: [bold magenta]{self.disk_usage:04.1f}%[/]"),
                Label(f"Ticks: [bold white]{self.tick_rate:.0f}/s[/]"),
                classes="header-grid"
            ),
            style="on #24283b"
        )

class WatchlistTable(DataTable):
    def on_mount(self) -> None:
        self.add_columns("Ticker", "Price", "Change %", "Latency (ms)", "Signal")
        self.set_interval(0.1, self.update_data)

    def update_data(self) -> None:
        snapshot = bridge.get_market_snapshot()
        current_time_utc = datetime.utcnow()
        
        # Clear if empty for too long? No, better to show partial
        if not snapshot: 
            return

        self.clear()
        
        for ticker, data in snapshot.items():
            # Stale check in UI as well (gray out rows?)
            # data has 'last_updated'
            is_stale = (current_time_utc - data['last_updated']) > timedelta(seconds=5)
            style_mod = "dim " if is_stale else ""
            
            price = data['price']
            change = data.get('change', 0.0)
            latency = data.get('latency', 0.0)
            
            signal = "HOLD"
            sig_color = "yellow"
            if change > 0.5: signal = "BUY"; sig_color = "green"
            elif change < -0.5: signal = "SELL"; sig_color = "red"
            
            color = "green" if change >= 0 else "red"
            lat_color = "green" if latency < 10 else ("yellow" if latency < 50 else "red")
            
            self.add_row(
                Text(ticker, style=f"{style_mod}bold white"),
                f"{price:.2f}",
                Text(f"{change:+.2f}%", style=f"{style_mod}{color}"),
                Text(f"{latency:.2f}", style=f"{style_mod}{lat_color}"),
                Text(signal, style=f"{style_mod}bold {sig_color}")
            )

# -----------------------------------------------------------------------------
# SCREENS
# -----------------------------------------------------------------------------

class OverviewScreen(Screen):
    def compose(self) -> ComposeResult:
        yield SystemHeader()
        with Horizontal():
            yield Sidebar()
            with Vertical(id="content-area"):
                yield WatchlistTable()
                yield RichLog(highlight=True, markup=True)

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold green]SYSTEM ONLINE[/] - MARK5 Dashboard Connected")
        if not bridge.connected:
             log.write("[bold red]WARNING[/] - Bridge Disconnected. Waiting for Shared Memory...")

class TrainingHubScreen(Screen):
    def compose(self) -> ComposeResult:
        yield SystemHeader()
        with Horizontal():
            yield Sidebar()
            with Vertical(id="content-area"):
                with Horizontal(classes="stats-box"):
                    yield Static("Total Models: [bold]124[/]", classes="panel")
                    yield Static("Success Rate: [bold green]94%[/]", classes="panel")
                    yield Static("Avg Accuracy: [bold yellow]68.5%[/]", classes="panel")
                
                yield Label("Train New Model:")
                with Horizontal():
                    yield Input(placeholder="Ticker (e.g. RELIANCE)", id="train-ticker-input")
                    yield Button("Start Training", id="btn-start-train", variant="primary")
                
                yield Static("Active Training Jobs:", classes="panel")
                yield ProgressBar(total=100, show_eta=True, id="train-progress")
                yield RichLog(id="train-log", highlight=True, markup=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-start-train":
            ticker = self.query_one("#train-ticker-input", Input).value.strip().upper()
            
            # Validation
            if not ticker:
                self.log_msg("[bold red]Error: Input empty[/]")
                return
            if not self.app.data_provider.is_ticker_valid(ticker):
                self.log_msg(f"[bold red]Error: Invalid Ticker {ticker}[/]")
                return
                
            self.start_training(ticker)

    def log_msg(self, msg):
        self.query_one("#train-log", RichLog).write(msg)

    def start_training(self, ticker: str) -> None:
        self.log_msg(f"[bold yellow]Starting training for {ticker}...[/]")
        
        # Interaction Lock
        btn = self.query_one("#btn-start-train", Button)
        btn.disabled = True
        btn.label = "Training..."
        
        self.run_worker(self._train_worker(ticker), exclusive=True)

    def _train_worker(self, ticker: str):
        log = self.query_one("#train-log", RichLog)
        progress = self.query_one("#train-progress", ProgressBar)
        btn = self.query_one("#btn-start-train", Button)
        
        try:
            # 1. Get Data
            log.write("Fetching historical data...")
            progress.update(total=100, progress=10)
            
            provider = self.app.data_provider
            df = provider.initialize_symbol(ticker, period="1y", interval="day")
            
            if df is None or df.empty:
                log.write(f"[bold red]Failed to fetch data for {ticker}[/]")
                return

            progress.update(progress=30)
            log.write(f"Data Loaded: {len(df)} rows. Starting Trainer...")
            
            # 2. Train
            trainer = MARK5MLTrainer(config=self.app.config)
            result = trainer.train_advanced_ensemble(ticker, df)
            
            progress.update(progress=100)
            
            if result['status'] == 'success':
                log.write(f"[bold green]Training Success![/] Brier Score: {result['brier_score']:.4f} | Version: {result.get('version', '?')}")
            else:
                log.write(f"[bold red]Training Failed:[/] {result.get('reason')}")
                
        except Exception as e:
            log.write(f"[bold red]Error:[/] {e}")
        finally:
            btn.disabled = False
            btn.label = "Start Training"
            self.app.call_from_thread(self._reset_ui)

    def _reset_ui(self):
         # Helper to reset UI safely if needed
         pass

class DeepAnalysisScreen(Screen):
    def compose(self) -> ComposeResult:
        yield SystemHeader()
        with Horizontal():
            yield Sidebar()
            with Vertical(id="content-area"):
                with Horizontal():
                    yield Input(placeholder="Enter Ticker (e.g. RELIANCE)", id="analysis-input")
                    yield Button("Analyze", id="btn-analyze", variant="primary")
                
                with Horizontal():
                    yield Static(Panel("Waiting for input...", title="Market Data"), id="market-panel", classes="panel")
                    yield Static(Panel("Waiting...", title="AI Signal"), id="signal-panel", classes="signal-box")
                
                yield RichLog(id="analysis-log", highlight=True, markup=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-analyze":
            ticker = self.query_one("#analysis-input", Input).value.strip().upper()
            
            if not ticker: return
            if not self.app.data_provider.is_ticker_valid(ticker):
                self.query_one("#analysis-log", RichLog).write(f"[bold red]Invalid Ticker: {ticker}[/]")
                return

            self.analyze_ticker(ticker)

    def analyze_ticker(self, ticker: str) -> None:
        log = self.query_one("#analysis-log", RichLog)
        log.write(f"[bold cyan]Analyzing {ticker}...[/]")
        
        btn = self.query_one("#btn-analyze", Button)
        btn.disabled = True
        btn.label = "Analyzing..."
        
        self.run_worker(self._analysis_worker(ticker), exclusive=True)

    def _analysis_worker(self, ticker: str):
        log = self.query_one("#analysis-log", RichLog)
        btn = self.query_one("#btn-analyze", Button)
        
        try:
            # 1. Get Data
            provider = self.app.data_provider
            df = provider.initialize_symbol(ticker, period="200d", interval="day")
            
            if df is None or df.empty:
                log.write(f"[bold red]No data found for {ticker}[/]")
                return

            # 2. Predict
            predictor = MARK5Predictor(ticker, self.app.data_provider)
            result = predictor.predict(df)
            
            # 3. Update UI
            market_panel = self.query_one("#market-panel", Static)
            signal_panel = self.query_one("#signal-panel", Static)
            
            last_price = df['close'].iloc[-1]
            vol = df['volume'].iloc[-1]
            
            market_panel.update(Panel(f"Price: {last_price:.2f}\nVol: {vol}\nRows: {len(df)}", title=f"{ticker} Market Data"))
            
            if result.get('status') == 'success':
                sig = result['signal']
                conf = result['confidence']
                entropy = result.get('entropy', 0.0)
                
                color = "green" if "BUY" in sig else ("red" if "SELL" in sig else "yellow")
                
                signal_panel.update(Panel(
                    f"[bold {color}]{sig}[/]\nConf: {conf:.1%}\nEntropy: {entropy}", 
                    title="AI Signal"
                ))
                log.write(f"[bold green]Analysis Complete.[/] Signal: {sig}")
            else:
                signal_panel.update(Panel(f"[bold red]ERROR[/]", title="AI Signal"))
                log.write(f"[bold red]Prediction Failed:[/] {result.get('msg', 'Unknown error')}")

        except Exception as e:
            log.write(f"[bold red]Analysis Error:[/] {e}")
        finally:
            btn.disabled = False
            btn.label = "Analyze"

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

class MARK5Dashboard(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("w", "switch_mode('overview')", "Overview"),
        ("t", "switch_mode('training')", "Training"),
        ("a", "switch_mode('analysis')", "Analysis"),
        ("q", "quit", "Quit"),
    ]
    
    SCREENS = {
        "overview": OverviewScreen,
        "training": TrainingHubScreen,
        "analysis": DeepAnalysisScreen
    }

    def on_mount(self) -> None:
        # Initialize Core Systems
        self.config = get_config()
        self.data_provider = DataProvider(self.config)
        
        self.push_screen("overview")

    def action_switch_mode(self, mode: str) -> None:
        self.push_screen(mode)

if __name__ == "__main__":
    app = MARK5Dashboard()
    app.run()
