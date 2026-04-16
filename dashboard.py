#!/usr/bin/env python3
"""
MARK5 CENTRALIZED DASHBOARD v4.0 - INSTITUTIONAL EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unified CLI entry point for training, backtesting, and paper trading.
Optimized for Meta-Labeling and Non-Negative Stacking.
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime, date
from typing import Optional, List, Dict

# --- LOGGING RECONFIGURATION (Institutional Standard) ---
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "system.log")

# Environment vars to silence noise before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.FileHandler(_LOG_FILE, mode='a')],
    force=True
)

_QUIET_LOGGERS = ["tensorflow", "lightgbm", "urllib3", "absl", "matplotlib", "py.warnings"]
for logger_name in _QUIET_LOGGERS:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import json
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import IntPrompt, Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from core.models.training.trainer import MARK5MLTrainer
    from core.models.predictor import MARK5Predictor
    from core.models.tcn.backtester import RobustBacktester
    from core.data.data_pipeline import DataPipeline
    from core.utils.config_manager import get_config
    from core.models.registry import RobustModelRegistry
except ImportError as e:
    console.print(f"[bold red]❌ Critical Import Error: {e}[/bold red]")
    sys.exit(1)

def load_dynamic_universe() -> List[str]:
    """Loads active universe from config/universe.json or returns default."""
    config_path = os.path.join(_PROJECT_ROOT, "config", "universe.json")
    default_list = [
        'COFORGE.NS', 'HAL.NS', 'IDFCFIRSTB.NS', 'RELIANCE.NS', 'HDFCBANK.NS',
        'INFY.NS', 'TCS.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS'
    ]
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                return data.get("active_universe", default_list)
        except Exception as e:
            logging.error(f"Error loading universe.json: {e}")
    return default_list

_DEFAULT_UNIVERSE = load_dynamic_universe()

def show_banner():
    banner = """
    [bold cyan]
    ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗
    ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝
    ██╔████╔██║███████║██████╔╝█████╔╝ ███████╗
    ██║╚██╔╝██║██╔══██║██╔══██╗██║ ██╗ ╚════██║
    ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████║
    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
    [/bold cyan]
    [bold white]Advanced Financial Analytics & Trading System v10.0[/bold white]
    """
    console.print(Panel(banner, border_style="cyan", expand=False))

def ensure_model_ready(ticker: str) -> bool:
    """Verifies model exists, prompts to train if missing."""
    predictor = MARK5Predictor(ticker, allow_shadow=True)
    if predictor._container:
        return True
    
    if Confirm.ask(f"[yellow]No model found for {ticker}. Train now?[/yellow]"):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            progress.add_task(description=f"Training {ticker}...", total=None)
            trainer = MARK5MLTrainer()
            res = trainer.train_model(ticker)
            return res.get("status") == "success"
    return False

def run_backtest(ticker: str, days: int = 60) -> Dict:
    """Core backtest execution logic."""
    try:
        if not ensure_model_ready(ticker):
            return {"error": "No model available"}

        trainer = MARK5MLTrainer()
        # v13.5: Need 500+ bars for deep learning feature warm-up (rolling Z-scores)
        df = trainer.fetch_data_for_training(ticker, years=3.0)
        if df is None or len(df) < days + 250:
            return {"error": "Insufficient data"}

        predictor = MARK5Predictor(ticker, allow_shadow=True)
        buffer_days = 250 # Extended buffer
        test_df = df.tail(days + buffer_days)
        start_idx = len(test_df) - days

        full_signals = [0] * len(test_df)
        for i in range(start_idx, len(test_df)):
            subset = test_df.iloc[:i+1]
            res = predictor.predict(subset)
            sig = res.get('signal', 'HOLD')
            
            # Institutional Logic: Use the model's signal directly. 
            # The predictor already handles regime gating and hurdles.
            full_signals[i] = 1 if 'BUY' in sig else (-1 if 'SELL' in sig else 0)

        signals_series = pd.Series(full_signals, index=test_df.index)
        # v14.0: Standardize on INTRADAY for backtests to evaluate both LONG and SHORT edge.
        backtester = RobustBacktester(segment='EQUITY_INTRADAY')
        _, metrics = backtester.run_simulation(test_df, signals_series, symbol=ticker)
        return metrics
    except Exception as e:
        logging.exception(f"Backtest error for {ticker}")
        return {"error": str(e)}

def backtest_menu():
    while True:
        console.clear()
        show_banner()
        table = Table(title="Backtesting Module", show_header=True, header_style="bold magenta")
        table.add_row("1", "Single Ticker Backtest")
        table.add_row("2", "Run All Universe Backtest")
        table.add_row("3", "Back")
        console.print(table)
        
        choice = IntPrompt.ask("Select", choices=["1", "2", "3"])
        if choice == 3: break

        days = IntPrompt.ask("Enter Days", default=60)
        
        if choice == 1:
            ticker = Prompt.ask("Ticker (e.g. SBIN.NS)").upper()
            with console.status(f"[bold green]Running {ticker}..."):
                metrics = run_backtest(ticker, days)
            
            if "error" in metrics:
                console.print(f"[red]Error: {metrics['error']}[/red]")
            else:
                res_table = Table(title=f"Results: {ticker}")
                for k, v in metrics.items():
                    if k != 'trades': res_table.add_row(k, f"{v:.2f}" if isinstance(v, float) else str(v))
                console.print(res_table)
            input("Press Enter...")

        elif choice == 2:
            results = []
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), transient=True) as progress:
                task = progress.add_task("Backtesting Universe...", total=len(_DEFAULT_UNIVERSE))
                for sym in _DEFAULT_UNIVERSE:
                    progress.update(task, description=f"Backtesting {sym}...")
                    m = run_backtest(sym, days)
                    results.append({"Ticker": sym, "Metrics": m})
                    progress.advance(task)
            
            summary = Table(title=f"Universe Summary ({days} Days)")
            summary.add_column("Ticker")
            summary.add_column("Return %", justify="right")
            summary.add_column("Sharpe", justify="right")
            summary.add_column("Trades", justify="right")
            summary.add_column("Status")

            for r in results:
                m = r["Metrics"]
                if "error" in m:
                    summary.add_row(r["Ticker"], "N/A", "N/A", "N/A", f"[red]{m['error']}[/red]")
                else:
                    ret = m.get('Total Return %', 0)
                    sr = m.get('Sharpe Ratio', 0)
                    cnt = m.get('Total Trades', 0)
                    color = "green" if ret > 0 else "white"
                    summary.add_row(r["Ticker"], f"[{color}]{ret:.2f}%[/{color}]", f"{sr:.2f}", str(cnt), "[green]Success[/green]")
            console.print(summary)
            input("Press Enter...")

def training_menu():
    while True:
        console.clear()
        show_banner()
        table = Table(title="Training Module")
        table.add_row("1", "Train Single")
        table.add_row("2", "Train All Universe")
        table.add_row("3", "Back")
        console.print(table)
        choice = IntPrompt.ask("Select", choices=["1", "2", "3"])
        if choice == 3: break

        trainer = MARK5MLTrainer()
        if choice == 1:
            ticker = Prompt.ask("Ticker").upper()
            with console.status(f"Training {ticker}..."):
                res = trainer.train_model(ticker)
            console.print(f"[bold green]Complete: {res.get('status')}[/bold green]")
            input("Press Enter...")
        elif choice == 2:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn()) as progress:
                task = progress.add_task("Universe Training...", total=len(_DEFAULT_UNIVERSE))
                for sym in _DEFAULT_UNIVERSE:
                    progress.update(task, description=f"Training {sym}...")
                    trainer.train_model(sym)
                    progress.advance(task)
            input("Universe Training Complete. Press Enter...")

def ise_intelligence_menu():
    pipeline = DataPipeline()
    while True:
        console.clear()
        show_banner()
        table = Table(title="ISE Intelligence", show_header=True, header_style="bold magenta")
        table.add_row("1", "Trending Stocks")
        table.add_row("2", "Most Active (NSE)")
        table.add_row("3", "Most Active (BSE)")
        table.add_row("4", "Price Shockers")
        table.add_row("5", "Company Fundamental Scan")
        table.add_row("6", "Back")
        console.print(table)
        
        choice = IntPrompt.ask("Select", choices=["1", "2", "3", "4", "5", "6"])
        if choice == 6: break

        with console.status("[bold green]Fetching ISE Data..."):
            try:
                df = pd.DataFrame()
                title = ""
                if choice == 1:
                    df = pipeline.get_trending_stocks()
                    title = "Trending Stocks"
                elif choice == 2:
                    df = pipeline.get_most_active(exchange='NSE')
                    title = "Most Active (NSE)"
                elif choice == 3:
                    df = pipeline.get_most_active(exchange='BSE')
                    title = "Most Active (BSE)"
                elif choice == 4:
                    df = pipeline.get_price_shockers()
                    title = "Price Shockers"
                elif choice == 5:
                    ticker = Prompt.ask("Enter Ticker (e.g. RELIANCE)").upper()
                    df = pipeline.get_fundamental_data(ticker)
                    title = f"Fundamental Scan: {ticker}"
                
                if df.empty:
                    console.print("[yellow]No data returned from ISE.[/yellow]")
                else:
                    res_table = Table(title=title)
                    # Limit columns for display if too many
                    cols = df.columns[:8] 
                    for col in cols:
                        res_table.add_column(str(col))
                    for _, row in df.head(15).iterrows():
                        res_table.add_row(*[str(row[c]) for c in cols])
                    console.print(res_table)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        input("Press Enter...")

def main():
    parser = argparse.ArgumentParser(description="MARK5 Dashboard")
    parser.add_argument("--backtest", type=str, help="Run backtest for ticker")
    parser.add_argument("--train", type=str, help="Train model for ticker")
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    if args.backtest:
        console.print(f"Running backtest for {args.backtest}...")
        print(run_backtest(args.backtest, args.days))
        return
    if args.train:
        console.print(f"Training {args.train}...")
        print(MARK5MLTrainer().train_model(args.train))
        return

    while True:
        console.clear()
        show_banner()
        main_table = Table(title="Main Menu", header_style="bold magenta")
        main_table.add_row("1", "Training")
        main_table.add_row("2", "Backtesting")
        main_table.add_row("3", "ISE Intelligence")
        main_table.add_row("4", "System Status")
        main_table.add_row("5", "Smart Universe Optimizer")
        main_table.add_row("6", "Exit")
        console.print(main_table)
        
        choice = IntPrompt.ask("Select", choices=["1", "2", "3", "4", "5", "6"])
        
        # Ensure we always use the latest dynamic universe
        global _DEFAULT_UNIVERSE
        _DEFAULT_UNIVERSE = load_dynamic_universe()
        
        if choice == 1: training_menu()
        elif choice == 2: backtest_menu()
        elif choice == 3: ise_intelligence_menu()
        elif choice == 4:
            # Simple Status
            reg = RobustModelRegistry()
            console.print(Panel(f"Active Universe: {len(_DEFAULT_UNIVERSE)} stocks\nModels Tracked: {len(reg.registry)}\nLogs: {_LOG_FILE}", title="System Status"))
            input("Press Enter...")
        elif choice == 5:
            console.print("[bold yellow]Launching Smart Universe Optimizer...[/bold yellow]")
            console.print("This process runs in the background and may take hours depending on the universe size.")
            console.print("You can view progress in `logs/system.log`.")
            
            # Run the optimizer
            try:
                with console.status("Running Smart Universe Optimizer... (This may take a while)", spinner="bouncingBar"):
                    subprocess.run([sys.executable, "-c", "from core.optimization.universe_optimizer import UniverseOptimizer; UniverseOptimizer().optimize_universe()"], check=True)
                console.print("[bold green]✅ Smart Universe Optimizer completed successfully![/bold green]")
                _DEFAULT_UNIVERSE = load_dynamic_universe() # reload
            except Exception as e:
                console.print(f"[bold red]❌ Error running optimizer: {e}[/bold red]")
            input("Press Enter...")
        elif choice == 6:
            sys.exit(0)

if __name__ == "__main__":
    main()
