import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MARK5.InstantReport")

from core.models.training.financial_engineer import FinancialEngineer
from core.models.tcn.backtester import RobustBacktester

def simulate_ticker(ticker):
    try:
        base_sym = ticker.replace('.NS', '')
        spot_path = f"data/cache/{base_sym}_NS_1d.parquet"
        if not os.path.exists(spot_path):
            spot_path = f"data/cache/{base_sym.replace('.', '_')}_NS_1d.parquet"
            
        if not os.path.exists(spot_path):
            return {"ticker": ticker, "error": "No cached data found"}
            
        df = pd.read_parquet(spot_path)
        if df.empty:
            return {"ticker": ticker, "error": "Empty data"}

        # Use 2-year OOS
        oos_start = pd.Timestamp("2024-04-01")
        if df.index.max() < oos_start:
             return {"ticker": ticker, "error": "Insufficient data"}
             
        test_df = df[df.index >= oos_start - pd.Timedelta(days=100)].copy()
        
        fe = FinancialEngineer()
        signals = fe.get_primary_signals(test_df)
        
        # Zero out pre-OOS signals
        signals.loc[signals.index < oos_start] = 0
        test_df = test_df[test_df.index >= oos_start]
        signals = signals[signals.index >= oos_start]
        
        bt = RobustBacktester(segment='FUTURES')
        _, metrics = bt.run_simulation(test_df, signals, symbol=ticker)
        
        metrics["ticker"] = ticker
        metrics["start_date"] = str(test_df.index[0].date())
        metrics["end_date"] = str(test_df.index[-1].date())
        return metrics
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

def run_instant_report():
    import glob
    cached_files = glob.glob("data/cache/*_1d.parquet")
    candidates = []
    for f in cached_files:
        filename = os.path.basename(f)
        if filename.startswith("NIFTY"): continue
        parts = filename.split("_")
        if len(parts) >= 3:
            ticker = f"{parts[0]}.NS"
            candidates.append(ticker)
    
    candidates = sorted(list(set(candidates)))[:150]
    logger.info(f"Loaded {len(candidates)} candidates.")

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(simulate_ticker, ticker): ticker for ticker in candidates}
        for i, future in enumerate(as_completed(future_to_ticker), 1):
            try:
                res = future.result()
                results.append(res)
                if "error" in res:
                    logger.warning(f"[{i}/{len(candidates)}] {res['ticker']} FAILED: {res['error']}")
                else:
                    logger.info(f"[{i}/{len(candidates)}] {res['ticker']} COMPLETE: Ret={res.get('Total Return %', 0):.2f}%")
            except Exception as e:
                pass

    return results

def generate_report(results: list, output_path: str):
    valid_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    df = pd.DataFrame(valid_results)
    avg_return = df.get("Total Return %", pd.Series([0])).mean()
    avg_sharpe = df.get("Sharpe Ratio", pd.Series([0])).mean()
    avg_win_rate = df.get("Win Rate %", pd.Series([0])).mean()
    avg_drawdown = df.get("Max Drawdown %", pd.Series([0])).mean()
    avg_trades = df.get("Total Trades", pd.Series([0])).mean()

    # Dynamic Needs
    needs_met = []
    if avg_return >= 15.0: needs_met.append("✅ Average Return >= 15% target")
    else: needs_met.append(f"❌ Average Return < 15% target (Actual: {avg_return:.2f}%)")
    if avg_sharpe > 0.5: needs_met.append("✅ Average Sharpe Ratio > 0.5 target")
    else: needs_met.append(f"❌ Average Sharpe Ratio <= 0.5 target (Actual: {avg_sharpe:.2f})")
    if avg_win_rate > 44.0: needs_met.append("✅ Average Win Rate > 44% target")
    else: needs_met.append(f"❌ Average Win Rate <= 44% target (Actual: {avg_win_rate:.2f}%)")

    md = [
        "# MARK5 - 150 Stock Systematic Backtest Report (FUTURES MIGRATION)",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Validation Type:** 2-Year Strict Out-of-Sample (OOS)",
        f"**Training Cutoff:** 2024-03-31",
        f"**Backtest Horizon:** 2024-04-01 to Present",
        f"**Processed:** {len(results)} / 150",
        f"**Passed Gate:** {len(valid_results)}",
        f"**Failed Gate/Data:** {len(errors)}",
        "",
        "## System Averages vs Needs",
        f"- **Average Total Return:** {avg_return:.2f}%",
        f"- **Average Sharpe Ratio:** {avg_sharpe:.2f}",
        f"- **Average Win Rate:** {avg_win_rate:.2f}%",
        f"- **Average Max Drawdown:** {avg_drawdown:.2f}%",
        f"- **Average Trades per Stock:** {avg_trades:.1f}",
        "",
        "### Requirements Check",
    ] + [f"- {n}" for n in needs_met] + [
        "",
        "## Individual Stock Results (Top 20 by Return)",
        "| Ticker | Return % | Sharpe | Win Rate % | Max Drawdown % | Trades |",
        "|--------|----------|--------|------------|----------------|--------|"
    ]

    if valid_results:
        df_sorted = df.sort_values(by="Total Return %", ascending=False)
        for _, row in df_sorted.head(20).iterrows():
            t = row.get("ticker", "N/A")
            ret = row.get("Total Return %", 0)
            sh = row.get("Sharpe Ratio", 0)
            wr = row.get("Win Rate %", 0)
            dd = row.get("Max Drawdown %", 0)
            tr = row.get("Total Trades", 0)
            md.append(f"| {t} | {ret:.2f}% | {sh:.2f} | {wr:.2f}% | {dd:.2f}% | {tr} |")

    if errors:
        md.append("\n## Gate Failures & Errors")
        md.append("| Ticker | Error Message |")
        md.append("|--------|---------------|")
        for e in errors[-10:]:
            md.append(f"| {e['ticker']} | {e['error']} |")
        if len(errors) > 10:
            md.append(f"| ... | and {len(errors)-10} more. |")

    report_content = "\n".join(md)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)
    print(report_content)

if __name__ == "__main__":
    results = run_instant_report()
    generate_report(results, "reports/backtest_150_report.md")
