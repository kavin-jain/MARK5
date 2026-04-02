import pandas as pd
import os
from datetime import datetime

def generate_report():
    csv_path = 'reports/max/validation_2026/validation_summary.csv'
    out_path = 'reports/max/validation_2026/detailed_stock_report.md'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Sort by Sharpe Ratio (Trades Only) descending
    if 'sharpe_trades_only' in df.columns:
        df = df.sort_values(by='sharpe_trades_only', ascending=False)
        
    with open(out_path, 'w') as f:
        f.write("# MARK5 Detailed Stock Validation Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Stocks Analyzed:** {len(df)}\n\n")
        
        f.write("---\n\n")
        f.write("## 🏆 Top Performers (By Trade Sharpe)\n\n")
        
        # Write a summary table of the top 20
        top_20 = df.head(20)
        f.write("| Symbol | Win Rate | Strategy Return | Market Return | Excess | Max DD | Trade Sharpe |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        def pct(val):
            return f"{val*100:.2f}%" if pd.notnull(val) else "N/A"
            
        def dec(val):
            return f"{val:.2f}" if pd.notnull(val) else "N/A"
            
        for _, row in top_20.iterrows():
            f.write(f"| {row['Symbol']} | {pct(row.get('win_rate'))} | {pct(row.get('total_return_strategy'))} | "
                    f"{pct(row.get('total_return_market'))} | {pct(row.get('excess_return'))} | "
                    f"{pct(row.get('max_drawdown'))} | {dec(row.get('sharpe_trades_only'))} |\n")
                    
        f.write("\n---\n\n")
        f.write("## 📊 Detailed Individual Stock Metrics\n\n")
        
        for _, row in df.iterrows():
            f.write(f"### {row['Symbol']}\n")
            f.write(f"- **Win Rate:** {pct(row.get('win_rate'))}\n")
            f.write(f"- **Strategy Total Return:** {pct(row.get('total_return_strategy'))}\n")
            f.write(f"- **Market Total Return:** {pct(row.get('total_return_market'))}\n")
            f.write(f"- **Alpha (Excess Return):** {pct(row.get('excess_return'))}\n")
            f.write(f"- **Max Drawdown:** {pct(row.get('max_drawdown'))}\n")
            f.write(f"- **Trade Sharpe Ratio:** {dec(row.get('sharpe_trades_only'))}\n")
            f.write(f"- **Overall Sharpe Ratio:** {dec(row.get('sharpe_ratio'))}\n")
            f.write(f"- **Sortino Ratio:** {dec(row.get('sortino_ratio'))}\n")
            f.write(f"- **Calmar Ratio:** {dec(row.get('calmar_ratio'))}\n")
            f.write(f"- **Brier Score (Loss):** {dec(row.get('brier_score'))}\n")
            f.write(f"- **Capital Utilization:** {pct(row.get('capital_utilization'))}\n")
            f.write(f"- **Risk Alert Count:** {int(row.get('risk_alerts_count', 0))}\n\n")
            
    print(f"✅ Text report generated at: {out_path}")

if __name__ == "__main__":
    generate_report()
