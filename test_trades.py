from core.models.backtester import RobustBacktester
import pandas as pd
import yfinance as yf

# Get some data
df = yf.download("HDFCBANK.NS", start="2024-01-01", end="2025-01-01")
df.columns = [c[0].lower() for c in df.columns]

# Dummy signal (all 1s)
signals = pd.Series(1, index=df.index)

bt = RobustBacktester(segment="EQUITY_DELIVERY")
eq, metrics = bt.run_simulation(df, signals)

trades = metrics['trades']
if trades:
    print(pd.DataFrame([vars(t) for t in trades[:5]]))
