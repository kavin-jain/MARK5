import pandas as pd
import os

TICKERS = ["IRFC", "IOB", "M&M", "CONCOR", "HCLTECH", "ABB", "BAJAJ-AUTO", "PIIND"]
oos = pd.Timestamp("2024-04-01")

avg = 0
count = 0
for t in TICKERS:
    path = f"data/cache/{t}_NS_1d.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df = df[df.index >= oos]
        if len(df) > 0:
            ret = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            print(f"{t}: {ret:.2f}%")
            avg += ret
            count += 1
if count > 0:
    print(f"AVG: {avg/count:.2f}%")
