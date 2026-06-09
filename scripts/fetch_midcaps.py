"""Fetch a liquid NSE midcap universe (split/div-adjusted) to data/cache/ for the
factor-tilt backtest. Read-only public data. Saves {TICKER}_daily.parquet."""
import os, sys, time
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_ROOT, "data", "cache")

# Liquid NIFTY Midcap 100/150 names (current constituents — survivorship caveat applies;
# mitigated by measuring tilt-excess vs same-universe buy-and-hold).
MIDCAPS = [
    "FEDERALBNK","BANKBARODA","CANBK","INDIANB","IDFCFIRSTB","AUBANK","BANKINDIA","UNIONBANK",
    "CHOLAFIN","MUTHOOTFIN","MANAPPURAM","LICHSGFIN","PNBHOUSING","SHRIRAMFIN","M&MFIN",
    "MPHASIS","LTTS","OFSS","COFORGE","PERSISTENT","KPITTECH","CYIENT","SONACOMS",
    "ABB","SIEMENS","BHEL","CUMMINSIND","BHARATFORG","ASHOKLEY","TVSMOTOR","BALKRISIND",
    "MRF","APOLLOTYRE","ESCORTS","EXIDEIND","BOSCHLTD","ENDURANCE","MOTHERSON",
    "GODREJPROP","OBEROIRLTY","DLF","PHOENIXLTD","PRESTIGE","BRIGADE",
    "AUROPHARMA","BIOCON","ALKEM","TORNTPHARM","GLENMARK","IPCALAB","LAURUSLABS","ZYDUSLIFE","ABBOTINDIA",
    "ABCAPITAL","ABFRL","PAGEIND","JUBLFOOD","VBL","COLPAL","MARICO","DABUR","GODREJCP","EMAMILTD","BERGEPAINT",
    "PIIND","SRF","DEEPAKNTR","AARTIIND","NAVINFLUOR","ATUL","TATACHEM","GNFC","GSFC","COROMANDEL","UPL",
    "ACC","AMBUJACEM","DALBHARAT","RAMCOCEM","JKCEMENT",
    "JINDALSTEL","SAIL","NMDC","HINDZINC","NATIONALUM","APLAPOLLO",
    "TATAPOWER","NHPC","TORNTPOWER","CESC","IGL","MGL","GUJGASLTD","PETRONET","GAIL","OIL",
    "IRCTC","CONCOR","INDIGO","INDHOTEL","CROMPTON","HAVELLS","DIXON","POLYCAB","ASTRAL","SUPREMEIND",
]


def main():
    os.makedirs(CACHE, exist_ok=True)
    ok, fail = [], []
    for i, t in enumerate(MIDCAPS):
        out = os.path.join(CACHE, f"{t}_daily.parquet")
        if os.path.exists(out):
            ok.append(t); continue
        try:
            df = yf.download(f"{t}.NS", start="2015-01-01", end="2026-05-22",
                             auto_adjust=True, progress=False)
            if df is None or len(df) < 200:
                fail.append(t); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)
            df.index.name = "date"
            df.reset_index().to_parquet(out)
            ok.append(t)
        except Exception as e:
            fail.append(t)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(MIDCAPS)} done", flush=True)
        time.sleep(0.2)
    print(f"\nFetched/cached: {len(ok)}  |  failed: {len(fail)}")
    if fail:
        print(f"  failed: {fail}")
    # report coverage
    import re
    good16 = good18 = 0
    for t in ok:
        try:
            df = pd.read_parquet(os.path.join(CACHE, f"{t}_daily.parquet"))
            idx = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
            if idx.min() <= pd.Timestamp("2016-01-15"): good16 += 1
            if idx.min() <= pd.Timestamp("2018-01-15"): good18 += 1
        except: pass
    print(f"Coverage: {good16} names from 2016, {good18} names from 2018")
    print(",".join(ok))


if __name__ == "__main__":
    main()
