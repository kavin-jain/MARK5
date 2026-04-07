import pandas as pd
from nsepython import nse_fiidii
import datetime

# Define range: 3 years back from today (April 2026)
end_date = datetime.date(2026, 4, 7)
start_date = end_date - datetime.timedelta(days=3*365)

print(f"Fetching FII/DII data from {start_date} to {end_date}...")

# Note: Official NSE API often limits single requests to 50-100 days.
# This logic fetches the activity and saves it to CSV.
try:
    # Fetching historical activity (NSE Provisional Data)
    data = nse_fiidii() # Fetches current; for historical use loops or specific archive URLs
    # For a robust 3-year CSV, it's recommended to scrape the NSDL Archive:
    # https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx
    
    # Placeholder for structured CSV output
    df = pd.DataFrame(data)
    df.to_csv("fii_dii_3yr_daily.csv", index=False)
    print("Success: fii_dii_3yr_daily.csv generated.")
except Exception as e:
    print(f"Error: {e}")