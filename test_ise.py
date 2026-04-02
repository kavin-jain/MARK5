import sys
import os
sys.path.insert(0, '/home/lynx/Documents/MARK5')
from dotenv import load_dotenv
load_dotenv('/home/lynx/Documents/MARK5/.env')
from core.data.adapters.ise_adapter import ISEAdapter

print("Connecting to ISE...")
ise = ISEAdapter()
print("Fetching RELIANCE...")
res = ise.fetch_stock('RELIANCE')
print(res)
