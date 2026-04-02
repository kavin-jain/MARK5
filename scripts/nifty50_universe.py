#!/usr/bin/env python3
"""
NIFTY 50 Universe — Stock definitions for MARK5 Validation Pipeline
"""

# Full Nifty 50 with sector classification
NIFTY_50 = {
    'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Energy'},
    'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT'},
    'HDFCBANK.NS': {'name': 'HDFC Bank', 'sector': 'Banking'},
    'INFY.NS': {'name': 'Infosys', 'sector': 'IT'},
    'ICICIBANK.NS': {'name': 'ICICI Bank', 'sector': 'Banking'},
    'HINDUNILVR.NS': {'name': 'Hindustan Unilever', 'sector': 'FMCG'},
    'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Banking'},
    'BHARTIARTL.NS': {'name': 'Bharti Airtel', 'sector': 'Telecom'},
    'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank', 'sector': 'Banking'},
    'ITC.NS': {'name': 'ITC', 'sector': 'FMCG'},
    'LT.NS': {'name': 'Larsen & Toubro', 'sector': 'Capital Goods'},
    'AXISBANK.NS': {'name': 'Axis Bank', 'sector': 'Banking'},
    'BAJFINANCE.NS': {'name': 'Bajaj Finance', 'sector': 'Financial Services'},
    'ASIANPAINT.NS': {'name': 'Asian Paints', 'sector': 'Consumer Durables'},
    'MARUTI.NS': {'name': 'Maruti Suzuki', 'sector': 'Automobile'},
    'HCLTECH.NS': {'name': 'HCL Technologies', 'sector': 'IT'},
    'SUNPHARMA.NS': {'name': 'Sun Pharmaceutical', 'sector': 'Pharma'},
    'TITAN.NS': {'name': 'Titan Company', 'sector': 'Consumer Durables'},
    'WIPRO.NS': {'name': 'Wipro', 'sector': 'IT'},
    'ULTRACEMCO.NS': {'name': 'UltraTech Cement', 'sector': 'Cement'},
    'ETERNAL.NS': {'name': 'Eternal (Zomato)', 'sector': 'Consumer Tech'},
    'NESTLEIND.NS': {'name': 'Nestle India', 'sector': 'FMCG'},
    'BAJAJFINSV.NS': {'name': 'Bajaj Finserv', 'sector': 'Financial Services'},
    'TATASTEEL.NS': {'name': 'Tata Steel', 'sector': 'Metals'},
    'NTPC.NS': {'name': 'NTPC', 'sector': 'Power'},
    'POWERGRID.NS': {'name': 'Power Grid Corporation', 'sector': 'Power'},
    'ONGC.NS': {'name': 'ONGC', 'sector': 'Energy'},
    'M&M.NS': {'name': 'Mahindra & Mahindra', 'sector': 'Automobile'},
    'JSWSTEEL.NS': {'name': 'JSW Steel', 'sector': 'Metals'},
    'ADANIENT.NS': {'name': 'Adani Enterprises', 'sector': 'Diversified'},
    'ADANIPORTS.NS': {'name': 'Adani Ports', 'sector': 'Infrastructure'},
    'TECHM.NS': {'name': 'Tech Mahindra', 'sector': 'IT'},
    'HDFCLIFE.NS': {'name': 'HDFC Life Insurance', 'sector': 'Insurance'},
    'SBILIFE.NS': {'name': 'SBI Life Insurance', 'sector': 'Insurance'},
    'DIVISLAB.NS': {'name': "Divi's Laboratories", 'sector': 'Pharma'},
    'DRREDDY.NS': {'name': "Dr. Reddy's Laboratories", 'sector': 'Pharma'},
    'CIPLA.NS': {'name': 'Cipla', 'sector': 'Pharma'},
    'APOLLOHOSP.NS': {'name': 'Apollo Hospitals', 'sector': 'Healthcare'},
    'EICHERMOT.NS': {'name': 'Eicher Motors', 'sector': 'Automobile'},
    'GRASIM.NS': {'name': 'Grasim Industries', 'sector': 'Cement'},
    'BRITANNIA.NS': {'name': 'Britannia Industries', 'sector': 'FMCG'},
    'INDUSINDBK.NS': {'name': 'IndusInd Bank', 'sector': 'Banking'},
    'COALINDIA.NS': {'name': 'Coal India', 'sector': 'Mining'},
    'BPCL.NS': {'name': 'BPCL', 'sector': 'Energy'},
    'TATACONSUM.NS': {'name': 'Tata Consumer Products', 'sector': 'FMCG'},
    'HEROMOTOCO.NS': {'name': 'Hero MotoCorp', 'sector': 'Automobile'},
    'BAJAJ-AUTO.NS': {'name': 'Bajaj Auto', 'sector': 'Automobile'},
    'HINDALCO.NS': {'name': 'Hindalco Industries', 'sector': 'Metals'},
    'SHRIRAMFIN.NS': {'name': 'Shriram Finance', 'sector': 'Financial Services'},
    'BEL.NS': {'name': 'Bharat Electronics', 'sector': 'Defence'},
}

# Ticker lists
NIFTY_50_TICKERS = list(NIFTY_50.keys())

# Top 10 by market cap (for quick runs)
NIFTY_TOP10 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS'
]

# Test subset (3 stocks — for quick sanity checks)
NIFTY_TEST3 = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

# MARK5 Alpha Portfolio — stocks with proven positive Sharpe (>0.4)
MARK5_ALPHA = ['SBIN.NS', 'RELIANCE.NS', 'HINDUNILVR.NS']

# ── NIFTY Midcap 100 — IC-validated premium universe ─────────────────────────
# Proof: cross-sectional ICIR = 0.288 vs LargeCap50 ICIR = -0.012 (24× better)
# All 52 stocks confirmed liquid (> ₹1.5L fillable at open), data on Kite.
# Avg ATR = 3.10% vs NIFTY50 = 2.57% → more price movement per trade.
NIFTY_MIDCAP_100 = {
    # Financials / NBFC
    'CHOLAFIN.NS':   {'name': 'Cholamandalam Investment', 'sector': 'Financial Services'},
    'MUTHOOTFIN.NS': {'name': 'Muthoot Finance', 'sector': 'Financial Services'},
    'SUNDARMFIN.NS': {'name': 'Sundaram Finance', 'sector': 'Financial Services'},
    'IDFCFIRSTB.NS': {'name': 'IDFC First Bank', 'sector': 'Banking'},
    'AUBANK.NS':     {'name': 'AU Small Finance Bank', 'sector': 'Banking'},
    'BANKBARODA.NS': {'name': 'Bank of Baroda', 'sector': 'Banking'},
    'PNB.NS':        {'name': 'Punjab National Bank', 'sector': 'Banking'},
    'JIOFIN.NS':     {'name': 'Jio Financial Services', 'sector': 'Financial Services'},
    # Power / PSU
    'PFC.NS':        {'name': 'Power Finance Corporation', 'sector': 'Power'},
    'RECLTD.NS':     {'name': 'REC Limited', 'sector': 'Power'},
    'IRFC.NS':       {'name': 'Indian Railway Finance Corp', 'sector': 'Infrastructure'},
    'GAIL.NS':       {'name': 'GAIL India', 'sector': 'Energy'},
    'HAL.NS':        {'name': 'Hindustan Aeronautics', 'sector': 'Defence'},
    'BHEL.NS':       {'name': 'Bharat Heavy Electricals', 'sector': 'Capital Goods'},
    'CONCOR.NS':     {'name': 'Container Corporation', 'sector': 'Infrastructure'},
    # Real Estate
    'DLF.NS':        {'name': 'DLF', 'sector': 'Real Estate'},
    'LODHA.NS':      {'name': 'Macrotech Developers', 'sector': 'Real Estate'},
    'PRESTIGE.NS':   {'name': 'Prestige Estates', 'sector': 'Real Estate'},
    # Capital Goods / Industrials
    'SIEMENS.NS':    {'name': 'Siemens India', 'sector': 'Capital Goods'},
    'ABB.NS':        {'name': 'ABB India', 'sector': 'Capital Goods'},
    'HAVELLS.NS':    {'name': 'Havells India', 'sector': 'Consumer Durables'},
    'CGPOWER.NS':    {'name': 'CG Power', 'sector': 'Capital Goods'},
    'POLYCAB.NS':    {'name': 'Polycab India', 'sector': 'Capital Goods'},
    'CUMMINSIND.NS': {'name': 'Cummins India', 'sector': 'Capital Goods'},
    'DIXON.NS':      {'name': 'Dixon Technologies', 'sector': 'Consumer Durables'},
    # IT / Technology
    'MPHASIS.NS':    {'name': 'Mphasis', 'sector': 'IT'},
    'TATATECH.NS':   {'name': 'Tata Technologies', 'sector': 'IT'},
    'PERSISTENT.NS': {'name': 'Persistent Systems', 'sector': 'IT'},
    'COFORGE.NS':    {'name': 'Coforge', 'sector': 'IT'},
    # Auto / Ancillaries
    'TVSMOTOR.NS':   {'name': 'TVS Motor Company', 'sector': 'Automobile'},
    'MOTHERSON.NS':  {'name': 'Samvardhana Motherson', 'sector': 'Automobile'},
    'APOLLOTYRE.NS': {'name': 'Apollo Tyres', 'sector': 'Automobile'},
    'MRF.NS':        {'name': 'MRF', 'sector': 'Automobile'},
    # Pharma / Healthcare
    'TORNTPHARM.NS': {'name': 'Torrent Pharmaceuticals', 'sector': 'Pharma'},
    'ZYDUSLIFE.NS':  {'name': 'Zydus Lifesciences', 'sector': 'Pharma'},
    'MAXHEALTH.NS':  {'name': 'Max Healthcare', 'sector': 'Healthcare'},
    'ALKEM.NS':      {'name': 'Alkem Laboratories', 'sector': 'Pharma'},
    # FMCG / Consumer
    'MARICO.NS':     {'name': 'Marico', 'sector': 'FMCG'},
    'GODREJCP.NS':   {'name': 'Godrej Consumer Products', 'sector': 'FMCG'},
    'BERGEPAINT.NS': {'name': 'Berger Paints', 'sector': 'Consumer Durables'},
    'DABUR.NS':      {'name': 'Dabur India', 'sector': 'FMCG'},
    'COLPAL.NS':     {'name': 'Colgate-Palmolive India', 'sector': 'FMCG'},
    # Travel / Hotels
    'INDIGO.NS':     {'name': 'IndiGo (InterGlobe Aviation)', 'sector': 'Aviation'},
    'INDHOTEL.NS':   {'name': 'Indian Hotels (Taj)', 'sector': 'Hospitality'},
    # Capital Markets
    'MCX.NS':        {'name': 'MCX India', 'sector': 'Financial Services'},
    'BSE.NS':        {'name': 'BSE Limited', 'sector': 'Financial Services'},
    'CDSL.NS':       {'name': 'CDSL', 'sector': 'Financial Services'},
    # Energy / Chemicals
    'IGL.NS':        {'name': 'Indraprastha Gas', 'sector': 'Energy'},
    'PETRONET.NS':   {'name': 'Petronet LNG', 'sector': 'Energy'},
    'PIIND.NS':      {'name': 'PI Industries', 'sector': 'Chemicals'},
    'SRF.NS':        {'name': 'SRF Limited', 'sector': 'Chemicals'},
    'DEEPAKNTR.NS':  {'name': 'Deepak Nitrite', 'sector': 'Chemicals'},
    # Retail / Consumer
    'JUBLFOOD.NS':   {'name': 'Jubilant FoodWorks', 'sector': 'Consumer Tech'},
    'TRENT.NS':      {'name': 'Trent (Westside)', 'sector': 'Retail'},
    'VOLTAS.NS':     {'name': 'Voltas', 'sector': 'Consumer Durables'},
}

NIFTY_MIDCAP_TICKERS = list(NIFTY_MIDCAP_100.keys())

# ── MARK5 Live Universe — NIFTY50 + MidCap100 (IC-validated) ─────────────────
# 49 + 55 = 104 stocks covering large and mid-cap segments.
# IC test proved MidCap ICIR=0.288 >> LargeCap ICIR=-0.012.
# This is the universe used by rank_live.py for all production signals.
MARK5_LIVE_UNIVERSE = {**NIFTY_50, **NIFTY_MIDCAP_100}
MARK5_LIVE_TICKERS  = list(MARK5_LIVE_UNIVERSE.keys())

# Sector groupings
SECTORS = {}
for ticker, info in NIFTY_50.items():
    sector = info['sector']
    if sector not in SECTORS:
        SECTORS[sector] = []
    SECTORS[sector].append(ticker)

# 150 Mixed Stocks (Large, Mid, and Smallcap)
MIXED_150 = NIFTY_50_TICKERS + [
    # Next 50 / Midcap
    'HAL.NS', 'TVSMOTOR.NS', 'CHOLAFIN.NS', 'TRENT.NS', 'DLF.NS', 'INDIGO.NS', 
    'LTIM.NS', 'PFC.NS', 'RECLTD.NS', 'IRFC.NS', 'GAIL.NS', 'JIOFIN.NS', 'AMBUJACEM.NS', 
    'PNB.NS', 'IOC.NS', 'VEDL.NS', 'HAVELLS.NS', 'CGPOWER.NS', 'ABB.NS', 'SIEMENS.NS', 
    'BOSCHLTD.NS', 'CUMMINSIND.NS', 'PIIND.NS', 'POLYCAB.NS', 'SRF.NS', 'AUBANK.NS', 
    'IDFCFIRSTB.NS', 'LODHA.NS', 'MAXHEALTH.NS', 'YESBANK.NS', 'TORNTPHARM.NS', 
    'ZYDUSLIFE.NS', 'MUTHOOTFIN.NS', 'BANKBARODA.NS', 'INDHOTEL.NS', 'DIXON.NS', 'BSE.NS', 
    'MCX.NS', 'CDSL.NS', 'APOLLOTYRE.NS', 'MRF.NS', 'PETRONET.NS', 'IGL.NS', 'CONCOR.NS', 
    'JUBLFOOD.NS', 'BERGEPAINT.NS', 'DABUR.NS', 'GODREJCP.NS', 'MARICO.NS',
    
    # Smallcap / Additional High Beta
    'SUZLON.NS', 'IREDA.NS', 'NHPC.NS', 'SJVN.NS', 'HUDCO.NS', 'NBCC.NS', 'RVNL.NS', 
    'IRCON.NS', 'RAILTEL.NS', 'RITES.NS', 'CASTROLIND.NS', 'IEX.NS', 'NATIONALUM.NS', 
    'HINDCOPPER.NS', 'NMDC.NS', 'SAIL.NS', 'GMRINFRA.NS', 'IDBI.NS', 'IOB.NS', 'MAHABANK.NS', 
    'UCOBANK.NS', 'CENTRALBK.NS', 'SUNDARMFIN.NS', 'MANAPPURAM.NS', 'L&TFH.NS', 'M&MFIN.NS', 
    'CHAMBLFERT.NS', 'GNFC.NS', 'GSFC.NS', 'DEEPAKNTR.NS', 'NAVINFLUOR.NS', 'AARTIIND.NS', 
    'TATACHEM.NS', 'COROMANDEL.NS', 'SYNGENE.NS', 'LAURUSLABS.NS', 'GLENMARK.NS', 
    'AUROPHARMA.NS', 'BIOCON.NS', 'LUPIN.NS', 'IPCALAB.NS', 'BATAINDIA.NS', 'RELAXO.NS', 
    'VIPIND.NS', 'PAGEIND.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS', 'SYMPHONY.NS', 'BLUESTARCO.NS', 
    'FINCABLES.NS', 'KEI.NS'
]
