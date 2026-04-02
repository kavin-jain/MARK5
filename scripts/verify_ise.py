"""
ISE INTEGRATION VERIFIER
━━━━━━━━━━━━━━━━━━━━━━━━

Runs a live end-to-end test of the ISE Enrichment layer.
Checks:
  - Token budget management
  - Dual-endpoint fetching (/stock + /stock_target_price)
  - Rule 25 Veto logic
  - Fundamental modifier calculation (analyst gap, etc.)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from core.data.adapters import ISESignalEnricher
import json

def run_verification():
    print("MARK5 ISE INTEGRATION VERIFIER")
    print("="*40)
    
    # 1. Setup
    enricher = ISESignalEnricher()
    
    # 2. Test Candidates
    # We use some real candidates to see if they pass gates
    # RELIANCE (known), TCS (known)
    test_candidates = [
        ("RELIANCE.NS", 0.62),
        ("TCS.NS", 0.58)
    ]
    
    print(f"Testing Enrichment for: {[c[0] for c in test_candidates]}")
    
    # 3. Run Enrichment
    tradeable, vetoed, budget = enricher.enrich(test_candidates)
    
    # 4. Report
    print("\n" + "="*40)
    print("TRADEABLE SIGNALS (Ordered by Adjusted Confidence)")
    print("="*40)
    if not tradeable:
        print("  None")
    for sig in tradeable:
        mod = sig.ise_breakdown.get('modifier', 0)
        print(f"🟢 {sig.ticker}")
        print(f"   Base Conf: {sig.base_confidence:.2%}")
        print(f"   Adj  Conf: {sig.adjusted_confidence:.2%} (Mod: {mod:+.2f})")
        print(f"   Breakdown: {json.dumps(sig.ise_breakdown, indent=2)}")
        print("-" * 20)
        
    print("\n" + "="*40)
    print("VETOED / DROPPED SIGNALS")
    print("="*40)
    if not vetoed:
        print("  None")
    for sig in vetoed:
        print(f"🔴 {sig.ticker}")
        if not sig.is_safe:
            print(f"   REASON: {sig.veto_reason}")
        else:
            print(f"   REASON: Confidence dropped to {sig.adjusted_confidence:.2%}")
        print("-" * 20)

    print("\n" + "="*40)
    print("BUDGET STATUS")
    print("="*40)
    print(f"  Used this month: {budget['used']}")
    print(f"  Remaining:       {budget['remaining']}")
    print(f"  Limit:           {budget['limit']}")
    print("="*40)

if __name__ == "__main__":
    run_verification()
