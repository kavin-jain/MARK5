#!/usr/bin/env python3
"""
MARK5 KITE TOKEN GENERATOR v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification

TRADING ROLE: Authentication Utility
SAFETY LEVEL: CRITICAL - Access Token Generation

FEATURES:
✅ Zerodha Login Flow Automation
✅ Secure Token Extraction
✅ Environment (.env) Auto-Update
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from kiteconnect import KiteConnect
except ImportError:
    print("❌ kiteconnect not installed!")
    print("   Install it: pip install kiteconnect")
    sys.exit(1)

def generate_access_token():
    """Generate Kite Connect access token"""
    
    # Get credentials from environment
    api_key = os.getenv('KITE_API_KEY') 
    api_secret = os.getenv('KITE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Missing credentials in .env file!")
        print("   Make sure KITE_API_KEY and KITE_API_SECRET are set")
        return
    
    print("="*70)
    print("🔐 ZERODHA KITE CONNECT - ACCESS TOKEN GENERATOR")
    print("="*70)
    print()
    
    # Initialize Kite Connect
    kite = KiteConnect(api_key=api_key)
    
    # Step 1: Get login URL
    print("Step 1: Login to Zerodha")
    print("-" * 70)
    login_url = kite.login_url()
    print(f"📱 Open this URL in your browser:")
    print()
    print(f"   {login_url}")
    print()
    print("After login, you'll be redirected to:")
    print("   http://localhost:8080/callback?request_token=XXXXX&action=login")
    print()
    
    # Step 2: Get request token from user
    print("-" * 70)
    print("Step 2: Copy the request token from URL")
    print("-" * 70)
    print("💡 TIP: You can paste the ENTIRE URL or just the request_token")
    user_input = input("Paste URL or request_token: ").strip()
    
    if not user_input:
        print("❌ No input provided!")
        return
    
    # Extract request_token from URL if full URL was pasted
    request_token = user_input
    if 'request_token=' in user_input:
        try:
            # Parse the URL to extract request_token
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(user_input)
            params = parse_qs(parsed.query)
            request_token = params.get('request_token', [user_input])[0]
            print(f"✅ Extracted request_token: {request_token[:10]}...{request_token[-4:]}")
        except Exception as e:
            print(f"⚠️ Could not parse URL, using input as-is: {e}")
    
    if not request_token:
        print("❌ No request token found!")
        return
    
    # Step 3: Generate access token
    print()
    print("-" * 70)
    print("Step 3: Generating access token...")
    print("-" * 70)
    
    try:
        print()
        print(f"🔑 Using API Secret: {api_secret[:8]}...{api_secret[-4:]}")
        print(f"🔑 Request Token: {request_token[:10]}...{request_token[-4:]}")
        print()
        
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print()
        print("✅ ACCESS TOKEN GENERATED SUCCESSFULLY!")
        print("="*70)
        print()
        print(f"Access Token: {access_token}")
        print()
        print("="*70)
        print()
        
        # Step 4: Update .env file
        print("Step 4: Updating .env file...")
        print("-" * 70)
        
        # Calculate project root (4 levels up from core/utils/tools/generate_kite_token.py)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env_path = os.path.join(project_root, '.env')
        
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update KITE_ACCESS_TOKEN line
        updated = False
        with open(env_path, 'w') as f:
            for line in lines:
                if line.startswith('KITE_ACCESS_TOKEN='):
                    f.write(f'KITE_ACCESS_TOKEN={access_token}\n')
                    updated = True
                else:
                    f.write(line)
        
        if updated:
            print("✅ .env file updated with access token")
        else:
            print("⚠️  Could not update .env file automatically")
            print(f"   Manually add: KITE_ACCESS_TOKEN={access_token}")
        
        print()
        print("="*70)
        print("🎉 SETUP COMPLETE!")
        print("="*70)
        print()
        print("✅ Your MARK5 system can now connect to Zerodha Kite!")
        print()
        print("⚠️  IMPORTANT:")
        print("   - Access tokens expire after 24 hours")
        print("   - Run this script daily to get a fresh token")
        print("   - Or set up automatic token refresh in your code")
        print()
        print("🚀 Start MARK5:")
        print("   python3 core/system/launcher.py")
        print()
        
    except Exception as e:
        print(f"❌ Error generating access token: {e}")
        print()
        print("="*70)
        print("🔍 TROUBLESHOOTING GUIDE")
        print("="*70)
        print()
        print("Common issues:")
        print()
        print("1️⃣ Request token already used")
        print("   → Request tokens are SINGLE-USE only")
        print("   → If you already tried once, it won't work again")
        print("   → Solution: Login again to get a NEW request token")
        print()
        print("2️⃣ Request token expired")
        print("   → Request tokens expire after 5 MINUTES")
        print("   → If you waited too long, it expired")
        print("   → Solution: Login again to get a fresh token")
        print()
        print("3️⃣ Wrong API secret in .env file")
        print(f"   → Check .env file has correct KITE_API_SECRET")
        print(f"   → Current value: {api_secret[:8]}...{api_secret[-4:]}")
        print("   → Solution: Verify this matches your Kite Connect app settings")
        print("      at https://developers.kite.trade/")
        print()
        print("4️⃣ Wrong API key")
        print(f"   → Current KITE_API_KEY: {api_key}")
        print("   → Solution: Verify this matches your Kite Connect app")
        print()
        print("="*70)
        print("🔄 NEXT STEPS:")
        print("="*70)
        print()
        print("1. Go to the login URL again (see above)")
        print("2. Login with your Zerodha credentials")
        print("3. Get a FRESH request_token from the redirect URL")
        print("4. Run this script again")
        print()
        print("If issue persists, verify your .env file:")
        print(f"   cat {os.path.join(os.getcwd(), '.env')}")
        print()

if __name__ == "__main__":
    generate_access_token()
