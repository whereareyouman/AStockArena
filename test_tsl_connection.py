#!/usr/bin/env python3
"""Quick test script to verify TinySoft credentials and connection."""
import os
from datetime import datetime, timedelta

def test_tsl_connection():
    print("=== TinySoft Connection Test ===\n")
    
    # 1. Check environment variables
    username = os.getenv("TSL_USERNAME") or os.getenv("TSL_USER")
    password = os.getenv("TSL_PASSWORD") or os.getenv("TSL_PASS")
    server = os.getenv("TSL_SERVER", "tsl.tinysoft.com.cn")
    port = os.getenv("TSL_PORT", "443")
    
    print(f"TSL_USERNAME: {'✓ set' if username else '✗ missing'}")
    print(f"TSL_PASSWORD: {'✓ set' if password else '✗ missing'}")
    print(f"TSL_SERVER: {server}")
    print(f"TSL_PORT: {port}\n")
    
    if not username or not password:
        print("❌ Missing TinySoft credentials. Please set:")
        print("   export TSL_USERNAME='your_username'")
        print("   export TSL_PASSWORD='your_password'")
        return False
    
    # 2. Try importing pyTSL
    try:
        import pyTSL as ts
        print("✓ pyTSL module imported successfully\n")
    except ImportError as e:
        print(f"❌ Cannot import pyTSL: {e}")
        print("   Install via: conda install tspytsl")
        return False
    
    # 3. Try connecting and logging in
    try:
        print(f"Connecting to {server}:{port}...")
        client = ts.Client(username, password, server, int(port))
        
        result = client.login()
        if result != 1:
            last_err = getattr(client, "last_error", lambda: "Unknown error")()
            print(f"❌ Login failed: {last_err}")
            if "Relogin refused" in str(last_err):
                print("\n   Tip: Another session may be active. Try running:")
                print("   python scripts/tsl_logout.py")
            return False
        
        print("✓ Login successful!\n")
        
        # 4. Try a simple query (SH688008 - 澜起科技)
        print("Testing query for SH688008 (澜起科技)...")
        end_time = datetime.now()
        begin_time = end_time - timedelta(days=5)
        
        r = client.query(
            stock='SH688008',
            begin_time=begin_time,
            end_time=end_time,
            cycle='日线',
            fields='date, close, vol'
        )
        
        if r.error() != 0:
            print(f"❌ Query failed: {r.message()}")
            client.logout()
            return False
        
        df = r.dataframe()
        if df is None or df.empty:
            print("⚠️  Query returned empty data (might be holiday/weekend)")
        else:
            print(f"✓ Query successful! Got {len(df)} rows")
            print(f"  Latest date: {df.iloc[-1]['date']}")
            print(f"  Latest close: {df.iloc[-1]['close']:.2f}")
        
        # 5. Logout
        client.logout()
        print("\n✓ Logged out successfully")
        print("\n=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tsl_connection()
    exit(0 if success else 1)
