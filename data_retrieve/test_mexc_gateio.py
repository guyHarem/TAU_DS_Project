"""
Test MEXC API for historical data availability
"""
import sys
import os
sys.path.insert(0, '/Users/guyharem/TAU/Projects/TAU_DS_Project/data_retrieve')

if __name__ == "__main__":
    print("="*60)
    print("TESTING MEXC - DATA AVAILABILITY BY DATE")
    print("="*60 + "\n")
    
    print("Testing MEXC for various dates to find oldest available data:\n")
    
    mexc_test_dates = [
        ("2026-03-15 00:00", "2026-03-15 23:59", "Mar 15 2026 (recent)"),
        ("2026-02-15 00:00", "2026-02-15 23:59", "Feb 15 2026 (1 month)"),
        ("2026-01-15 00:00", "2026-01-15 23:59", "Jan 15 2026 (2 months)"),
        ("2025-12-15 00:00", "2025-12-15 23:59", "Dec 15 2025 (3 months)"),
        ("2025-11-15 00:00", "2025-11-15 23:59", "Nov 15 2025 (4 months)"),
        ("2025-10-15 00:00", "2025-10-15 23:59", "Oct 15 2025 (5 months)"),
        ("2025-09-01 00:00", "2025-09-01 23:59", "Sep 1 2025 (6+ months)"),
    ]
    
    for start, end, label in mexc_test_dates:
        try:
            from mexc_api import fetch_data as mexc_fetch
            df = mexc_fetch("BTC/USD", start, end)
            if len(df) > 0:
                print(f"✅ {label}: {len(df):5} rows")
            else:
                print(f"⚠️  {label}: 0 rows (no data)")
        except Exception as e:
            print(f"❌ {label}: Error - {str(e)[:40]}")
    
    print(f"\n{'='*60}")
    print("MEXC TEST COMPLETE")
    print(f"{'='*60}")