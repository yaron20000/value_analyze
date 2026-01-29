#!/usr/bin/env python3
"""
Quick test script to check if Borsdata API supports batch KPI endpoints.
This runs independently and doesn't modify any data.
"""

import requests
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def test_batch_kpi_endpoint(api_key):
    """Test various batch endpoint patterns to see what works."""

    BASE_URL = "https://apiservice.borsdata.se/v1"

    # Test with 2 instruments and 2 KPIs
    inst_list = "3,19"  # ABB, Atlas Copco
    kpi_list = "2,3"    # P/E, P/S

    print("="*70)
    print("TESTING BORSDATA BATCH KPI ENDPOINTS")
    print("="*70)
    print(f"Instruments: {inst_list} (ABB, Atlas Copco)")
    print(f"KPIs: {kpi_list} (P/E, P/S)")
    print()

    # Test patterns
    test_cases = [
        {
            "name": "Pattern 1: Both as query params",
            "url": f"{BASE_URL}/instruments/kpis",
            "params": {"instList": inst_list, "kpiList": kpi_list, "authKey": api_key}
        },
        {
            "name": "Pattern 2: KPI in path, inst in params",
            "url": f"{BASE_URL}/instruments/kpis/{kpi_list}",
            "params": {"instList": inst_list, "authKey": api_key}
        },
        {
            "name": "Pattern 3: Inst in path, KPI in params",
            "url": f"{BASE_URL}/instruments/{inst_list}/kpis",
            "params": {"kpiList": kpi_list, "authKey": api_key}
        },
        {
            "name": "Pattern 4: KPI calc type in path",
            "url": f"{BASE_URL}/instruments/kpis/{kpi_list}/year/mean/history",
            "params": {"instList": inst_list, "authKey": api_key}
        },
        {
            "name": "Pattern 5: Array-style (like stockprices)",
            "url": f"{BASE_URL}/instruments/kpis/history",
            "params": {"instList": inst_list, "kpiList": kpi_list, "authKey": api_key}
        },
    ]

    successful_patterns = []

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"URL: {test['url']}")

        # Build params string for display (without auth key)
        display_params = {k: v for k, v in test['params'].items() if k != 'authKey'}
        print(f"Params: {display_params}")

        try:
            response = requests.get(test['url'], params=test['params'], timeout=10)

            if response.status_code == 200:
                data = response.json()
                print(f"✓ SUCCESS!")
                print(f"  Status: {response.status_code}")
                print(f"  Response type: {type(data)}")

                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
                    # Print first few characters of response for inspection
                    import json
                    preview = json.dumps(data, indent=2)[:500]
                    print(f"  Preview:\n{preview}...")

                successful_patterns.append(test['name'])
            else:
                print(f"✗ Failed")
                print(f"  Status: {response.status_code}")
                print(f"  Error: {response.text[:200]}")

        except Exception as e:
            print(f"✗ Exception: {e}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if successful_patterns:
        print(f"✓ Found {len(successful_patterns)} working pattern(s):")
        for pattern in successful_patterns:
            print(f"  - {pattern}")
    else:
        print("✗ No batch KPI endpoints found")
        print("\nThis means:")
        print("  - Must fetch each KPI individually for each instrument")
        print("  - For 4 instruments × 4 KPIs = 16 separate API calls")
        print("  - Can optimize with parallel requests")

    print("="*70)

    return len(successful_patterns) > 0


if __name__ == "__main__":
    api_key = os.getenv('BORSDATA_API_KEY')

    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]

    if not api_key:
        print("Error: No API key provided")
        print("\nUsage:")
        print("  python test_batch_api.py YOUR_API_KEY")
        print("  OR set BORSDATA_API_KEY environment variable")
        sys.exit(1)

    print(f"Using API key: {api_key[:8]}...{api_key[-4:]}\n")

    success = test_batch_kpi_endpoint(api_key)

    sys.exit(0 if success else 1)
