#!/usr/bin/env python3
"""
Borsdata API Explorer
=====================
This script makes one call to each Borsdata API endpoint to show the data structure.

API Documentation: https://apidoc.borsdata.se/swagger/index.html
GitHub Wiki: https://github.com/Borsdata-Sweden/API/wiki

Usage:
    python borsdata_api_explorer.py YOUR_API_KEY
    
Note: 
    - API is rate limited to 100 calls per 10 seconds
    - Some endpoints require Pro+ membership (Global data, Holdings)
"""

import requests
import json
import sys
import time
from datetime import datetime, timedelta

BASE_URL = "https://apiservice.borsdata.se/v1"


def make_request(endpoint: str, api_key: str, params: dict = None) -> dict:
    """Make a request to the Borsdata API and return the JSON response."""
    url = f"{BASE_URL}{endpoint}"
    
    if params is None:
        params = {}
    params["authKey"] = api_key
    
    print(f"\n{'='*80}")
    print(f"Endpoint: {endpoint}")
    print(f"Full URL: {url}")
    print(f"Params: {params}")
    print("-" * 80)
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Pretty print with truncation for large responses
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            if len(formatted) > 3000:
                print(f"Response (truncated, total {len(formatted)} chars):")
                print(formatted[:3000])
                print("\n... [truncated] ...")
            else:
                print("Response:")
                print(formatted)
            return data
        elif response.status_code == 403:
            print("Error 403: Forbidden - Check your API key or make sure it's passed as ?authKey=")
            return None
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            print(f"Error 429: Rate limited. Retry after: {retry_after} seconds")
            return None
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500] if response.text else "No response body")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def explore_all_endpoints(api_key: str):
    """Call each API endpoint once to explore the data structure."""
    
    results = {}
    
    # =========================================================================
    # METADATA ENDPOINTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: METADATA ENDPOINTS")
    print("=" * 80)
    
    # 1. Get all instruments (companies/stocks)
    results["instruments"] = make_request("/instruments", api_key)
    time.sleep(0.2)
    
    # 2. Get updated instruments (recently modified)
    results["instruments_updated"] = make_request("/instruments/updated", api_key)
    time.sleep(0.2)
    
    # 3. Get all markets (Stockholm, Helsinki, Copenhagen, Oslo, etc.)
    results["markets"] = make_request("/markets", api_key)
    time.sleep(0.2)
    
    # 4. Get all branches (sub-sectors)
    results["branches"] = make_request("/branches", api_key)
    time.sleep(0.2)
    
    # 5. Get all sectors
    results["sectors"] = make_request("/sectors", api_key)
    time.sleep(0.2)
    
    # 6. Get all countries
    results["countries"] = make_request("/countries", api_key)
    time.sleep(0.2)
    
    # 7. Get translation metadata (Swedish/English translations)
    results["translation_metadata"] = make_request("/translationmetadata", api_key)
    time.sleep(0.2)
    
    # =========================================================================
    # STOCK PRICE ENDPOINTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: STOCK PRICE ENDPOINTS")
    print("=" * 80)
    
    # Use a sample instrument ID (e.g., 3 = ABB, 97 = H&M, 750 = Evolution)
    sample_inst_id = 3
    
    # 8. Get stock prices for one instrument
    results["stockprices_single"] = make_request(
        f"/instruments/{sample_inst_id}/stockprices", 
        api_key,
        {"maxcount": 10}  # Limit to last 10 for demo
    )
    time.sleep(0.2)
    
    # 9. Get last stock price for all instruments
    results["stockprices_last"] = make_request("/instruments/stockprices/last", api_key)
    time.sleep(0.2)
    
    # 10. Get stock prices by specific date
    yesterday = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    results["stockprices_by_date"] = make_request(
        "/instruments/stockprices/date",
        api_key,
        {"date": yesterday}
    )
    time.sleep(0.2)
    
    # 11. Get stock prices for multiple instruments (array call)
    results["stockprices_array"] = make_request(
        "/instruments/stockprices",
        api_key,
        {"instList": "3,97,750", "maxcount": 5}
    )
    time.sleep(0.2)
    
    # =========================================================================
    # REPORT DATA ENDPOINTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: REPORT DATA ENDPOINTS")
    print("=" * 80)
    
    # 12. Get yearly reports for one instrument
    results["reports_year"] = make_request(
        f"/instruments/{sample_inst_id}/reports/year",
        api_key,
        {"maxcount": 5}
    )
    time.sleep(0.2)
    
    # 13. Get R12 (rolling 12 months) reports
    results["reports_r12"] = make_request(
        f"/instruments/{sample_inst_id}/reports/r12",
        api_key,
        {"maxcount": 5}
    )
    time.sleep(0.2)
    
    # 14. Get quarterly reports
    results["reports_quarter"] = make_request(
        f"/instruments/{sample_inst_id}/reports/quarter",
        api_key,
        {"maxcount": 5}
    )
    time.sleep(0.2)
    
    # 15. Get all reports for one instrument
    results["reports_all"] = make_request(
        f"/instruments/{sample_inst_id}/reports",
        api_key,
        {"maxcount": 5}
    )
    time.sleep(0.2)
    
    # 16. Get reports for multiple instruments (array call)
    results["reports_array"] = make_request(
        "/instruments/reports",
        api_key,
        {"instList": "3,97"}
    )
    time.sleep(0.2)
    
    # =========================================================================
    # KPI (Key Performance Indicators) ENDPOINTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: KPI ENDPOINTS")
    print("=" * 80)
    
    # 17. Get KPI metadata (list of available KPIs)
    results["kpi_metadata"] = make_request("/instruments/kpis/metadata", api_key)
    time.sleep(0.2)
    
    # 18. Get updated KPI metadata
    results["kpi_metadata_updated"] = make_request("/instruments/kpis/updated", api_key)
    time.sleep(0.2)
    
    # 19. KPI Screener - Get one KPI for all instruments
    # KPI ID 2 = P/E, Time period: year, Calc type: latest, Calc group: mean
    # Format: /instruments/kpis/{kpiId}/{reportType}/{priceType}/{calcGroup}
    results["kpi_screener"] = make_request(
        "/instruments/kpis/2/year/latest/mean",
        api_key
    )
    time.sleep(0.2)
    
    # 20. Historical KPI data for one instrument
    # Get P/E (kpiId=2) history for instrument
    results["kpi_history"] = make_request(
        f"/instruments/{sample_inst_id}/kpis/2/year/mean/history",
        api_key
    )
    time.sleep(0.2)
    
    # =========================================================================
    # CALENDAR ENDPOINTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: CALENDAR ENDPOINTS")
    print("=" * 80)
    
    # 21. Report calendar (upcoming earnings reports)
    results["calendar_report"] = make_request("/instruments/calendar/report", api_key)
    time.sleep(0.2)
    
    # 22. Dividend calendar
    results["calendar_dividend"] = make_request("/instruments/calendar/dividend", api_key)
    time.sleep(0.2)
    
    # =========================================================================
    # STOCK SPLITS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: STOCK SPLITS")
    print("=" * 80)
    
    # 23. Get stock splits
    results["stocksplits"] = make_request("/instruments/stocksplits", api_key)
    time.sleep(0.2)
    
    # =========================================================================
    # HOLDINGS ENDPOINTS (Pro+ only - may fail for Pro members)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: HOLDINGS ENDPOINTS (Pro+ only)")
    print("=" * 80)
    
    # 24. Insider holdings/transactions
    results["holdings_insider"] = make_request("/instruments/holdings/insider", api_key)
    time.sleep(0.2)
    
    # 25. Short positions
    results["holdings_shorts"] = make_request("/instruments/holdings/shorts", api_key)
    time.sleep(0.2)
    
    # 26. Buybacks
    results["holdings_buyback"] = make_request("/instruments/holdings/buyback", api_key)
    time.sleep(0.2)
    
    # =========================================================================
    # GLOBAL INSTRUMENTS (Pro+ only)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION: GLOBAL ENDPOINTS (Pro+ only)")
    print("=" * 80)
    
    # 27. Get global instruments
    results["global_instruments"] = make_request("/instruments/global", api_key)
    time.sleep(0.2)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for v in results.values() if v is not None)
    failed = sum(1 for v in results.values() if v is None)
    
    print(f"\nTotal endpoints tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print("\nEndpoint status:")
    for name, result in results.items():
        status = "✓" if result is not None else "✗"
        print(f"  {status} {name}")
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python borsdata_api_explorer.py YOUR_API_KEY")
        print("\nGet your API key from: https://borsdata.se/en/mypage/direct")
        print("Note: You need a Pro or Pro+ membership to use the API")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    print("=" * 80)
    print("BORSDATA API EXPLORER")
    print("=" * 80)
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    explore_all_endpoints(api_key)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()