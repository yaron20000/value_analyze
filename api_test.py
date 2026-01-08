#!/usr/bin/env python3
"""
Borsdata API Explorer (Nordic Only)
===================================
This script makes one call to each Borsdata API endpoint and saves results to JSON files.
Modified to only fetch Nordic data (global endpoint removed).

API Documentation: https://apidoc.borsdata.se/swagger/index.html
GitHub Wiki: https://github.com/Borsdata-Sweden/API/wiki

Usage:
    python borsdata_api_explorer.py YOUR_API_KEY
    
Note: 
    - API is rate limited to 100 calls per 10 seconds
    - Some endpoints require Pro+ membership (Holdings)
    - Results are saved to the 'results' folder
"""

import requests
import json
import sys
import time
import os
from datetime import datetime, timedelta

BASE_URL = "https://apiservice.borsdata.se/v1"
RESULTS_DIR = "results"


def ensure_results_dir():
    """Create the results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")


def save_result(name: str, data: dict, error: str = None):
    """Save the result to a JSON file."""
    filepath = os.path.join(RESULTS_DIR, f"{name}.json")
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "endpoint_name": name,
        "success": data is not None,
        "error": error,
        "data": data
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return filepath


def make_request(endpoint: str, api_key: str, params: dict = None) -> tuple[dict, str]:
    """Make a request to the Borsdata API and return the JSON response and any error."""
    url = f"{BASE_URL}{endpoint}"
    
    if params is None:
        params = {}
    params["authKey"] = api_key
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 403:
            return None, "Error 403: Forbidden - Check your API key or membership level"
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            return None, f"Error 429: Rate limited. Retry after: {retry_after} seconds"
        else:
            return None, f"Error {response.status_code}: {response.text[:200] if response.text else 'No response body'}"
            
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"


def explore_all_endpoints(api_key: str):
    """Call each API endpoint once and save results to JSON files."""
    
    ensure_results_dir()
    
    results_summary = {}
    
    # Define all endpoints to test
    endpoints = []
    
    # Sample instrument ID for single-instrument endpoints
    sample_inst_id = 750
    yesterday = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    
    # =========================================================================
    # METADATA ENDPOINTS
    # =========================================================================
    endpoints.extend([
        ("instruments", "/instruments", {}),
        ("instruments_updated", "/instruments/updated", {}),
        ("markets", "/markets", {}),
        ("branches", "/branches", {}),
        ("sectors", "/sectors", {}),
        ("countries", "/countries", {}),
        ("translation_metadata", "/translationmetadata", {}),
    ])
    
    # =========================================================================
    # STOCK PRICE ENDPOINTS
    # =========================================================================
    endpoints.extend([
        ("stockprices_single", f"/instruments/{sample_inst_id}/stockprices", {"maxcount": 10}),
        ("stockprices_last", "/instruments/stockprices/last", {}),
        ("stockprices_by_date", "/instruments/stockprices/date", {"date": yesterday}),
        ("stockprices_array", "/instruments/stockprices", {"instList": "3,97,750", "maxcount": 5}),
    ])
    
    # =========================================================================
    # REPORT DATA ENDPOINTS
    # =========================================================================
    endpoints.extend([
        ("reports_year", f"/instruments/{sample_inst_id}/reports/year", {"maxcount": 5}),
        ("reports_r12", f"/instruments/{sample_inst_id}/reports/r12", {"maxcount": 5}),
        ("reports_quarter", f"/instruments/{sample_inst_id}/reports/quarter", {"maxcount": 5}),
        ("reports_all", f"/instruments/{sample_inst_id}/reports", {"maxcount": 5}),
        ("reports_array", "/instruments/reports", {"instList": "3,97"}),
    ])
    
    # =========================================================================
    # KPI ENDPOINTS
    # =========================================================================
    endpoints.extend([
        ("kpi_metadata", "/instruments/kpis/metadata", {}),
        ("kpi_metadata_updated", "/instruments/kpis/updated", {}),
        # kpi_screener endpoint removed - requires specific KPI calculation type that varies
        # Use kpi_history instead for individual instruments
        ("kpi_history", f"/instruments/{sample_inst_id}/kpis/2/year/mean/history", {}),
    ])
    
    # =========================================================================
    # CALENDAR ENDPOINTS
    # =========================================================================
    endpoints.extend([
        ("calendar_report", "/instruments/report/calendar/", {}),
        ("calendar_dividend", "/instruments/dividend/calendar/", {}),
    ])
    
    # =========================================================================
    # STOCK SPLITS
    # =========================================================================
    endpoints.append(
        ("stocksplits", "/instruments/stocksplits", {})
    )
    
    # =========================================================================
    # HOLDINGS ENDPOINTS (Pro+ only)
    # =========================================================================
    endpoints.extend([
        ("holdings_insider", "/holdings/insider", {}),
        ("holdings_shorts", "/holdings/shorts", {}),
        ("holdings_buyback", "/holdings/buyback", {}),
    ])
    
    # =========================================================================
    # GLOBAL INSTRUMENTS - SKIPPED (Nordic only mode)
    # =========================================================================
    # Removed to only fetch Nordic data
    
    # =========================================================================
    # Process all endpoints
    # =========================================================================
    total = len(endpoints)
    
    for i, (name, endpoint, params) in enumerate(endpoints, 1):
        print(f"[{i}/{total}] Fetching {name}... endpoint {endpoint} {params}", end=" ")
    
        data, error = make_request(endpoint, api_key, params)        
        filepath = save_result(name, data, error)
        
        if data is not None:
            print(f"✓ Saved to {filepath}")
            results_summary[name] = {"success": True, "file": filepath}
        else:
            print(f"✗ {error}")
            results_summary[name] = {"success": False, "error": error, "file": filepath}
        
        time.sleep(0.2)  # Rate limiting
    
    # =========================================================================
    # Save summary
    # =========================================================================
    summary = {
        "timestamp": datetime.now().isoformat(),
        "api_key_prefix": f"{api_key[:8]}...{api_key[-4:]}",
        "total_endpoints": total,
        "successful": sum(1 for v in results_summary.values() if v["success"]),
        "failed": sum(1 for v in results_summary.values() if not v["success"]),
        "endpoints": results_summary
    }
    
    summary_path = os.path.join(RESULTS_DIR, "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY (Nordic Only)")
    print(f"{'='*60}")
    print(f"Total endpoints: {total}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Summary saved to: {summary_path}")
    
    return results_summary


def main():
    if len(sys.argv) != 2:
        print("Usage: python borsdata_api_explorer.py YOUR_API_KEY")
        print("\nGet your API key from: https://borsdata.se/en/mypage/direct")
        print("Note: You need a Pro or Pro+ membership to use the API")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    print("=" * 60)
    print("BORSDATA API EXPLORER (Nordic Only)")
    print("=" * 60)
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Output: {RESULTS_DIR}/")
    print("=" * 60 + "\n")
    
    explore_all_endpoints(api_key)
    
    print("\nDone!")


if __name__ == "__main__":
    main()