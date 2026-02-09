#!/usr/bin/env python3
"""
Borsdata API Data Fetcher and PostgreSQL Storer
================================================
This script fetches data from Borsdata API endpoints and stores it in PostgreSQL.

Features:
- Fetches all Instrument Meta APIs
- Fetches stock-specific APIs for specified stocks or all Nordic instruments
- Implements rate limiting: 100 API calls per 10 seconds
- Saves JSON responses to results/ directory before DB insertion
- Stores raw data in PostgreSQL for future AI processing (optional)
- Handles API rate limiting and errors gracefully
- Can run without database (JSON files only)
- Automatically skips re-fetching data that already exists (within 24 hours)
- Prevents duplicate data insertion using database constraints

Usage:
    # Fetch all Nordic instruments (Sweden, Norway, Finland, Denmark)
    python fetch_and_store.py YOUR_API_KEY --nordics --db-password yourpass

    # Fetch specific instruments only (no database)
    python fetch_and_store.py YOUR_API_KEY --instruments 3,19,199 --no-db

    # With PostgreSQL database (skips existing data by default)
    python fetch_and_store.py YOUR_API_KEY --db-password yourpass

    # Force refetch all data even if it exists
    python fetch_and_store.py YOUR_API_KEY --db-password yourpass --force-refetch

    # With environment variables
    export BORSDATA_API_KEY=your_key
    export DB_PASSWORD=yourpass
    python fetch_and_store.py --nordics

    # Full database configuration
    python fetch_and_store.py YOUR_API_KEY --db-host localhost --db-name borsdata --db-user postgres --db-password yourpass --nordics

Requirements:
    pip install requests psycopg2-binary python-dotenv
"""

import requests
import json
import sys
import time
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

BASE_URL = "https://apiservice.borsdata.se/v1"
RESULTS_DIR = "results"

# Nordic country IDs (Sverige, Norge, Finland, Danmark)
NORDIC_COUNTRY_IDS = [1, 2, 3, 4]


class RateLimiter:
    """Rate limiter that allows max_calls within time_window seconds."""

    def __init__(self, max_calls: int = 100, time_window: float = 10.0):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in time_window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()

        # Remove calls outside the time window
        while self.call_times and self.call_times[0] <= now - self.time_window:
            self.call_times.popleft()

        # If we've hit the limit, wait until we can make another call
        if len(self.call_times) >= self.max_calls:
            sleep_time = self.call_times[0] + self.time_window - now
            if sleep_time > 0:
                print(f"    ⏳ Rate limit reached ({self.max_calls} calls/{self.time_window}s). Waiting {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                # Clean up old entries after waiting
                now = time.time()
                while self.call_times and self.call_times[0] <= now - self.time_window:
                    self.call_times.popleft()

        # Record this call
        self.call_times.append(now)

    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        now = time.time()
        # Count calls in current window
        recent_calls = sum(1 for t in self.call_times if t > now - self.time_window)
        return {
            "recent_calls": recent_calls,
            "max_calls": self.max_calls,
            "time_window": self.time_window,
            "remaining": self.max_calls - recent_calls
        }


class BorsdataFetcher:
    """Handles fetching data from Borsdata API and storing to PostgreSQL."""

    def __init__(self, api_key: str, db_config: dict = None, skip_existing: bool = True):
        self.api_key = api_key
        self.db_config = db_config
        self.db_enabled = db_config is not None
        self.skip_existing = skip_existing
        self.conn = None
        self.cursor = None
        self.fetch_log_id = None
        self.rate_limiter = RateLimiter(max_calls=100, time_window=10.0)
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": time.time()
        }

    def ensure_results_dir(self):
        """Create the results directory if it doesn't exist."""
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            print(f"Created directory: {RESULTS_DIR}")

    def connect_db(self):
        """Connect to PostgreSQL database."""
        if not self.db_enabled:
            print("ℹ Database disabled - saving to JSON files only")
            return

        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config.get('port', 5432)
            )
            self.cursor = self.conn.cursor()
            print(f"✓ Connected to PostgreSQL database: {self.db_config['database']}")
        except Exception as e:
            print(f"✗ Failed to connect to database: {e}")
            raise

    def close_db(self):
        """Close database connection."""
        if not self.db_enabled:
            return

        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")

    def create_fetch_log(self, instruments: List[int]):
        """Create a new fetch log entry and return its ID."""
        if not self.db_enabled:
            return

        try:
            self.cursor.execute("""
                INSERT INTO api_fetch_log (run_timestamp, total_endpoints, successful_endpoints,
                                          failed_endpoints, instruments_fetched, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                datetime.now(),
                0,  # Will update at the end
                0,
                0,
                [str(i) for i in instruments],
                f"Fetching data for instruments: {instruments}"
            ))
            self.fetch_log_id = self.cursor.fetchone()[0]
            self.conn.commit()
            print(f"✓ Created fetch log entry: {self.fetch_log_id}")
        except Exception as e:
            print(f"✗ Failed to create fetch log: {e}")
            self.conn.rollback()

    def update_fetch_log(self):
        """Update the fetch log with final statistics."""
        if not self.db_enabled:
            return

        try:
            duration = time.time() - self.stats['start_time']
            self.cursor.execute("""
                UPDATE api_fetch_log
                SET total_endpoints = %s,
                    successful_endpoints = %s,
                    failed_endpoints = %s,
                    duration_seconds = %s
                WHERE id = %s
            """, (
                self.stats['total'],
                self.stats['successful'],
                self.stats['failed'],
                round(duration, 2),
                self.fetch_log_id
            ))
            self.conn.commit()
            print(f"✓ Updated fetch log with final statistics")
        except Exception as e:
            print(f"✗ Failed to update fetch log: {e}")
            self.conn.rollback()

    def save_to_json(self, name: str, data: dict, error: str = None, kpi_name: str = None) -> str:
        return
        """Save the result to a JSON file in results directory."""
        filepath = os.path.join(RESULTS_DIR, f"{name}.json")

        result = {
            "timestamp": datetime.now().isoformat(),
            "endpoint_name": name,
            "kpi_name": kpi_name,
            "success": data is not None,
            "error": error,
            "data": data
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return filepath

    def check_existing_data(self, name: str, instrument_id: int = None,
                           hours_threshold: int = 24) -> bool:
        """Check if data already exists for this endpoint within the threshold period."""
        if not self.db_enabled:
            return False

        try:
            threshold_time = datetime.now() - timedelta(hours=hours_threshold)

            self.cursor.execute("""
                SELECT COUNT(*) FROM api_raw_data
                WHERE endpoint_name = %s
                AND (instrument_id = %s OR (instrument_id IS NULL AND %s IS NULL))
                AND fetch_timestamp >= %s
                AND success = true
            """, (name, instrument_id, instrument_id, threshold_time))

            count = self.cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            print(f"    ⚠ Failed to check existing data: {e}")
            return False

    def save_to_db(self, name: str, endpoint_path: str, data: dict,
                   error: str = None, instrument_id: int = None, params: dict = None, kpi_name: str = None):
        """Save to DB - only keep new record if data has changed."""
        if not self.db_enabled:
            return

        try:
            # Get the most recent existing data for comparison
            self.cursor.execute("""
                SELECT raw_data FROM api_raw_data
                WHERE endpoint_name = %s
                AND (instrument_id = %s OR (instrument_id IS NULL AND %s IS NULL))
                ORDER BY fetch_timestamp DESC LIMIT 1
            """, (name, instrument_id, instrument_id))

            existing = self.cursor.fetchone()

            if existing and existing[0] == data:
                # Data unchanged - just update timestamp of existing record
                self.cursor.execute("""
                    UPDATE api_raw_data
                    SET fetch_timestamp = %s,
                        kpi_name = COALESCE(%s, kpi_name)
                    WHERE endpoint_name = %s
                    AND (instrument_id = %s OR (instrument_id IS NULL AND %s IS NULL))
                    AND raw_data = %s
                """, (datetime.now(), kpi_name, name, instrument_id, instrument_id, Json(data)))
                print(f"    ℹ Data unchanged - updated timestamp only")
            else:
                # Data changed (or new) - insert new record
                self.cursor.execute("""
                    INSERT INTO api_raw_data
                    (endpoint_name, endpoint_path, instrument_id, kpi_name, fetch_timestamp,
                     success, error_message, raw_data, request_params)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    name, endpoint_path, instrument_id, kpi_name,
                    datetime.now(), data is not None, error,
                    Json(data) if data else None,
                    Json(params) if params else None
                ))
                if existing:
                    print(f"    ✓ Data changed - new version saved")
                else:
                    print(f"    ✓ New data saved")

            self.conn.commit()
        except Exception as e:
            print(f"    ✗ Failed to save to DB: {e}")
            self.conn.rollback()

    def make_request(self, endpoint: str, params: dict = None) -> Tuple[Optional[dict], Optional[str]]:
        """Make a request to the Borsdata API and return the JSON response and any error."""
        # Apply rate limiting before making the request
        self.rate_limiter.wait_if_needed()

        url = f"{BASE_URL}{endpoint}"

        if params is None:
            params = {}

        # Don't include authKey in the log
        params_copy = params.copy()
        params["authKey"] = self.api_key

        # Build query string for logging (without auth key)
        if params_copy:
            query_str = "&".join([f"{k}={v}" for k, v in params_copy.items()])
            full_url = f"{url}?{query_str}"
        else:
            full_url = url

        print(f"    → API: GET {full_url}")

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                # Log response size
                if isinstance(data, dict):
                    if 'instruments' in data:
                        print(f"    ← Response: {len(data['instruments'])} instruments")
                    elif 'stockPricesList' in data:
                        print(f"    ← Response: {len(data['stockPricesList'])} prices")
                    elif 'reports' in data:
                        print(f"    ← Response: {len(data['reports'])} reports")
                    elif 'list' in data:
                        print(f"    ← Response: {len(data['list'])} items in list")
                    else:
                        print(f"    ← Response: {len(str(data))} bytes")
                return data, None
            elif response.status_code == 403:
                return None, "Error 403: Forbidden - Check your API key or membership level"
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "unknown")
                return None, f"Error 429: Rate limited. Retry after: {retry_after} seconds"
            else:
                return None, f"Error {response.status_code}: {response.text[:200] if response.text else 'No response body'}"

        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {e}"

    def fetch_endpoint(self, name: str, endpoint: str, params: dict = None,
                      instrument_id: int = None, skip_if_exists: bool = True, kpi_name: str = None):
        """Fetch a single endpoint, save to JSON and DB.

        Args:
            name: Endpoint name for identification
            endpoint: API endpoint path
            params: Request parameters
            instrument_id: Instrument ID for stock-specific endpoints
            skip_if_exists: If True, skip fetching if data exists within 24 hours
            kpi_name: KPI name for KPI endpoints (e.g., "P/E", "ROE", "Revenue Growth")
        """
        self.stats['total'] += 1

        # Check if we already have recent data for this endpoint
        if skip_if_exists and self.skip_existing and self.db_enabled:
            if self.check_existing_data(name, instrument_id):
                print(f"    ⏭ Skipping - recent data already exists (within 24 hours)")
                self.stats['skipped'] += 1
                return True

        data, error = self.make_request(endpoint, params)

        # Save to JSON file
        filepath = self.save_to_json(name, data, error, kpi_name)

        # Save to database
        if self.db_enabled:
            self.save_to_db(name, endpoint, data, error, instrument_id, params, kpi_name)

        if data is not None:
            self.stats['successful'] += 1
            print(f"    ✓ Saved to {filepath}")
            if self.db_enabled:
                print(f"    ✓ Saved to database")
            return True
        else:
            self.stats['failed'] += 1
            print(f"    ✗ Error: {error}")
            print(f"    ✓ Error logged to {filepath}")
            return False

    # Max instruments per batch KPI request to avoid API URL length limits
    KPI_BATCH_CHUNK_SIZE = 100

    def _get_batch_instrument_count(self, endpoint_name: str) -> int:
        """Count how many instruments are in an existing KPI batch record."""
        try:
            self.cursor.execute("""
                SELECT jsonb_array_length(raw_data->'kpisList')
                FROM api_raw_data
                WHERE endpoint_name = %s AND success = true
                ORDER BY fetch_timestamp DESC
                LIMIT 1
            """, (endpoint_name,))
            row = self.cursor.fetchone()
            return row[0] if row and row[0] else 0
        except Exception:
            return 0

    def fetch_kpi_batch(self, kpi_id: int, kpi_name: str, kpi_display_name: str,
                        instrument_ids: List[int], skip_if_exists: bool = True):
        """Fetch a single KPI for ALL instruments, chunked to avoid API URL limits.

        Splits instrument_ids into chunks of KPI_BATCH_CHUNK_SIZE and merges
        results, since the Borsdata API silently truncates long instList params.

        Args:
            kpi_id: The KPI ID to fetch
            kpi_name: Short name for identification (e.g., "pe", "roe")
            kpi_display_name: Display name (e.g., "P/E", "Return on Equity")
            instrument_ids: List of all instrument IDs to fetch
            skip_if_exists: If True, skip if data already exists
        """
        self.stats['total'] += 1

        endpoint = f"/instruments/kpis/{kpi_id}/year/mean/history"
        endpoint_name = f"kpi_{kpi_name}_batch"

        # Check if we already have recent data with sufficient instrument coverage
        if skip_if_exists and self.skip_existing and self.db_enabled:
            if self.check_existing_data(endpoint_name, None):
                # Also verify the existing batch covers enough instruments
                existing_count = self._get_batch_instrument_count(endpoint_name)
                if existing_count >= len(instrument_ids) * 0.8:
                    print(f"    ⏭ Skipping - recent data exists ({existing_count} instruments)")
                    self.stats['skipped'] += 1
                    return True
                else:
                    print(f"    ↻ Re-fetching - existing data has {existing_count} instruments, "
                          f"need {len(instrument_ids)}")

        # Split instrument_ids into chunks to avoid API URL length limits
        chunks = [
            instrument_ids[i:i + self.KPI_BATCH_CHUNK_SIZE]
            for i in range(0, len(instrument_ids), self.KPI_BATCH_CHUNK_SIZE)
        ]

        merged_kpis_list = []
        any_success = False

        for chunk_idx, chunk in enumerate(chunks):
            inst_list = ",".join(str(i) for i in chunk)
            params = {"instList": inst_list}

            data, error = self.make_request(endpoint, params)

            if data is not None and 'kpisList' in data:
                merged_kpis_list.extend(data.get('kpisList', []))
                any_success = True
                if len(chunks) > 1:
                    print(f"      Chunk {chunk_idx + 1}/{len(chunks)}: "
                          f"{len(data.get('kpisList', []))} instruments")
            elif error:
                print(f"      Chunk {chunk_idx + 1}/{len(chunks)} failed: {error}")

        if not any_success:
            self.stats['failed'] += 1
            print(f"    ✗ All chunks failed for KPI {kpi_display_name}")
            if self.db_enabled:
                self.save_to_db(endpoint_name, endpoint, None, "All chunks failed", None, {}, kpi_display_name)
            return False

        self.stats['successful'] += 1

        # Build merged response matching the original batch format
        merged_data = {"kpiId": kpi_id, "kpisList": merged_kpis_list}

        # Save the merged batch response to DB
        if self.db_enabled:
            self.save_to_db(endpoint_name, endpoint, merged_data, None, None, {}, kpi_display_name)
            print(f"    ✓ Batch KPI {kpi_display_name}: {len(merged_kpis_list)} instruments "
                  f"({len(chunks)} chunk{'s' if len(chunks) > 1 else ''})")

            # Also save individual instrument records
            for entry in merged_kpis_list:
                inst_id = entry.get('instrument')
                if inst_id is None:
                    continue
                inst_data = {"values": entry.get('values', [])}
                inst_name = f"kpi_{kpi_name}_{inst_id}"
                self.save_to_db(inst_name, endpoint, inst_data, None, inst_id, {}, kpi_display_name)

        return True

    def get_nordic_instruments(self) -> List[int]:
        """Fetch all instruments and filter for Nordic countries."""
        print("\n" + "="*70)
        print("GETTING NORDIC INSTRUMENTS")
        print("="*70)

        # Fetch instruments list
        data, error = self.make_request("/instruments")

        if error or not data:
            print(f"✗ Failed to fetch instruments: {error}")
            return []

        instruments = data.get("instruments", [])
        print(f"✓ Fetched {len(instruments)} total instruments")

        # Filter for Nordic countries (1=Sverige, 2=Norge, 3=Finland, 4=Danmark)
        nordic_instruments = [
            inst["insId"] for inst in instruments
            if inst.get("countryId") in NORDIC_COUNTRY_IDS
        ]

        print(f"✓ Found {len(nordic_instruments)} Nordic instruments:")
        country_counts = {}
        for inst in instruments:
            if inst.get("countryId") in NORDIC_COUNTRY_IDS:
                country_id = inst.get("countryId")
                country_counts[country_id] = country_counts.get(country_id, 0) + 1

        country_names = {1: "Sweden", 2: "Norway", 3: "Finland", 4: "Denmark"}
        for country_id, count in sorted(country_counts.items()):
            print(f"  - {country_names.get(country_id, f'Country {country_id}')}: {count} instruments")

        return nordic_instruments

    def fetch_all_metadata(self):
        """Fetch all Instrument Meta APIs (metadata endpoints)."""
        print("\n" + "="*70)
        print("FETCHING INSTRUMENT META APIS (METADATA)")
        print("="*70)

        metadata_endpoints = [
            ("instruments", "/instruments", {}),
            ("instruments_updated", "/instruments/updated", {}),
            ("markets", "/markets", {}),
            ("branches", "/branches", {}),
            ("sectors", "/sectors", {}),
            ("countries", "/countries", {}),
            ("translation_metadata", "/translationmetadata", {}),
            ("kpi_metadata", "/instruments/kpis/metadata", {}),
            ("kpi_metadata_updated", "/instruments/kpis/updated", {}),
        ]

        for name, endpoint, params in metadata_endpoints:
            print(f"\n[{self.stats['total']+1}] Fetching {name}...")
            self.fetch_endpoint(name, endpoint, params)

    def fetch_stock_data(self, instrument_ids: List[int]):
        """Fetch stock-specific data for given instrument IDs."""
        print("\n" + "="*70)
        print(f"FETCHING STOCK-SPECIFIC DATA FOR {len(instrument_ids)} INSTRUMENTS")
        print("="*70)

        yesterday = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # =================================================================
        # PART 1: Per-instrument endpoints (stockprices, reports)
        # These don't have efficient batch alternatives for historical data
        # =================================================================
        print("\n--- Per-Instrument Endpoints (stockprices, reports) ---")
        for inst_id in instrument_ids:
            print(f"\n  Instrument ID: {inst_id}")

            endpoints = [
                (f"stockprices_{inst_id}", f"/instruments/{inst_id}/stockprices",
                 {"maxcount": 100}, inst_id),
                (f"reports_year_{inst_id}", f"/instruments/{inst_id}/reports/year",
                 {"maxcount": 10}, inst_id),
                (f"reports_r12_{inst_id}", f"/instruments/{inst_id}/reports/r12",
                 {"maxcount": 10}, inst_id),
                (f"reports_quarter_{inst_id}", f"/instruments/{inst_id}/reports/quarter",
                 {"maxcount": 20}, inst_id),
                (f"reports_all_{inst_id}", f"/instruments/{inst_id}/reports",
                 {"maxcount": 20}, inst_id),
            ]

            for name, endpoint, params, iid in endpoints:
                print(f"\n    [{self.stats['total']+1}] Fetching {name}...")
                self.fetch_endpoint(name, endpoint, params, iid)

        print("\n" + "="*70)
        print("DEBUG: PART 1 COMPLETED - Per-instrument endpoints done")
        print("="*70)

        # =================================================================
        # PART 2: Batch KPI fetching (OPTIMIZED!)
        # Fetch each KPI for ALL instruments in a single API call
        # This reduces API calls from (num_instruments × num_kpis) to (num_kpis)
        # For 700 instruments × 52 KPIs: 36,400 calls → 52 calls
        # =================================================================
        print("\n" + "="*70)
        print(f"DEBUG: Entering PART 2 - KPI Batch Fetching")
        print(f"FETCHING BATCH KPIs FOR ALL {len(instrument_ids)} INSTRUMENTS")
        print("="*70)
        print(f"DEBUG: instrument_ids count = {len(instrument_ids)}")
        print(f"DEBUG: First few instrument_ids = {instrument_ids[:5] if len(instrument_ids) > 5 else instrument_ids}")
        print(f"Using batch endpoint: /instruments/kpis/{{kpi_id}}/year/mean/history?instList=...")
        print(f"This fetches one KPI for ALL instruments in a single API call!")

        # WORKING KPIs ONLY - 52 Total (removed 28 failing KPIs)
        # See FAILING_KPIS_ANALYSIS.md for details on what was removed and why
        kpi_list = [
            # Tier 1 KPIs - All 35 Work Perfectly (100% success rate)
            # Valuation Metrics (8 KPIs)
            (1, "dividend_yield", "Dividend Yield"),
            (2, "pe", "P/E"),
            (3, "ps", "P/S"),
            (4, "pb", "P/B"),
            (10, "ev_ebit", "EV/EBIT"),
            (11, "ev_ebitda", "EV/EBITDA"),
            (13, "ev_fcf", "EV/FCF"),
            (15, "ev_s", "EV/S"),

            # Profitability & Margins (6 KPIs)
            (28, "gross_margin", "Gross Margin"),
            (29, "operating_margin", "Operating Margin"),
            (30, "profit_margin", "Profit Margin"),
            (31, "fcf_margin", "FCF Margin"),
            (32, "ebitda_margin", "EBITDA Margin"),
            (51, "ocf_margin", "OCF Margin"),

            # Growth Metrics (6 KPIs)
            (94, "revenue_growth", "Revenue Growth"),
            (96, "ebit_growth", "EBIT Growth"),
            (97, "earnings_growth", "Earnings Growth"),
            (98, "dividend_growth", "Dividend Growth"),
            (99, "book_value_growth", "Book Value Growth"),
            (100, "assets_growth", "Assets Growth"),

            # Return Metrics (4 KPIs)
            (33, "roe", "Return on Equity"),
            (34, "roa", "Return on Assets"),
            (36, "roc", "Return on Capital"),
            (37, "roic", "Return on Invested Capital"),

            # Financial Health (6 KPIs)
            (39, "equity_ratio", "Equity Ratio"),
            (40, "debt_to_equity", "Debt to Equity"),
            (41, "net_debt_pct", "Net Debt %"),
            (42, "net_debt_ebitda", "Net Debt/EBITDA"),
            (44, "current_ratio", "Current Ratio"),
            (46, "cash_pct", "Cash %"),

            # Size/Absolute Metrics (5 KPIs)
            (49, "enterprise_value", "Enterprise Value"),
            (50, "market_cap", "Market Cap"),
            (53, "revenue", "Revenue"),
            (56, "earnings", "Earnings"),
            (63, "fcf", "Free Cash Flow"),

            # Tier 2 KPIs - Working Subset (10 out of 25)
            # Per-Share Metrics (6 KPIs) - All work
            (5, "revenue_per_share", "Revenue per Share"),
            (6, "eps", "Earnings per Share"),
            (7, "dividend_per_share", "Dividend per Share"),
            (8, "book_value_per_share", "Book Value per Share"),
            (23, "fcf_per_share", "FCF per Share"),
            (68, "ocf_per_share", "OCF per Share"),

            # Additional Cash Flow (4 KPIs) - All work
            (24, "fcf_margin_pct", "FCF Margin %"),
            (27, "earnings_fcf", "Earnings/FCF"),
            (62, "ocf", "Operating Cash Flow"),
            (64, "capex", "Capex"),

            # Tier 3 KPIs - Working Subset (7 out of 20)
            # Additional Valuation (2 KPIs) - All work
            (19, "peg", "PEG Ratio"),
            (20, "dividend_payout", "Dividend Payout %"),

            # Additional Absolute Metrics (5 KPIs) - All work
            (54, "ebitda", "EBITDA"),
            (57, "total_assets", "Total Assets"),
            (58, "total_equity", "Total Equity"),
            (60, "net_debt", "Net Debt"),
            (61, "num_shares", "Number of Shares"),
        ]

        print(f"\nDEBUG: kpi_list has {len(kpi_list)} KPIs")
        print(f"Fetching {len(kpi_list)} KPIs using batch endpoint...")

        for idx, (kpi_id, kpi_name, kpi_display_name) in enumerate(kpi_list):
            print(f"\n  DEBUG: KPI loop iteration {idx+1}/{len(kpi_list)}")
            print(f"  [{self.stats['total']+1}] Fetching KPI: {kpi_display_name} (ID: {kpi_id})...")
            try:
                self.fetch_kpi_batch(kpi_id, kpi_name, kpi_display_name, instrument_ids)
                print(f"  DEBUG: fetch_kpi_batch completed for {kpi_display_name}")
            except Exception as e:
                print(f"  DEBUG: EXCEPTION in fetch_kpi_batch: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*70)
        print("DEBUG: PART 2 COMPLETED - KPI batch fetching done")
        print(f"DEBUG: Stats so far - total={self.stats['total']}, successful={self.stats['successful']}, failed={self.stats['failed']}")
        print("="*70)

        # =================================================================
        # PART 3: Multi-stock array endpoints
        # =================================================================
        print(f"\n--- Multi-Stock Array Endpoints ---")
        inst_list = ",".join(str(i) for i in instrument_ids)

        array_endpoints = [
            ("stockprices_array", "/instruments/stockprices",
             {"instList": inst_list, "maxcount": 50}),
            ("reports_array", "/instruments/reports",
             {"instList": inst_list}),
        ]

        for name, endpoint, params in array_endpoints:
            print(f"\n  [{self.stats['total']+1}] Fetching {name}...")
            self.fetch_endpoint(name, endpoint, params)

    def fetch_global_endpoints(self, instrument_ids: List[int] = None):
        """Fetch global endpoints (not stock-specific)."""
        print("\n" + "="*70)
        print("FETCHING GLOBAL ENDPOINTS")
        print("="*70)

        yesterday = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # Build instrument list for holdings endpoints
        inst_list = ",".join(str(i) for i in instrument_ids) if instrument_ids else None

        global_endpoints = [
            ("stockprices_last", "/instruments/stockprices/last", {}),
            ("stockprices_by_date", "/instruments/stockprices/date", {"date": yesterday}),
            ("calendar_report", "/instruments/report/calendar/", {}),
            ("calendar_dividend", "/instruments/dividend/calendar/", {}),
            ("stocksplits", "/instruments/stocksplits", {}),
        ]

        # Holdings endpoints - can optionally filter by instrument list
        if inst_list:
            global_endpoints.extend([
                ("holdings_insider", "/holdings/insider", {"instList": inst_list}),
                ("holdings_shorts", "/holdings/shorts", {"instList": inst_list}),
                ("holdings_buyback", "/holdings/buyback", {"instList": inst_list}),
            ])
        else:
            # Fetch all holdings data if no instruments specified
            global_endpoints.extend([
                ("holdings_insider", "/holdings/insider", {}),
                ("holdings_shorts", "/holdings/shorts", {}),
                ("holdings_buyback", "/holdings/buyback", {}),
            ])

        for name, endpoint, params in global_endpoints:
            print(f"\n[{self.stats['total']+1}] Fetching {name}...")
            self.fetch_endpoint(name, endpoint, params)

    def run(self, instrument_ids: List[int], fetch_nordics: bool = False):
        """Main execution method."""
        try:
            self.ensure_results_dir()
            self.connect_db()

            # If nordics flag is set, get all nordic instruments
            if fetch_nordics:
                # First fetch instruments metadata to get the list
                print("\n" + "="*70)
                print("FETCHING INSTRUMENTS METADATA FOR NORDICS")
                print("="*70)
                print(f"\n[1] Fetching instruments...")
                self.fetch_endpoint("instruments", "/instruments", {})

                # Get Nordic instruments
                instrument_ids = self.get_nordic_instruments()

                if not instrument_ids:
                    print("✗ No Nordic instruments found. Exiting.")
                    return

            self.create_fetch_log(instrument_ids)

            # Fetch all metadata (skip instruments if already fetched for nordics)
            if not fetch_nordics:
                self.fetch_all_metadata()
            else:
                # Fetch remaining metadata (instruments already fetched)
                print("\n" + "="*70)
                print("FETCHING REMAINING METADATA")
                print("="*70)
                metadata_endpoints = [
                    ("instruments_updated", "/instruments/updated", {}),
                    ("markets", "/markets", {}),
                    ("branches", "/branches", {}),
                    ("sectors", "/sectors", {}),
                    ("countries", "/countries", {}),
                    ("translation_metadata", "/translationmetadata", {}),
                    ("kpi_metadata", "/instruments/kpis/metadata", {}),
                    ("kpi_metadata_updated", "/instruments/kpis/updated", {}),
                ]
                for name, endpoint, params in metadata_endpoints:
                    print(f"\n[{self.stats['total']+1}] Fetching {name}...")
                    self.fetch_endpoint(name, endpoint, params)

            # Fetch stock-specific data
            self.fetch_stock_data(instrument_ids)

            # Fetch global endpoints (pass instrument_ids for holdings filtering)
            self.fetch_global_endpoints(instrument_ids)

            # Update fetch log with final stats
            self.update_fetch_log()

            # Print summary
            self.print_summary()

        finally:
            self.close_db()

    def print_summary(self):
        """Print execution summary."""
        duration = time.time() - self.stats['start_time']
        rate_stats = self.rate_limiter.get_stats()

        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        print(f"Total endpoints: {self.stats['total']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped (already exists): {self.stats['skipped']}")
        print(f"Duration: {duration:.2f} seconds")

        if self.stats['successful'] > 0:
            avg_rate = self.stats['successful'] / duration if duration > 0 else 0
            print(f"Average rate: {avg_rate:.2f} calls/second")

        print(f"Rate limit: {rate_stats['max_calls']} calls per {rate_stats['time_window']} seconds")
        print(f"Results saved to: {RESULTS_DIR}/")

        if self.db_enabled:
            print(f"Database: {self.db_config['database']} on {self.db_config['host']}")
            print(f"Fetch log ID: {self.fetch_log_id}")
            if self.skip_existing:
                print(f"Skip existing: Enabled (within 24 hours)")
            else:
                print(f"Skip existing: Disabled (fetch all)")
        else:
            print(f"Database: Disabled (JSON files only)")

        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch Borsdata API data and store in PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('api_key', nargs='?',
                       default=os.getenv('BORSDATA_API_KEY'),
                       help='Borsdata API key (or set BORSDATA_API_KEY env var)')

    parser.add_argument('--db-host',
                       default=os.getenv('DB_HOST', 'localhost'),
                       help='PostgreSQL host (default: localhost)')

    parser.add_argument('--db-name',
                       default=os.getenv('DB_NAME', 'borsdata'),
                       help='PostgreSQL database name (default: borsdata)')

    parser.add_argument('--db-user',
                       default=os.getenv('DB_USER', 'postgres'),
                       help='PostgreSQL user (default: postgres)')

    parser.add_argument('--db-password',
                       default=os.getenv('DB_PASSWORD'),
                       help='PostgreSQL password (or set DB_PASSWORD env var)')

    parser.add_argument('--db-port',
                       type=int,
                       default=int(os.getenv('DB_PORT', '5432')),
                       help='PostgreSQL port (default: 5432)')

    parser.add_argument('--instruments',
                       default='3,19,199,750',
                       help='Comma-separated list of instrument IDs (default: 3,19,199,750 = ABB,Atlas Copco,SEB,Securitas). Ignored if --nordics is used.')

    parser.add_argument('--nordics',
                       action='store_true',
                       help='Fetch all instruments from Nordic countries (Sweden, Norway, Finland, Denmark). Overrides --instruments.')

    parser.add_argument('--no-db',
                       action='store_true',
                       help='Skip database storage and only save to JSON files')

    parser.add_argument('--force-refetch',
                       action='store_true',
                       help='Force refetch all data even if it already exists in database')

    args = parser.parse_args()

    # Validate required parameters
    if not args.api_key:
        print("Error: API key is required. Provide it as argument or set BORSDATA_API_KEY env var")
        print("\nUsage: python fetch_and_store.py YOUR_API_KEY")
        print("       python fetch_and_store.py YOUR_API_KEY --no-db  # JSON files only")
        print("       python fetch_and_store.py YOUR_API_KEY --db-password yourpass  # With database")
        sys.exit(1)

    # Parse instrument IDs (will be overridden if --nordics is used)
    instrument_ids = []
    if not args.nordics:
        instrument_ids = [int(x.strip()) for x in args.instruments.split(',')]

    # Database configuration
    db_config = None    
    if not args.no_db:
        if not args.db_password:
            print("Warning: No database password provided. Running in JSON-only mode.")
            print("Use --db-password to enable database storage, or --no-db to suppress this warning.")
            print()
        else:            
            db_config = {
                'host': args.db_host,
                'database': args.db_name,
                'user': args.db_user,
                'password': args.db_password,
                'port': args.db_port
            }

    print("="*70)
    print("BORSDATA API FETCHER & POSTGRESQL STORER")
    print("="*70)
    print(f"API Key: {args.api_key[:8]}...{args.api_key[-4:]}")
    if db_config:
        print(f"Database: {db_config['database']} @ {db_config['host']}:{db_config['port']}")
        print(f"Skip existing: {'No (force refetch)' if args.force_refetch else 'Yes (within 24 hours)'}")
    else:
        print(f"Database: Disabled (JSON files only)")

    if args.nordics:
        print(f"Mode: Fetch all Nordic instruments (Sweden, Norway, Finland, Denmark)")
        print(f"Rate Limit: 100 calls per 10 seconds")
    else:
        print(f"Instruments: {instrument_ids}")
        print(f"Rate Limit: 100 calls per 10 seconds")

    print(f"Time: {datetime.now().isoformat()}")
    print("="*70)

    # Create and run fetcher
    skip_existing = not args.force_refetch
    fetcher = BorsdataFetcher(args.api_key, db_config, skip_existing=skip_existing)
    fetcher.run(instrument_ids, fetch_nordics=args.nordics)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
