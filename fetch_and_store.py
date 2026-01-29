#!/usr/bin/env python3
"""
Borsdata API Data Fetcher and PostgreSQL Storer
================================================
This script fetches data from Borsdata API endpoints and stores it in PostgreSQL.

Features:
- Fetches all Instrument Meta APIs
- Fetches stock-specific APIs for specified stocks
- Saves JSON responses to results/ directory before DB insertion
- Stores raw data in PostgreSQL for future AI processing (optional)
- Handles API rate limiting and errors gracefully
- Can run without database (JSON files only)
- Automatically skips re-fetching data that already exists (within 24 hours)
- Prevents duplicate data insertion using database constraints

Usage:
    # JSON files only (no database)
    python fetch_and_store.py YOUR_API_KEY --no-db

    # With PostgreSQL database (skips existing data by default)
    python fetch_and_store.py YOUR_API_KEY --db-password yourpass

    # Force refetch all data even if it exists
    python fetch_and_store.py YOUR_API_KEY --db-password yourpass --force-refetch

    # With environment variables
    export BORSDATA_API_KEY=your_key
    export DB_PASSWORD=yourpass
    python fetch_and_store.py

    # Full database configuration
    python fetch_and_store.py YOUR_API_KEY --db-host localhost --db-name borsdata --db-user postgres --db-password yourpass

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
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

BASE_URL = "https://apiservice.borsdata.se/v1"
RESULTS_DIR = "results"


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
        """Save the result to PostgreSQL database using upsert to prevent duplicates."""
        if not self.db_enabled:
            return

        try:
            # Use ON CONFLICT to update if a similar record exists from today
            # This prevents duplicate entries for the same endpoint/instrument on the same day
            self.cursor.execute("""
                INSERT INTO api_raw_data
                (endpoint_name, endpoint_path, instrument_id, kpi_name, fetch_timestamp,
                 success, error_message, raw_data, request_params)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                name,
                endpoint_path,
                instrument_id,
                kpi_name,
                datetime.now(),
                data is not None,
                error,
                Json(data) if data else None,
                Json(params) if params else None
            ))

            # Check if the insert actually happened
            if self.cursor.rowcount == 0:
                print(f"    ℹ Data already exists in DB (skipped)")

            self.conn.commit()
        except Exception as e:
            print(f"    ✗ Failed to save to DB: {e}")
            self.conn.rollback()

    def make_request(self, endpoint: str, params: dict = None) -> Tuple[Optional[dict], Optional[str]]:
        """Make a request to the Borsdata API and return the JSON response and any error."""
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
            time.sleep(0.2)  # Rate limiting

    def fetch_stock_data(self, instrument_ids: List[int]):
        """Fetch stock-specific data for given instrument IDs."""
        print("\n" + "="*70)
        print(f"FETCHING STOCK-SPECIFIC DATA FOR INSTRUMENTS: {instrument_ids}")
        print("="*70)

        yesterday = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        for inst_id in instrument_ids:
            print(f"\n--- Instrument ID: {inst_id} ---")

            # Stock prices
            endpoints = [
                (f"stockprices_{inst_id}", f"/instruments/{inst_id}/stockprices",
                 {"maxcount": 100}, inst_id),

                # Reports
                (f"reports_year_{inst_id}", f"/instruments/{inst_id}/reports/year",
                 {"maxcount": 10}, inst_id),
                (f"reports_r12_{inst_id}", f"/instruments/{inst_id}/reports/r12",
                 {"maxcount": 10}, inst_id),
                (f"reports_quarter_{inst_id}", f"/instruments/{inst_id}/reports/quarter",
                 {"maxcount": 20}, inst_id),
                (f"reports_all_{inst_id}", f"/instruments/{inst_id}/reports",
                 {"maxcount": 20}, inst_id),
            ]

            # WORKING KPIs ONLY - 52 Total (removed 28 failing KPIs)
            # See FAILING_KPIS_ANALYSIS.md for details on what was removed and why

            # Tier 1 KPIs - All 35 Work Perfectly (100% success rate)
            # Valuation Metrics (8 KPIs)
            kpi_endpoints = [
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

                # REMOVED: Quality Scores (0/3 working)
                # - F-Score, Magic Formula, Earnings Stability all fail with Error 400
                # These are available from holdings endpoints or need to be calculated

                # REMOVED: Technical Indicators (0/5 working)
                # - Performance, Total Return, RSI, Volatility, Volume all fail
                # These require daily/real-time data, not annual aggregates
                # Can be calculated from stockprices_*.json files we already fetch

                # REMOVED: Insider/Ownership (0/7 working)
                # - All insider and shareholder KPIs fail via /kpis endpoint
                # This data is available in holdings_insider.json we already fetch

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

                # REMOVED: Additional Technical (0/4 working)
                # - MA200 Rank, MA(50)/MA(200), Volatility Std Dev, Volume Trend
                # Same issue as technical indicators above

                # REMOVED: Short Selling (0/4 working)
                # - All short selling KPIs fail via /kpis endpoint
                # This data is available in holdings_shorts.json we already fetch

                # REMOVED: Buybacks (0/3 working)
                # - All buyback KPIs fail via /kpis endpoint
                # This data is available in holdings_buyback.json we already fetch

                # REMOVED: Additional Quality (0/2 working)
                # - Graham Strategy, Cash Flow Stability
                # May be premium-only or need to be calculated
            ]

            # Note: We already fetch insider, short selling, and buyback data separately
            # via the /holdings endpoints. See fetch_global_endpoints() method.

            # Add KPI endpoints for this instrument
            for kpi_id, kpi_name, kpi_display_name in kpi_endpoints:
                endpoints.append((
                    f"kpi_{kpi_name}_{inst_id}",
                    f"/instruments/{inst_id}/kpis/{kpi_id}/year/mean/history",
                    {},
                    inst_id,
                    kpi_display_name  # Add the display name for JSON/DB
                ))

            # Fetch regular endpoints (without kpi_name)
            for item in endpoints:
                if len(item) == 4:
                    # Regular endpoint (name, endpoint, params, iid)
                    name, endpoint, params, iid = item
                    print(f"\n  [{self.stats['total']+1}] Fetching {name}...")
                    self.fetch_endpoint(name, endpoint, params, iid)
                elif len(item) == 5:
                    # KPI endpoint (name, endpoint, params, iid, kpi_display_name)
                    name, endpoint, params, iid, kpi_display_name = item
                    print(f"\n  [{self.stats['total']+1}] Fetching {name}...")
                    self.fetch_endpoint(name, endpoint, params, iid, kpi_name=kpi_display_name)
                time.sleep(0.2)  # Rate limiting

        # Fetch array-based endpoints (multiple stocks at once)
        print(f"\n--- Multi-Stock Endpoints ---")
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
            time.sleep(0.2)  # Rate limiting

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
            time.sleep(0.2)  # Rate limiting

    def run(self, instrument_ids: List[int]):
        """Main execution method."""
        try:
            self.ensure_results_dir()
            self.connect_db()
            self.create_fetch_log(instrument_ids)

            # Fetch all metadata
            self.fetch_all_metadata()

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

        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        print(f"Total endpoints: {self.stats['total']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped (already exists): {self.stats['skipped']}")
        print(f"Duration: {duration:.2f} seconds")
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
                       help='Comma-separated list of instrument IDs (default: 3,19,199,750 = ABB,Atlas Copco,SEB,Securitas)')

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

    # Parse instrument IDs
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
    print(f"Instruments: {instrument_ids}")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*70)

    # Create and run fetcher
    skip_existing = not args.force_refetch
    fetcher = BorsdataFetcher(args.api_key, db_config, skip_existing=skip_existing)
    fetcher.run(instrument_ids)

    print("\n✓ Done!")


if __name__ == "__main__":
    # Rate limit 100 calls in 10 seconds
    main()
