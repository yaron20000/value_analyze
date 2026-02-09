#!/usr/bin/env python3
"""
Borsdata Fetch Validator
========================
Validates that all expected data was fetched from Borsdata API and stored
in the PostgreSQL database. Checks metadata, KPIs, instruments, stock data,
and global endpoints for completeness and data quality.

Usage:
    python validate_fetch.py
    python validate_fetch.py --db-password yourpass
    python validate_fetch.py --hours 48         # Check data within last 48 hours
    python validate_fetch.py --verbose           # Show per-instrument details
    python validate_fetch.py --fix-summary       # Show commands to re-fetch missing data

Requirements:
    pip install psycopg2-binary python-dotenv
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Expected data definitions (must stay in sync with fetch_and_store.py)
# ---------------------------------------------------------------------------

EXPECTED_METADATA_ENDPOINTS = [
    "instruments",
    "instruments_updated",
    "markets",
    "branches",
    "sectors",
    "countries",
    "translation_metadata",
    "kpi_metadata",
    "kpi_metadata_updated",
]

EXPECTED_KPI_LIST = [
    # (kpi_id, short_name, display_name)
    # Tier 1 - Valuation
    (1, "dividend_yield", "Dividend Yield"),
    (2, "pe", "P/E"),
    (3, "ps", "P/S"),
    (4, "pb", "P/B"),
    (10, "ev_ebit", "EV/EBIT"),
    (11, "ev_ebitda", "EV/EBITDA"),
    (13, "ev_fcf", "EV/FCF"),
    (15, "ev_s", "EV/S"),
    # Tier 1 - Profitability & Margins
    (28, "gross_margin", "Gross Margin"),
    (29, "operating_margin", "Operating Margin"),
    (30, "profit_margin", "Profit Margin"),
    (31, "fcf_margin", "FCF Margin"),
    (32, "ebitda_margin", "EBITDA Margin"),
    (51, "ocf_margin", "OCF Margin"),
    # Tier 1 - Growth
    (94, "revenue_growth", "Revenue Growth"),
    (96, "ebit_growth", "EBIT Growth"),
    (97, "earnings_growth", "Earnings Growth"),
    (98, "dividend_growth", "Dividend Growth"),
    (99, "book_value_growth", "Book Value Growth"),
    (100, "assets_growth", "Assets Growth"),
    # Tier 1 - Returns
    (33, "roe", "Return on Equity"),
    (34, "roa", "Return on Assets"),
    (36, "roc", "Return on Capital"),
    (37, "roic", "Return on Invested Capital"),
    # Tier 1 - Financial Health
    (39, "equity_ratio", "Equity Ratio"),
    (40, "debt_to_equity", "Debt to Equity"),
    (41, "net_debt_pct", "Net Debt %"),
    (42, "net_debt_ebitda", "Net Debt/EBITDA"),
    (44, "current_ratio", "Current Ratio"),
    (46, "cash_pct", "Cash %"),
    # Tier 1 - Size
    (49, "enterprise_value", "Enterprise Value"),
    (50, "market_cap", "Market Cap"),
    (53, "revenue", "Revenue"),
    (56, "earnings", "Earnings"),
    (63, "fcf", "Free Cash Flow"),
    # Tier 2 - Per-Share
    (5, "revenue_per_share", "Revenue per Share"),
    (6, "eps", "Earnings per Share"),
    (7, "dividend_per_share", "Dividend per Share"),
    (8, "book_value_per_share", "Book Value per Share"),
    (23, "fcf_per_share", "FCF per Share"),
    (68, "ocf_per_share", "OCF per Share"),
    # Tier 2 - Cash Flow
    (24, "fcf_margin_pct", "FCF Margin %"),
    (27, "earnings_fcf", "Earnings/FCF"),
    (62, "ocf", "Operating Cash Flow"),
    (64, "capex", "Capex"),
    # Tier 3 - Additional Valuation
    (19, "peg", "PEG Ratio"),
    (20, "dividend_payout", "Dividend Payout %"),
    # Tier 3 - Additional Absolute
    (54, "ebitda", "EBITDA"),
    (57, "total_assets", "Total Assets"),
    (58, "total_equity", "Total Equity"),
    (60, "net_debt", "Net Debt"),
    (61, "num_shares", "Number of Shares"),
]

EXPECTED_PER_INSTRUMENT_PREFIXES = [
    "stockprices",
    "reports_year",
    "reports_r12",
    "reports_quarter",
    "reports_all",
]

EXPECTED_GLOBAL_ENDPOINTS = [
    "stockprices_last",
    "stockprices_by_date",
    "calendar_report",
    "calendar_dividend",
    "stocksplits",
    "holdings_insider",
    "holdings_shorts",
    "holdings_buyback",
]

EXPECTED_ARRAY_ENDPOINTS = [
    "stockprices_array",
    "reports_array",
]

NORDIC_COUNTRY_IDS = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

class ValidationResult:
    """Holds the result of a single validation check."""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.total = 0
        self.present = 0
        self.missing: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def pct(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.present / self.total) * 100

    def mark_missing(self, item: str):
        self.missing.append(item)
        self.passed = False

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


class BorsdataValidator:
    """Validates completeness of Borsdata data in the database."""

    def __init__(self, db_config: dict, hours: int = 24, verbose: bool = False):
        self.db_config = db_config
        self.hours = hours
        self.verbose = verbose
        self.conn = None
        self.cursor = None
        self.cutoff = datetime.now() - timedelta(hours=hours)
        self.results: List[ValidationResult] = []
        self.instrument_ids: List[int] = []

    # -- DB helpers ----------------------------------------------------------

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.db_config["host"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            port=self.db_config.get("port", 5432),
        )
        self.cursor = self.conn.cursor()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def query_endpoints(self, since: datetime = None) -> Dict[str, dict]:
        """Return a dict of endpoint_name -> {count, latest_ts, has_data}."""
        ts = since or self.cutoff
        self.cursor.execute("""
            SELECT endpoint_name,
                   COUNT(*) AS cnt,
                   MAX(fetch_timestamp) AS latest,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) AS ok,
                   SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failed
            FROM api_raw_data
            WHERE fetch_timestamp >= %s
            GROUP BY endpoint_name
            ORDER BY endpoint_name
        """, (ts,))
        rows = self.cursor.fetchall()
        return {
            r[0]: {"count": r[1], "latest": r[2], "ok": r[3], "failed": r[4]}
            for r in rows
        }

    def get_instrument_ids_from_db(self) -> List[int]:
        """Get the list of Nordic instrument IDs from the stored instruments data."""
        self.cursor.execute("""
            SELECT raw_data->'instruments' AS instruments
            FROM api_raw_data
            WHERE endpoint_name = 'instruments'
              AND success = true
            ORDER BY fetch_timestamp DESC
            LIMIT 1
        """)
        row = self.cursor.fetchone()
        if not row or not row[0]:
            return []
        instruments = row[0]
        return [
            inst["insId"]
            for inst in instruments
            if inst.get("countryId") in NORDIC_COUNTRY_IDS
        ]

    def get_fetched_instrument_ids(self) -> Set[int]:
        """Get the set of instrument IDs that have at least one stock-specific record."""
        self.cursor.execute("""
            SELECT DISTINCT instrument_id
            FROM api_raw_data
            WHERE instrument_id IS NOT NULL
              AND success = true
              AND fetch_timestamp >= %s
        """, (self.cutoff,))
        return {r[0] for r in self.cursor.fetchall()}

    # -- Checks --------------------------------------------------------------

    def check_metadata(self) -> ValidationResult:
        """Check that all metadata endpoints have been fetched."""
        res = ValidationResult("Metadata endpoints")
        endpoints = self.query_endpoints()
        res.total = len(EXPECTED_METADATA_ENDPOINTS)

        for ep in EXPECTED_METADATA_ENDPOINTS:
            info = endpoints.get(ep)
            if info and info["ok"] > 0:
                res.present += 1
            else:
                res.mark_missing(ep)

        return res

    def check_kpi_batches(self) -> ValidationResult:
        """Check that all 52 KPI batch endpoints have data."""
        res = ValidationResult("KPI batch endpoints")
        endpoints = self.query_endpoints()
        res.total = len(EXPECTED_KPI_LIST)

        for kpi_id, kpi_name, display_name in EXPECTED_KPI_LIST:
            batch_name = f"kpi_{kpi_name}_batch"
            info = endpoints.get(batch_name)
            if info and info["ok"] > 0:
                res.present += 1
            else:
                res.mark_missing(f"{batch_name} ({display_name}, ID={kpi_id})")

        return res

    def check_kpi_instrument_coverage(self) -> ValidationResult:
        """Check that KPI batch records contain data for the expected instruments.

        Inspects the kpisList array inside each batch JSONB to verify how many
        instruments are covered per KPI, without relying on per-instrument rows.
        """
        res = ValidationResult("KPI instrument coverage (inside batch data)")

        if not self.instrument_ids:
            res.add_warning("No instrument IDs available - skipping KPI coverage check")
            return res

        total_instruments = len(self.instrument_ids)
        expected_set = set(self.instrument_ids)
        res.total = len(EXPECTED_KPI_LIST)

        for _kpi_id, kpi_name, display_name in EXPECTED_KPI_LIST:
            batch_name = f"kpi_{kpi_name}_batch"
            self.cursor.execute("""
                SELECT raw_data->'kpisList' AS kpis_list
                FROM api_raw_data
                WHERE endpoint_name = %s
                  AND success = true
                  AND fetch_timestamp >= %s
                ORDER BY fetch_timestamp DESC
                LIMIT 1
            """, (batch_name, self.cutoff))
            row = self.cursor.fetchone()

            if not row or not row[0]:
                res.mark_missing(f"{display_name}: no batch data found")
                continue

            kpis_list = row[0]
            found_ids = {entry.get("instrument") for entry in kpis_list if entry.get("instrument") is not None}
            covered = found_ids & expected_set
            coverage_pct = (len(covered) / total_instruments) * 100

            if len(covered) == 0:
                res.mark_missing(f"{display_name}: 0/{total_instruments} instruments in batch")
            else:
                res.present += 1
                if coverage_pct < 50:
                    res.add_warning(
                        f"{display_name}: only {len(covered)}/{total_instruments} "
                        f"instruments ({coverage_pct:.0f}%)"
                    )

        return res

    def check_stock_data(self) -> ValidationResult:
        """Check per-instrument stock data (prices, reports)."""
        res = ValidationResult("Per-instrument stock data")

        if not self.instrument_ids:
            res.add_warning("No instrument IDs available - skipping stock data check")
            return res

        # Query all per-instrument endpoints
        self.cursor.execute("""
            SELECT endpoint_name, instrument_id
            FROM api_raw_data
            WHERE instrument_id IS NOT NULL
              AND success = true
              AND fetch_timestamp >= %s
              AND (   endpoint_name LIKE 'stockprices_%%'
                   OR endpoint_name LIKE 'reports_year_%%'
                   OR endpoint_name LIKE 'reports_r12_%%'
                   OR endpoint_name LIKE 'reports_quarter_%%'
                   OR endpoint_name LIKE 'reports_all_%%')
              AND endpoint_name NOT LIKE '%%_array'
              AND endpoint_name NOT LIKE '%%_last'
              AND endpoint_name NOT LIKE '%%_by_date'
        """, (self.cutoff,))
        rows = self.cursor.fetchall()

        # Build a set of (prefix, instrument_id) tuples
        found: Set[Tuple[str, int]] = set()
        for ep_name, inst_id in rows:
            for prefix in EXPECTED_PER_INSTRUMENT_PREFIXES:
                if ep_name.startswith(prefix + "_"):
                    found.add((prefix, inst_id))
                    break

        total_instruments = len(self.instrument_ids)
        res.total = len(EXPECTED_PER_INSTRUMENT_PREFIXES) * total_instruments

        # Count present
        missing_by_prefix: Dict[str, List[int]] = {p: [] for p in EXPECTED_PER_INSTRUMENT_PREFIXES}
        for prefix in EXPECTED_PER_INSTRUMENT_PREFIXES:
            for inst_id in self.instrument_ids:
                if (prefix, inst_id) in found:
                    res.present += 1
                else:
                    missing_by_prefix[prefix].append(inst_id)

        # Summarize missing
        for prefix, missing_ids in missing_by_prefix.items():
            if missing_ids:
                n = len(missing_ids)
                pct = (n / total_instruments) * 100
                sample = missing_ids[:5]
                extra = f" ... and {n - 5} more" if n > 5 else ""
                res.mark_missing(
                    f"{prefix}: missing for {n}/{total_instruments} instruments "
                    f"({pct:.0f}%) [e.g. {sample}{extra}]"
                )

        return res

    def check_global_endpoints(self) -> ValidationResult:
        """Check that all global endpoints have data."""
        res = ValidationResult("Global endpoints")
        endpoints = self.query_endpoints()
        res.total = len(EXPECTED_GLOBAL_ENDPOINTS)

        for ep in EXPECTED_GLOBAL_ENDPOINTS:
            info = endpoints.get(ep)
            if info and info["ok"] > 0:
                res.present += 1
            else:
                res.mark_missing(ep)

        return res

    def check_array_endpoints(self) -> ValidationResult:
        """Check multi-stock array endpoints."""
        res = ValidationResult("Array endpoints")
        endpoints = self.query_endpoints()
        res.total = len(EXPECTED_ARRAY_ENDPOINTS)

        for ep in EXPECTED_ARRAY_ENDPOINTS:
            info = endpoints.get(ep)
            if info and info["ok"] > 0:
                res.present += 1
            else:
                res.mark_missing(ep)

        return res

    def check_data_quality(self) -> ValidationResult:
        """Check for data quality issues: NULL raw_data, failed records, stale data."""
        res = ValidationResult("Data quality")

        # 1. Successful records with NULL raw_data
        self.cursor.execute("""
            SELECT COUNT(*) FROM api_raw_data
            WHERE success = true AND raw_data IS NULL
              AND fetch_timestamp >= %s
        """, (self.cutoff,))
        null_data = self.cursor.fetchone()[0]
        if null_data > 0:
            res.add_error(f"{null_data} successful records have NULL raw_data")

        # 2. Failed records
        self.cursor.execute("""
            SELECT endpoint_name, error_message
            FROM api_raw_data
            WHERE success = false
              AND fetch_timestamp >= %s
            ORDER BY fetch_timestamp DESC
        """, (self.cutoff,))
        failures = self.cursor.fetchall()
        if failures:
            res.add_warning(f"{len(failures)} failed API call(s) in the last {self.hours} hours")
            # Group by error type
            error_groups: Dict[str, int] = {}
            for ep, err in failures:
                key = (err or "unknown")[:80]
                error_groups[key] = error_groups.get(key, 0) + 1
            for err_msg, count in sorted(error_groups.items(), key=lambda x: -x[1]):
                res.add_warning(f"  {count}x: {err_msg}")

        # 3. Data freshness - most recent fetch
        self.cursor.execute("""
            SELECT MAX(fetch_timestamp) FROM api_raw_data WHERE success = true
        """)
        latest = self.cursor.fetchone()[0]
        if latest:
            age = datetime.now() - latest
            if age > timedelta(hours=self.hours):
                res.add_warning(
                    f"Most recent successful fetch is {age.total_seconds()/3600:.1f}h old "
                    f"(threshold: {self.hours}h)"
                )

        # 4. Total record counts
        self.cursor.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN success THEN 1 ELSE 0 END),
                   SUM(CASE WHEN NOT success THEN 1 ELSE 0 END)
            FROM api_raw_data
            WHERE fetch_timestamp >= %s
        """, (self.cutoff,))
        total, ok, failed = self.cursor.fetchone()
        res.total = 4  # quality checks count
        res.present = res.total - len(res.errors)

        if not res.errors:
            res.passed = True

        res.add_warning(f"Total records in window: {total} (ok={ok}, failed={failed})")

        return res

    def check_fetch_log(self) -> ValidationResult:
        """Check the fetch log for the latest run."""
        res = ValidationResult("Fetch log")

        self.cursor.execute("""
            SELECT id, run_timestamp, total_endpoints, successful_endpoints,
                   failed_endpoints, duration_seconds, instruments_fetched
            FROM api_fetch_log
            ORDER BY run_timestamp DESC
            LIMIT 1
        """)
        row = self.cursor.fetchone()
        if not row:
            res.add_error("No fetch log entries found")
            res.total = 1
            return res

        log_id, ts, total, ok, failed, duration, instruments = row
        res.total = 1
        res.present = 1

        if failed and failed > 0:
            res.add_warning(f"Last run (ID={log_id}): {failed}/{total} endpoints failed")

        age = datetime.now() - ts
        if age > timedelta(hours=self.hours):
            res.add_warning(
                f"Last fetch run was {age.total_seconds()/3600:.1f}h ago "
                f"(ID={log_id}, duration={duration}s)"
            )
        else:
            res.add_warning(
                f"Last fetch run: ID={log_id}, {age.total_seconds()/3600:.1f}h ago, "
                f"duration={duration}s, {ok}/{total} successful"
            )

        num_inst = len(instruments) if instruments else 0
        res.add_warning(f"Instruments in last run: {num_inst}")

        return res

    # -- Run all checks ------------------------------------------------------

    def run(self):
        """Execute all validation checks and print report."""
        try:
            self.connect()
            print(f"Connected to {self.db_config['database']} @ {self.db_config['host']}")
            print(f"Checking data fetched within the last {self.hours} hours "
                  f"(since {self.cutoff.strftime('%Y-%m-%d %H:%M')})")
            print()

            # Determine expected instrument list
            self.instrument_ids = self.get_instrument_ids_from_db()
            if self.instrument_ids:
                print(f"Found {len(self.instrument_ids)} Nordic instruments in database")
            else:
                print("WARNING: Could not load instrument list from database")
            print()

            # Run all checks
            checks = [
                self.check_fetch_log,
                self.check_metadata,
                self.check_kpi_batches,
                self.check_kpi_instrument_coverage,
                self.check_stock_data,
                self.check_global_endpoints,
                self.check_array_endpoints,
                self.check_data_quality,
            ]

            for check_fn in checks:
                result = check_fn()
                self.results.append(result)

            self.print_report()

        finally:
            self.close()

    # -- Report --------------------------------------------------------------

    def print_report(self):
        """Print the validation report."""
        print("=" * 70)
        print("BORSDATA FETCH VALIDATION REPORT")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Window: last {self.hours} hours")
        if self.instrument_ids:
            print(f"Expected instruments: {len(self.instrument_ids)}")
        print("=" * 70)

        all_passed = True

        for res in self.results:
            status = "PASS" if res.passed else "FAIL"
            if not res.passed:
                all_passed = False

            pct_str = f" ({res.pct:.0f}%)" if res.total > 0 else ""
            print(f"\n[{status}] {res.name}: {res.present}/{res.total}{pct_str}")

            for err in res.errors:
                print(f"  ERROR: {err}")

            if res.missing and self.verbose:
                for m in res.missing:
                    print(f"  MISSING: {m}")
            elif res.missing:
                print(f"  MISSING: {len(res.missing)} item(s) "
                      f"(use --verbose for details)")

            for w in res.warnings:
                print(f"  INFO: {w}")

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print("\n" + "=" * 70)
        if all_passed:
            print(f"RESULT: ALL CHECKS PASSED ({passed}/{total})")
        else:
            failed_names = [r.name for r in self.results if not r.passed]
            print(f"RESULT: {total - passed} CHECK(S) FAILED ({passed}/{total} passed)")
            print(f"  Failed: {', '.join(failed_names)}")
        print("=" * 70)

        return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Borsdata fetch completeness in PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "borsdata"))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    parser.add_argument("--hours", type=int, default=24,
                        help="Look-back window in hours (default: 24)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-item details for missing data")

    args = parser.parse_args()

    if not args.db_password:
        print("Error: Database password required. Use --db-password or set DB_PASSWORD env var.")
        sys.exit(1)

    db_config = {
        "host": args.db_host,
        "database": args.db_name,
        "user": args.db_user,
        "password": args.db_password,
        "port": args.db_port,
    }

    validator = BorsdataValidator(db_config, hours=args.hours, verbose=args.verbose)
    validator.run()


if __name__ == "__main__":
    main()
