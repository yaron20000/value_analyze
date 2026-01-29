#!/usr/bin/env python3
"""
Transform Raw API Data to ML-Ready Format
==========================================
Converts nested JSONB data from api_raw_data table into flattened ML tables.

This script:
1. Extracts KPI time series from JSONB and pivots into wide format
2. Calculates target variables (future returns, dividend growth)
3. Populates ml_features, ml_stock_prices, and ml_targets tables

Usage:
    python transform_to_ml.py --db-password yourpass
    python transform_to_ml.py --db-password yourpass --instrument-id 199
"""

import psycopg2
from psycopg2.extras import execute_values
import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()


class MLTransformer:
    """Transforms raw API data into ML-ready format."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

        # KPI ID to column name mapping (from your fetch_and_store.py)
        self.kpi_mapping = {
            # Valuation (13 KPIs)
            1: 'pe_ratio',
            2: 'ps_ratio',
            3: 'pb_ratio',
            4: 'ev_ebitda',
            19: 'peg_ratio',
            22: 'ev_sales',

            # Profitability (11 KPIs)
            10: 'roe',
            11: 'roi',
            12: 'roa',
            13: 'ebitda_margin',
            14: 'operating_margin',
            15: 'gross_margin',
            16: 'net_margin',

            # Financial Health (12 KPIs)
            29: 'debt_equity',
            30: 'equity_ratio',
            32: 'current_ratio',
            33: 'quick_ratio',
            34: 'interest_coverage',

            # Growth (4 KPIs)
            35: 'revenue_growth',
            36: 'earnings_growth',
            37: 'dividend_growth',

            # Per Share (5 KPIs)
            6: 'eps',
            7: 'dividend_per_share',
            8: 'book_value_per_share',
            23: 'fcf_per_share',
            68: 'ocf_per_share',

            # Cash Flow (4 KPIs)
            24: 'fcf_margin',
            27: 'earnings_fcf',
            62: 'ocf',
            64: 'capex',

            # Dividend (1 KPI)
            20: 'dividend_payout',

            # Absolute Metrics (7 KPIs)
            56: 'earnings',
            55: 'revenue',
            54: 'ebitda',
            57: 'total_assets',
            58: 'total_equity',
            60: 'net_debt',
            61: 'num_shares',
        }

    def connect(self):
        """Connect to PostgreSQL."""
        self.conn = psycopg2.connect(
            host=self.db_config['host'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config.get('port', 5432)
        )
        self.cursor = self.conn.cursor()
        print(f"✓ Connected to database: {self.db_config['database']}")

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def get_instrument_metadata(self) -> Dict[int, Dict]:
        """Fetch instrument metadata (name, sector, market, etc.)."""
        print("Fetching instrument metadata...")

        query = """
        SELECT
            raw_data->'instruments' as instruments
        FROM api_raw_data
        WHERE endpoint_name = 'instruments'
        ORDER BY fetch_timestamp DESC
        LIMIT 1
        """

        self.cursor.execute(query)
        result = self.cursor.fetchone()

        if not result:
            print("⚠ No instruments metadata found")
            return {}

        instruments = result[0]
        metadata = {}

        for inst in instruments:
            inst_id = inst.get('insId')
            metadata[inst_id] = {
                'name': inst.get('name'),
                'sector': inst.get('sector'),
                'market': inst.get('marketName'),
                'country': inst.get('countryName')
            }

        print(f"✓ Loaded metadata for {len(metadata)} instruments")
        return metadata

    def transform_kpi_data(self, instrument_id: int = None):
        """
        Extract KPI time series from JSONB and pivot into wide format.

        SQL Strategy:
        1. Unnest JSONB arrays to get (instrument, kpi, year, period, value)
        2. Pivot using CASE statements to create columns
        3. Group by (instrument, year, period)
        """
        print(f"Transforming KPI data{f' for instrument {instrument_id}' if instrument_id else ''}...")

        # Build WHERE clause
        where_clause = "WHERE kpi_name IS NOT NULL AND success = true"
        if instrument_id:
            where_clause += f" AND instrument_id = {instrument_id}"

        # First, get metadata
        metadata = self.get_instrument_metadata()

        # Clear existing data for this instrument
        if instrument_id:
            self.cursor.execute("DELETE FROM ml_features WHERE instrument_id = %s", (instrument_id,))
            print(f"  Cleared existing data for instrument {instrument_id}")

        # Build dynamic PIVOT query
        kpi_cases = []
        for kpi_id, col_name in self.kpi_mapping.items():
            kpi_cases.append(f"""
                MAX(CASE WHEN kpi_id = {kpi_id} THEN value END) as {col_name}
            """)

        pivot_sql = ", ".join(kpi_cases)

        query = f"""
        WITH unnested_kpis AS (
            SELECT
                instrument_id,
                (raw_data->>'kpiId')::integer as kpi_id,
                (jsonb_array_elements(raw_data->'values')->>'y')::integer as year,
                (jsonb_array_elements(raw_data->'values')->>'p')::integer as period,
                (jsonb_array_elements(raw_data->'values')->>'v')::numeric as value,
                fetch_timestamp
            FROM api_raw_data
            {where_clause}
        ),
        pivoted AS (
            SELECT
                instrument_id,
                year,
                period,
                {pivot_sql},
                MAX(fetch_timestamp) as fetch_date
            FROM unnested_kpis
            GROUP BY instrument_id, year, period
        )
        INSERT INTO ml_features (
            instrument_id, year, period,
            company_name, sector, market, country,
            pe_ratio, ps_ratio, pb_ratio, ev_ebitda, peg_ratio, ev_sales,
            roe, roi, roa, ebitda_margin, operating_margin, gross_margin, net_margin,
            debt_equity, equity_ratio, current_ratio, quick_ratio, interest_coverage,
            revenue_growth, earnings_growth, dividend_growth,
            eps, dividend_per_share, book_value_per_share, fcf_per_share, ocf_per_share,
            fcf_margin, earnings_fcf, ocf, capex,
            dividend_payout,
            earnings, revenue, ebitda, total_assets, total_equity, net_debt, num_shares,
            fetch_date
        )
        SELECT
            p.instrument_id,
            p.year,
            p.period,
            %s as company_name,
            %s as sector,
            %s as market,
            %s as country,
            p.pe_ratio, p.ps_ratio, p.pb_ratio, p.ev_ebitda, p.peg_ratio, p.ev_sales,
            p.roe, p.roi, p.roa, p.ebitda_margin, p.operating_margin, p.gross_margin, p.net_margin,
            p.debt_equity, p.equity_ratio, p.current_ratio, p.quick_ratio, p.interest_coverage,
            p.revenue_growth, p.earnings_growth, p.dividend_growth,
            p.eps, p.dividend_per_share, p.book_value_per_share, p.fcf_per_share, p.ocf_per_share,
            p.fcf_margin, p.earnings_fcf, p.ocf, p.capex,
            p.dividend_payout,
            p.earnings, p.revenue, p.ebitda, p.total_assets, p.total_equity, p.net_debt, p.num_shares,
            p.fetch_date
        FROM pivoted p
        ON CONFLICT (instrument_id, year, period) DO UPDATE SET
            pe_ratio = EXCLUDED.pe_ratio,
            ps_ratio = EXCLUDED.ps_ratio,
            pb_ratio = EXCLUDED.pb_ratio,
            -- ... (update all columns)
            fetch_date = EXCLUDED.fetch_date
        """

        # Execute for each instrument (to get metadata)
        if instrument_id:
            instruments = [instrument_id]
        else:
            # Get all instruments with KPI data
            self.cursor.execute("""
                SELECT DISTINCT instrument_id
                FROM api_raw_data
                WHERE kpi_name IS NOT NULL AND success = true
                ORDER BY instrument_id
            """)
            instruments = [row[0] for row in self.cursor.fetchall()]

        total_rows = 0
        for inst_id in instruments:
            meta = metadata.get(inst_id, {})
            self.cursor.execute(query.replace(where_clause,
                f"WHERE kpi_name IS NOT NULL AND success = true AND instrument_id = {inst_id}"),
                (meta.get('name'), meta.get('sector'), meta.get('market'), meta.get('country'))
            )
            rows = self.cursor.rowcount
            total_rows += rows
            print(f"  ✓ Instrument {inst_id}: {rows} rows")

        self.conn.commit()
        print(f"✓ Transformed {total_rows} total rows into ml_features")

    def calculate_targets(self, instrument_id: int = None):
        """
        Calculate target variables: future returns and dividend growth.

        For each (instrument, year), calculate:
        - next_year_dividend_growth = (dividend[year+1] - dividend[year]) / dividend[year]
        - next_3year_avg_dividend_growth = average of above for years +1, +2, +3
        """
        print(f"Calculating target variables{f' for instrument {instrument_id}' if instrument_id else ''}...")

        where_clause = "WHERE period = 5"  # Only full-year data
        if instrument_id:
            where_clause += f" AND instrument_id = {instrument_id}"

        # Clear existing targets
        if instrument_id:
            self.cursor.execute("DELETE FROM ml_targets WHERE instrument_id = %s", (instrument_id,))

        query = f"""
        WITH dividend_data AS (
            SELECT
                instrument_id,
                year,
                dividend_per_share as dividend,
                LEAD(dividend_per_share, 1) OVER (PARTITION BY instrument_id ORDER BY year) as next_year_dividend,
                LEAD(dividend_per_share, 2) OVER (PARTITION BY instrument_id ORDER BY year) as next_2year_dividend,
                LEAD(dividend_per_share, 3) OVER (PARTITION BY instrument_id ORDER BY year) as next_3year_dividend
            FROM ml_features
            {where_clause}
        )
        INSERT INTO ml_targets (
            instrument_id,
            year,
            next_year_dividend_growth,
            next_3year_avg_dividend_growth,
            dividend_increased,
            calculated_date
        )
        SELECT
            instrument_id,
            year,
            -- Next year dividend growth
            CASE
                WHEN dividend > 0 AND next_year_dividend IS NOT NULL
                THEN ((next_year_dividend - dividend) / dividend) * 100
                ELSE NULL
            END as next_year_dividend_growth,

            -- Average dividend growth over 3 years
            CASE
                WHEN dividend > 0 AND next_3year_dividend IS NOT NULL
                THEN (((next_3year_dividend - dividend) / dividend) / 3) * 100
                ELSE NULL
            END as next_3year_avg_dividend_growth,

            -- Binary: did dividend increase?
            CASE
                WHEN next_year_dividend > dividend THEN TRUE
                WHEN next_year_dividend <= dividend THEN FALSE
                ELSE NULL
            END as dividend_increased,

            NOW() as calculated_date
        FROM dividend_data
        WHERE dividend IS NOT NULL
        ON CONFLICT (instrument_id, year) DO UPDATE SET
            next_year_dividend_growth = EXCLUDED.next_year_dividend_growth,
            next_3year_avg_dividend_growth = EXCLUDED.next_3year_avg_dividend_growth,
            dividend_increased = EXCLUDED.dividend_increased,
            calculated_date = EXCLUDED.calculated_date
        """

        self.cursor.execute(query)
        rows = self.cursor.rowcount
        self.conn.commit()
        print(f"✓ Calculated targets for {rows} rows")

    def transform_stock_prices(self, instrument_id: int = None):
        """Extract daily stock prices from JSONB."""
        print(f"Transforming stock prices{f' for instrument {instrument_id}' if instrument_id else ''}...")

        where_clause = "WHERE endpoint_name LIKE 'stockprices_%' AND success = true"
        if instrument_id:
            where_clause += f" AND instrument_id = {instrument_id}"

        # Clear existing prices
        if instrument_id:
            self.cursor.execute("DELETE FROM ml_stock_prices WHERE instrument_id = %s", (instrument_id,))

        query = f"""
        WITH stock_data AS (
            SELECT
                instrument_id,
                (jsonb_array_elements(raw_data->'stockPricesList')->>'d')::date as date,
                (jsonb_array_elements(raw_data->'stockPricesList')->>'o')::numeric as open,
                (jsonb_array_elements(raw_data->'stockPricesList')->>'h')::numeric as high,
                (jsonb_array_elements(raw_data->'stockPricesList')->>'l')::numeric as low,
                (jsonb_array_elements(raw_data->'stockPricesList')->>'c')::numeric as close,
                (jsonb_array_elements(raw_data->'stockPricesList')->>'v')::bigint as volume,
                fetch_timestamp
            FROM api_raw_data
            {where_clause}
        ),
        with_returns AS (
            SELECT
                *,
                LAG(close) OVER (PARTITION BY instrument_id ORDER BY date) as prev_close
            FROM stock_data
        )
        INSERT INTO ml_stock_prices (
            instrument_id, date, open, high, low, close, volume, daily_return, fetch_date
        )
        SELECT
            instrument_id,
            date,
            open,
            high,
            low,
            close,
            volume,
            CASE WHEN prev_close > 0 THEN ((close - prev_close) / prev_close) ELSE NULL END as daily_return,
            fetch_timestamp
        FROM with_returns
        ON CONFLICT (instrument_id, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            daily_return = EXCLUDED.daily_return,
            fetch_date = EXCLUDED.fetch_date
        """

        self.cursor.execute(query)
        rows = self.cursor.rowcount
        self.conn.commit()
        print(f"✓ Transformed {rows} stock price records")

    def run(self, instrument_id: int = None):
        """Run full transformation pipeline."""
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"Starting ML Transformation")
        print(f"{'='*60}\n")

        try:
            self.connect()

            # Transform in order
            self.transform_kpi_data(instrument_id)
            self.calculate_targets(instrument_id)
            self.transform_stock_prices(instrument_id)

            duration = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*60}")
            print(f"✓ Transformation complete in {duration:.1f}s")
            print(f"{'='*60}\n")

            # Show sample data
            self.cursor.execute("""
                SELECT COUNT(*) as total_rows,
                       COUNT(DISTINCT instrument_id) as instruments,
                       MIN(year) as min_year,
                       MAX(year) as max_year
                FROM ml_features
            """)
            stats = self.cursor.fetchone()
            print(f"ML Features: {stats[0]} rows, {stats[1]} instruments, years {stats[2]}-{stats[3]}")

            self.cursor.execute("SELECT COUNT(*) FROM ml_targets WHERE next_year_dividend_growth IS NOT NULL")
            target_count = self.cursor.fetchone()[0]
            print(f"ML Targets: {target_count} rows with dividend growth targets")

            self.cursor.execute("SELECT COUNT(*) FROM ml_stock_prices")
            price_count = self.cursor.fetchone()[0]
            print(f"Stock Prices: {price_count} daily records")

        except Exception as e:
            print(f"✗ Error: {e}")
            self.conn.rollback()
            raise
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description='Transform raw API data to ML format')
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', type=int, default=int(os.getenv('DB_PORT', 5432)))
    parser.add_argument('--instrument-id', type=int, help='Transform specific instrument only')

    args = parser.parse_args()

    if not args.db_password:
        print("Error: Database password required (--db-password or DB_PASSWORD env var)")
        sys.exit(1)

    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }

    transformer = MLTransformer(db_config)
    transformer.run(args.instrument_id)


if __name__ == '__main__':
    import sys
    main()
