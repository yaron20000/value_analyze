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

        # KPI ID to column name mapping (matching Borsdata API IDs from fetch_and_store.py)
        self.kpi_mapping = {
            # Valuation
            1: 'dividend_yield', # Dividend Yield
            2: 'pe_ratio',      # P/E
            3: 'ps_ratio',      # P/S
            4: 'pb_ratio',      # P/B
            10: 'ev_ebit',      # EV/EBIT
            11: 'ev_ebitda',    # EV/EBITDA
            13: 'ev_fcf',       # EV/FCF
            19: 'peg_ratio',    # PEG Ratio
            15: 'ev_sales',     # EV/S

            # Profitability & Returns
            33: 'roe',              # Return on Equity
            34: 'roa',              # Return on Assets
            36: 'roc',              # Return on Capital
            37: 'roic',             # Return on Invested Capital
            32: 'ebitda_margin',    # EBITDA Margin
            29: 'operating_margin', # Operating Margin
            28: 'gross_margin',     # Gross Margin
            30: 'net_margin',       # Profit Margin
            24: 'fcf_margin_pct',   # FCF Margin % (KPI 24, distinct from KPI 31)
            51: 'ocf_margin',       # OCF Margin

            # Financial Health
            40: 'debt_equity',      # Debt to Equity
            39: 'equity_ratio',     # Equity Ratio
            44: 'current_ratio',    # Current Ratio
            41: 'net_debt_pct',     # Net Debt %
            42: 'net_debt_ebitda',  # Net Debt/EBITDA
            46: 'cash_pct',         # Cash %

            # Growth
            94: 'revenue_growth',      # Revenue Growth
            97: 'earnings_growth',     # Earnings Growth
            98: 'dividend_growth',     # Dividend Growth
            96: 'ebit_growth',         # EBIT Growth
            99: 'book_value_growth',   # Book Value Growth
            100: 'assets_growth',      # Assets Growth

            # Per Share
            5: 'revenue_per_share',        # Revenue per Share
            6: 'eps',                      # Earnings per Share
            7: 'dividend_per_share',       # Dividend per Share
            8: 'book_value_per_share',     # Book Value per Share
            23: 'fcf_per_share',           # FCF per Share
            68: 'ocf_per_share',           # OCF per Share

            # Cash Flow
            31: 'fcf_margin',    # FCF Margin
            27: 'earnings_fcf',  # Earnings/FCF
            62: 'ocf',           # Operating Cash Flow
            64: 'capex',         # Capex

            # Dividend
            20: 'dividend_payout',  # Dividend Payout %

            # Absolute Metrics
            56: 'earnings',          # Earnings
            53: 'revenue',           # Revenue
            54: 'ebitda',            # EBITDA
            57: 'total_assets',      # Total Assets
            58: 'total_equity',      # Total Equity
            60: 'net_debt',          # Net Debt
            61: 'num_shares',        # Number of Shares
            49: 'enterprise_value',  # Enterprise Value
            50: 'market_cap',        # Market Cap
            63: 'fcf',              # Free Cash Flow (absolute)
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

        # Build WHERE clause - now using batch KPI records
        # Note: %% escapes the % for psycopg2 parameter substitution
        where_clause = "WHERE endpoint_name LIKE 'kpi_%%_batch' AND success = true"

        # First, get metadata
        metadata = self.get_instrument_metadata()

        # Build dynamic PIVOT query
        kpi_cases = []
        for kpi_id, col_name in self.kpi_mapping.items():
            kpi_cases.append(f"""
                MAX(CASE WHEN kpi_id = {kpi_id} THEN value END) as {col_name}
            """)

        pivot_sql = ", ".join(kpi_cases)

        query = f"""
        WITH unnested_kpis AS (
            -- Handle batch KPI format: kpisList contains array of instruments
            SELECT
                (inst_data->>'instrument')::integer as instrument_id,
                (raw_data->>'kpiId')::integer as kpi_id,
                (jsonb_array_elements(inst_data->'values')->>'y')::integer as year,
                (jsonb_array_elements(inst_data->'values')->>'p')::integer as period,
                (jsonb_array_elements(inst_data->'values')->>'v')::numeric as value,
                fetch_timestamp
            FROM api_raw_data,
                jsonb_array_elements(raw_data->'kpisList') as inst_data
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
            dividend_yield, pe_ratio, ps_ratio, pb_ratio, ev_ebit, ev_ebitda, ev_fcf, peg_ratio, ev_sales,
            roe, roa, roc, roic, ebitda_margin, operating_margin, gross_margin, net_margin, fcf_margin_pct, ocf_margin,
            debt_equity, equity_ratio, current_ratio, net_debt_pct, net_debt_ebitda, cash_pct,
            revenue_growth, earnings_growth, dividend_growth, ebit_growth, book_value_growth, assets_growth,
            revenue_per_share, eps, dividend_per_share, book_value_per_share, fcf_per_share, ocf_per_share,
            fcf_margin, earnings_fcf, ocf, capex,
            dividend_payout,
            earnings, revenue, ebitda, total_assets, total_equity, net_debt, num_shares,
            enterprise_value, market_cap, fcf,
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
            p.dividend_yield, p.pe_ratio, p.ps_ratio, p.pb_ratio, p.ev_ebit, p.ev_ebitda, p.ev_fcf, p.peg_ratio, p.ev_sales,
            p.roe, p.roa, p.roc, p.roic, p.ebitda_margin, p.operating_margin, p.gross_margin, p.net_margin, p.fcf_margin_pct, p.ocf_margin,
            p.debt_equity, p.equity_ratio, p.current_ratio, p.net_debt_pct, p.net_debt_ebitda, p.cash_pct,
            p.revenue_growth, p.earnings_growth, p.dividend_growth, p.ebit_growth, p.book_value_growth, p.assets_growth,
            p.revenue_per_share, p.eps, p.dividend_per_share, p.book_value_per_share, p.fcf_per_share, p.ocf_per_share,
            p.fcf_margin, p.earnings_fcf, p.ocf, p.capex,
            p.dividend_payout,
            p.earnings, p.revenue, p.ebitda, p.total_assets, p.total_equity, p.net_debt, p.num_shares,
            p.enterprise_value, p.market_cap, p.fcf,
            p.fetch_date
        FROM pivoted p
        ON CONFLICT (instrument_id, year, period) DO UPDATE SET
            dividend_yield = EXCLUDED.dividend_yield,
            pe_ratio = EXCLUDED.pe_ratio,
            ps_ratio = EXCLUDED.ps_ratio,
            pb_ratio = EXCLUDED.pb_ratio,
            ev_ebit = EXCLUDED.ev_ebit,
            ev_ebitda = EXCLUDED.ev_ebitda,
            ev_fcf = EXCLUDED.ev_fcf,
            peg_ratio = EXCLUDED.peg_ratio,
            ev_sales = EXCLUDED.ev_sales,
            roe = EXCLUDED.roe,
            roa = EXCLUDED.roa,
            roc = EXCLUDED.roc,
            roic = EXCLUDED.roic,
            ebitda_margin = EXCLUDED.ebitda_margin,
            operating_margin = EXCLUDED.operating_margin,
            gross_margin = EXCLUDED.gross_margin,
            net_margin = EXCLUDED.net_margin,
            fcf_margin_pct = EXCLUDED.fcf_margin_pct,
            ocf_margin = EXCLUDED.ocf_margin,
            debt_equity = EXCLUDED.debt_equity,
            equity_ratio = EXCLUDED.equity_ratio,
            current_ratio = EXCLUDED.current_ratio,
            net_debt_pct = EXCLUDED.net_debt_pct,
            net_debt_ebitda = EXCLUDED.net_debt_ebitda,
            cash_pct = EXCLUDED.cash_pct,
            revenue_growth = EXCLUDED.revenue_growth,
            earnings_growth = EXCLUDED.earnings_growth,
            dividend_growth = EXCLUDED.dividend_growth,
            ebit_growth = EXCLUDED.ebit_growth,
            book_value_growth = EXCLUDED.book_value_growth,
            assets_growth = EXCLUDED.assets_growth,
            revenue_per_share = EXCLUDED.revenue_per_share,
            eps = EXCLUDED.eps,
            dividend_per_share = EXCLUDED.dividend_per_share,
            book_value_per_share = EXCLUDED.book_value_per_share,
            fcf_per_share = EXCLUDED.fcf_per_share,
            ocf_per_share = EXCLUDED.ocf_per_share,
            fcf_margin = EXCLUDED.fcf_margin,
            earnings_fcf = EXCLUDED.earnings_fcf,
            ocf = EXCLUDED.ocf,
            capex = EXCLUDED.capex,
            dividend_payout = EXCLUDED.dividend_payout,
            earnings = EXCLUDED.earnings,
            revenue = EXCLUDED.revenue,
            ebitda = EXCLUDED.ebitda,
            total_assets = EXCLUDED.total_assets,
            total_equity = EXCLUDED.total_equity,
            net_debt = EXCLUDED.net_debt,
            num_shares = EXCLUDED.num_shares,
            enterprise_value = EXCLUDED.enterprise_value,
            market_cap = EXCLUDED.market_cap,
            fcf = EXCLUDED.fcf,
            fetch_date = EXCLUDED.fetch_date
        """

        # Get instruments from batch data
        if instrument_id:
            instruments = [instrument_id]
        else:
            # Get all instruments from batch KPI data
            self.cursor.execute("""
                SELECT DISTINCT (inst_data->>'instrument')::integer as instrument_id
                FROM api_raw_data,
                    jsonb_array_elements(raw_data->'kpisList') as inst_data
                WHERE endpoint_name LIKE 'kpi_%%_batch' AND success = true
                ORDER BY instrument_id
            """)
            instruments = [row[0] for row in self.cursor.fetchall()]

        print(f"  Found {len(instruments)} instruments in batch data")

        # Clear existing data if needed
        if instrument_id:
            self.cursor.execute("DELETE FROM ml_features WHERE instrument_id = %s", (instrument_id,))

        total_rows = 0
        for inst_id in instruments:
            meta = metadata.get(inst_id, {})
            # Filter the batch data for this specific instrument
            # Note: %% escapes the % for psycopg2 parameter substitution
            inst_where = f"WHERE endpoint_name LIKE 'kpi_%%_batch' AND success = true AND (inst_data->>'instrument')::integer = {inst_id}"
            self.cursor.execute(query.replace(where_clause, inst_where),
                (meta.get('name'), meta.get('sector'), meta.get('market'), meta.get('country'))
            )
            rows = self.cursor.rowcount
            total_rows += rows
            if rows > 0:
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
                LEAD(dividend_per_share, 3) OVER (PARTITION BY instrument_id ORDER BY year) as next_3year_dividend,
                LEAD(dividend_per_share, 5) OVER (PARTITION BY instrument_id ORDER BY year) as next_5year_dividend
            FROM ml_features
            {where_clause}
        )
        INSERT INTO ml_targets (
            instrument_id,
            year,
            next_year_dividend_growth,
            next_3year_avg_dividend_growth,
            next_5year_avg_dividend_growth,
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

            -- Average dividend growth over 5 years
            CASE
                WHEN dividend > 0 AND next_5year_dividend IS NOT NULL
                THEN (((next_5year_dividend - dividend) / dividend) / 5) * 100
                ELSE NULL
            END as next_5year_avg_dividend_growth,

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
            next_5year_avg_dividend_growth = EXCLUDED.next_5year_avg_dividend_growth,
            dividend_increased = EXCLUDED.dividend_increased,
            calculated_date = EXCLUDED.calculated_date
        """

        self.cursor.execute(query)
        rows = self.cursor.rowcount
        self.conn.commit()
        print(f"✓ Calculated dividend targets for {rows} rows")

    def calculate_return_targets(self, instrument_id: int = None):
        """
        Calculate stock return targets using ml_stock_prices data.

        For each (instrument, year), calculate:
        - next_year_return: % return from end-of-year Y to end-of-year Y+1
        - next_3year_return: % return from end-of-year Y to end-of-year Y+3
        - next_5year_return: % return from end-of-year Y to end-of-year Y+5
        - next_year_outperformed: did stock beat median return that year?
        """
        print(f"Calculating stock return targets{f' for instrument {instrument_id}' if instrument_id else ''}...")

        where_clause = ""
        if instrument_id:
            where_clause = f"AND yr.instrument_id = {instrument_id}"

        query = f"""
        WITH yearly_prices AS (
            -- Get the last trading day's close price for each (instrument, calendar year)
            SELECT DISTINCT ON (instrument_id, EXTRACT(YEAR FROM date))
                instrument_id,
                EXTRACT(YEAR FROM date)::integer as year,
                close as year_end_price
            FROM ml_stock_prices
            WHERE close IS NOT NULL AND close > 0
            ORDER BY instrument_id, EXTRACT(YEAR FROM date), date DESC
        ),
        returns AS (
            SELECT
                yr.instrument_id,
                yr.year,
                CASE WHEN y1.year_end_price IS NOT NULL
                    THEN ((y1.year_end_price - yr.year_end_price) / yr.year_end_price) * 100
                END as next_year_return,
                CASE WHEN y3.year_end_price IS NOT NULL
                    THEN ((y3.year_end_price - yr.year_end_price) / yr.year_end_price) * 100
                END as next_3year_return,
                CASE WHEN y5.year_end_price IS NOT NULL
                    THEN ((y5.year_end_price - yr.year_end_price) / yr.year_end_price) * 100
                END as next_5year_return
            FROM yearly_prices yr
            LEFT JOIN yearly_prices y1 ON yr.instrument_id = y1.instrument_id AND y1.year = yr.year + 1
            LEFT JOIN yearly_prices y3 ON yr.instrument_id = y3.instrument_id AND y3.year = yr.year + 3
            LEFT JOIN yearly_prices y5 ON yr.instrument_id = y5.instrument_id AND y5.year = yr.year + 5
            WHERE 1=1 {where_clause}
        ),
        market_median AS (
            SELECT
                year,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY next_year_return) as median_return
            FROM returns
            WHERE next_year_return IS NOT NULL
            GROUP BY year
        )
        UPDATE ml_targets t
        SET
            next_year_return = r.next_year_return,
            next_3year_return = r.next_3year_return,
            next_5year_return = r.next_5year_return,
            next_year_excess_return = CASE WHEN r.next_year_return IS NOT NULL AND m.median_return IS NOT NULL
                THEN r.next_year_return - m.median_return ELSE NULL END,
            market_median_return = m.median_return,
            next_year_outperformed = (r.next_year_return > m.median_return),
            calculated_date = NOW()
        FROM returns r
        LEFT JOIN market_median m ON r.year = m.year
        WHERE t.instrument_id = r.instrument_id AND t.year = r.year
        """

        self.cursor.execute(query)
        updated = self.cursor.rowcount

        # Also insert return targets for instruments that have prices but no dividend data
        # (they won't have rows in ml_targets yet)
        insert_query = f"""
        WITH yearly_prices AS (
            SELECT DISTINCT ON (instrument_id, EXTRACT(YEAR FROM date))
                instrument_id,
                EXTRACT(YEAR FROM date)::integer as year,
                close as year_end_price
            FROM ml_stock_prices
            WHERE close IS NOT NULL AND close > 0
            ORDER BY instrument_id, EXTRACT(YEAR FROM date), date DESC
        ),
        returns AS (
            SELECT
                yr.instrument_id,
                yr.year,
                CASE WHEN y1.year_end_price IS NOT NULL
                    THEN ((y1.year_end_price - yr.year_end_price) / yr.year_end_price) * 100
                END as next_year_return,
                CASE WHEN y3.year_end_price IS NOT NULL
                    THEN ((y3.year_end_price - yr.year_end_price) / yr.year_end_price) * 100
                END as next_3year_return,
                CASE WHEN y5.year_end_price IS NOT NULL
                    THEN ((y5.year_end_price - yr.year_end_price) / yr.year_end_price) * 100
                END as next_5year_return
            FROM yearly_prices yr
            LEFT JOIN yearly_prices y1 ON yr.instrument_id = y1.instrument_id AND y1.year = yr.year + 1
            LEFT JOIN yearly_prices y3 ON yr.instrument_id = y3.instrument_id AND y3.year = yr.year + 3
            LEFT JOIN yearly_prices y5 ON yr.instrument_id = y5.instrument_id AND y5.year = yr.year + 5
            WHERE 1=1 {where_clause}
        ),
        market_median AS (
            SELECT
                year,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY next_year_return) as median_return
            FROM returns
            WHERE next_year_return IS NOT NULL
            GROUP BY year
        )
        INSERT INTO ml_targets (
            instrument_id, year,
            next_year_return, next_3year_return, next_5year_return,
            next_year_excess_return, market_median_return,
            next_year_outperformed, calculated_date
        )
        SELECT
            r.instrument_id, r.year,
            r.next_year_return, r.next_3year_return, r.next_5year_return,
            CASE WHEN r.next_year_return IS NOT NULL AND m.median_return IS NOT NULL
                THEN r.next_year_return - m.median_return ELSE NULL END,
            m.median_return,
            (r.next_year_return > m.median_return),
            NOW()
        FROM returns r
        LEFT JOIN market_median m ON r.year = m.year
        WHERE r.next_year_return IS NOT NULL
        ON CONFLICT (instrument_id, year) DO NOTHING
        """

        self.cursor.execute(insert_query)
        inserted = self.cursor.rowcount
        self.conn.commit()
        print(f"✓ Stock return targets: {updated} updated, {inserted} new rows")

    def transform_stock_prices(self, instrument_id: int = None):
        """Extract daily stock prices from JSONB."""
        print(f"Transforming stock prices{f' for instrument {instrument_id}' if instrument_id else ''}...")

        # Exclude stockprices_array, stockprices_last, stockprices_by_date - only use per-instrument records
        where_clause = "WHERE endpoint_name LIKE 'stockprices_%%' AND endpoint_name NOT IN ('stockprices_array', 'stockprices_last', 'stockprices_by_date') AND success = true AND instrument_id IS NOT NULL"
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
        -- Deduplicate: keep most recent fetch for each (instrument_id, date)
        deduplicated AS (
            SELECT DISTINCT ON (instrument_id, date)
                instrument_id, date, open, high, low, close, volume, fetch_timestamp
            FROM stock_data
            ORDER BY instrument_id, date, fetch_timestamp DESC
        ),
        with_returns AS (
            SELECT
                *,
                LAG(close) OVER (PARTITION BY instrument_id ORDER BY date) as prev_close
            FROM deduplicated
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
        WHERE instrument_id IS NOT NULL
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

    def calculate_pre_report_features(self, instrument_id: int = None):
        """
        Calculate pre-report price features - how stock behaved before earnings.

        For each report, calculates:
        - Price changes N days before report (5, 10, 20, 30 days)
        - Volume changes and spikes
        - Volatility measures
        - Trend indicators (was price rising?)
        """
        print(f"Calculating pre-report features{f' for instrument {instrument_id}' if instrument_id else ''}...")

        # Clear existing data
        if instrument_id:
            self.cursor.execute("DELETE FROM ml_pre_report_features WHERE instrument_id = %s", (instrument_id,))
        else:
            self.cursor.execute("TRUNCATE ml_pre_report_features")

        # Build WHERE clause for reports
        where_clause = "WHERE endpoint_name LIKE 'reports_year_%' AND success = true"
        if instrument_id:
            where_clause += f" AND instrument_id = {instrument_id}"

        query = f"""
        WITH report_dates AS (
            -- Extract report dates from the reports JSONB
            SELECT DISTINCT
                instrument_id,
                (elem->>'year')::integer as report_year,
                (elem->>'period')::integer as report_period,
                (elem->>'report_Date')::date as report_date
            FROM api_raw_data,
                jsonb_array_elements(raw_data->'reports') as elem
            {where_clause}
            AND (elem->>'report_Date') IS NOT NULL
        ),
        price_windows AS (
            -- For each report, get price data in the window before the report
            SELECT
                rd.instrument_id,
                rd.report_year,
                rd.report_period,
                rd.report_date,
                sp.date as price_date,
                sp.close,
                sp.volume,
                sp.daily_return,
                -- Calculate days before report (negative = before)
                rd.report_date - sp.date as days_before_report
            FROM report_dates rd
            JOIN ml_stock_prices sp ON rd.instrument_id = sp.instrument_id
            WHERE sp.date >= rd.report_date - INTERVAL '40 days'
              AND sp.date < rd.report_date
        ),
        aggregated AS (
            SELECT
                instrument_id,
                report_year,
                report_period,
                report_date,

                -- Price at different points before report
                MAX(CASE WHEN days_before_report <= 1 THEN close END) as price_day_before,
                MAX(CASE WHEN days_before_report BETWEEN 4 AND 6 THEN close END) as price_5d_before,
                MAX(CASE WHEN days_before_report BETWEEN 9 AND 11 THEN close END) as price_10d_before,
                MAX(CASE WHEN days_before_report BETWEEN 19 AND 21 THEN close END) as price_20d_before,
                MAX(CASE WHEN days_before_report BETWEEN 29 AND 31 THEN close END) as price_30d_before,

                -- Volume averages
                AVG(CASE WHEN days_before_report <= 5 THEN volume END) as avg_volume_5d,
                AVG(CASE WHEN days_before_report <= 20 THEN volume END) as avg_volume_20d,

                -- Volatility (std dev of daily returns)
                STDDEV(CASE WHEN days_before_report <= 5 THEN daily_return END) as volatility_5d,
                STDDEV(CASE WHEN days_before_report <= 20 THEN daily_return END) as volatility_20d,

                -- High/Low in 20 days before
                MAX(CASE WHEN days_before_report <= 20 THEN close END) as high_20d,
                MIN(CASE WHEN days_before_report <= 20 THEN close END) as low_20d

            FROM price_windows
            GROUP BY instrument_id, report_year, report_period, report_date
        )
        INSERT INTO ml_pre_report_features (
            instrument_id, report_year, report_period, report_date,
            price_change_5d, price_change_10d, price_change_20d, price_change_30d,
            avg_volume_5d, avg_volume_20d, volume_ratio_5d_20d,
            volatility_5d, volatility_20d,
            was_rising_5d, was_rising_10d, was_rising_20d,
            pct_from_20d_high, pct_from_20d_low,
            price_at_report, price_5d_before, price_20d_before,
            calculated_date
        )
        SELECT
            instrument_id,
            report_year,
            report_period,
            report_date,

            -- Price changes (%)
            CASE WHEN price_5d_before > 0 THEN
                ((price_day_before - price_5d_before) / price_5d_before) * 100
            END as price_change_5d,
            CASE WHEN price_10d_before > 0 THEN
                ((price_day_before - price_10d_before) / price_10d_before) * 100
            END as price_change_10d,
            CASE WHEN price_20d_before > 0 THEN
                ((price_day_before - price_20d_before) / price_20d_before) * 100
            END as price_change_20d,
            CASE WHEN price_30d_before > 0 THEN
                ((price_day_before - price_30d_before) / price_30d_before) * 100
            END as price_change_30d,

            -- Volume features
            avg_volume_5d,
            avg_volume_20d,
            CASE WHEN avg_volume_20d > 0 THEN avg_volume_5d / avg_volume_20d END as volume_ratio_5d_20d,

            -- Volatility
            volatility_5d,
            volatility_20d,

            -- Trend indicators
            price_day_before > price_5d_before as was_rising_5d,
            price_day_before > price_10d_before as was_rising_10d,
            price_day_before > price_20d_before as was_rising_20d,

            -- Position relative to high/low
            CASE WHEN high_20d > 0 THEN
                ((high_20d - price_day_before) / high_20d) * 100
            END as pct_from_20d_high,
            CASE WHEN low_20d > 0 THEN
                ((price_day_before - low_20d) / low_20d) * 100
            END as pct_from_20d_low,

            -- Prices
            price_day_before as price_at_report,
            price_5d_before,
            price_20d_before,

            NOW() as calculated_date

        FROM aggregated
        WHERE price_day_before IS NOT NULL
        ON CONFLICT (instrument_id, report_year, report_period) DO UPDATE SET
            price_change_5d = EXCLUDED.price_change_5d,
            price_change_10d = EXCLUDED.price_change_10d,
            price_change_20d = EXCLUDED.price_change_20d,
            price_change_30d = EXCLUDED.price_change_30d,
            avg_volume_5d = EXCLUDED.avg_volume_5d,
            avg_volume_20d = EXCLUDED.avg_volume_20d,
            volume_ratio_5d_20d = EXCLUDED.volume_ratio_5d_20d,
            volatility_5d = EXCLUDED.volatility_5d,
            volatility_20d = EXCLUDED.volatility_20d,
            was_rising_5d = EXCLUDED.was_rising_5d,
            was_rising_10d = EXCLUDED.was_rising_10d,
            was_rising_20d = EXCLUDED.was_rising_20d,
            pct_from_20d_high = EXCLUDED.pct_from_20d_high,
            pct_from_20d_low = EXCLUDED.pct_from_20d_low,
            price_at_report = EXCLUDED.price_at_report,
            price_5d_before = EXCLUDED.price_5d_before,
            price_20d_before = EXCLUDED.price_20d_before,
            calculated_date = EXCLUDED.calculated_date
        """

        self.cursor.execute(query)
        rows = self.cursor.rowcount
        self.conn.commit()
        print(f"✓ Calculated pre-report features for {rows} reports")

    def transform_holdings_data(self, instrument_id: int = None):
        """
        Transform insider trading and buyback data from raw JSONB into ml_holdings_features.

        Insider data: per-instrument list of transactions with shares (+buy/-sell), amount, transactionDate
        Buyback data: per-instrument list of events with change (shares), price, date

        Note: Shorts data is snapshot-only (no time series) and is excluded to prevent data leakage.
        """
        print(f"Transforming holdings data{f' for instrument {instrument_id}' if instrument_id else ''}...")

        # Clear existing data
        if instrument_id:
            self.cursor.execute("DELETE FROM ml_holdings_features WHERE instrument_id = %s", (instrument_id,))
        else:
            self.cursor.execute("TRUNCATE ml_holdings_features")

        # Transform insider transactions: aggregate by (instrument_id, year)
        insider_query = """
        WITH insider_raw AS (
            SELECT
                (inst->>'insId')::integer as instrument_id,
                (tx->>'transactionDate')::date as tx_date,
                EXTRACT(YEAR FROM (tx->>'transactionDate')::date)::integer as tx_year,
                (tx->>'shares')::integer as shares,
                (tx->>'amount')::numeric as amount
            FROM api_raw_data,
                jsonb_array_elements(raw_data->'list') as inst,
                jsonb_array_elements(inst->'values') as tx
            WHERE endpoint_name = 'holdings_insider'
              AND success = true
              AND (tx->>'transactionDate') IS NOT NULL
        ),
        insider_yearly AS (
            SELECT
                instrument_id,
                tx_year as year,
                SUM(shares) as insider_net_shares,
                SUM(amount) as insider_net_amount,
                COUNT(CASE WHEN shares > 0 THEN 1 END) as insider_buy_count,
                COUNT(CASE WHEN shares < 0 THEN 1 END) as insider_sell_count,
                COUNT(*) as insider_transaction_count,
                CASE WHEN COUNT(*) > 0
                    THEN COUNT(CASE WHEN shares > 0 THEN 1 END)::numeric / COUNT(*)
                    ELSE NULL
                END as insider_buy_ratio
            FROM insider_raw
            GROUP BY instrument_id, tx_year
        )
        INSERT INTO ml_holdings_features (
            instrument_id, year,
            insider_net_shares, insider_net_amount,
            insider_buy_count, insider_sell_count, insider_transaction_count, insider_buy_ratio
        )
        SELECT
            instrument_id, year,
            insider_net_shares, insider_net_amount,
            insider_buy_count, insider_sell_count, insider_transaction_count, insider_buy_ratio
        FROM insider_yearly
        ON CONFLICT (instrument_id, year) DO UPDATE SET
            insider_net_shares = EXCLUDED.insider_net_shares,
            insider_net_amount = EXCLUDED.insider_net_amount,
            insider_buy_count = EXCLUDED.insider_buy_count,
            insider_sell_count = EXCLUDED.insider_sell_count,
            insider_transaction_count = EXCLUDED.insider_transaction_count,
            insider_buy_ratio = EXCLUDED.insider_buy_ratio
        """

        self.cursor.execute(insider_query)
        insider_rows = self.cursor.rowcount
        print(f"  ✓ Insider features: {insider_rows} (instrument, year) rows")

        # Transform buyback data: aggregate by (instrument_id, year)
        buyback_query = """
        WITH buyback_raw AS (
            SELECT
                (inst->>'insId')::integer as instrument_id,
                (ev->>'date')::date as ev_date,
                EXTRACT(YEAR FROM (ev->>'date')::date)::integer as ev_year,
                (ev->>'change')::bigint as change_shares,
                (ev->>'price')::numeric as price,
                (ev->>'sharesProc')::numeric as shares_pct
            FROM api_raw_data,
                jsonb_array_elements(raw_data->'list') as inst,
                jsonb_array_elements(inst->'values') as ev
            WHERE endpoint_name = 'holdings_buyback'
              AND success = true
              AND (ev->>'date') IS NOT NULL
        ),
        buyback_yearly AS (
            SELECT
                instrument_id,
                ev_year as year,
                SUM(ABS(change_shares)) as buyback_total_shares,
                SUM(ABS(change_shares) * price) as buyback_total_amount,
                COUNT(*) as buyback_count,
                MAX(shares_pct) as buyback_shares_pct
            FROM buyback_raw
            WHERE change_shares > 0  -- only actual buybacks, not program cancellations
            GROUP BY instrument_id, ev_year
        )
        UPDATE ml_holdings_features h
        SET
            buyback_total_shares = b.buyback_total_shares,
            buyback_total_amount = b.buyback_total_amount,
            buyback_count = b.buyback_count,
            buyback_shares_pct = b.buyback_shares_pct
        FROM buyback_yearly b
        WHERE h.instrument_id = b.instrument_id AND h.year = b.year
        """

        self.cursor.execute(buyback_query)
        buyback_updated = self.cursor.rowcount

        # Also insert buyback data for instruments/years not yet in the table
        buyback_insert = """
        WITH buyback_raw AS (
            SELECT
                (inst->>'insId')::integer as instrument_id,
                (ev->>'date')::date as ev_date,
                EXTRACT(YEAR FROM (ev->>'date')::date)::integer as ev_year,
                (ev->>'change')::bigint as change_shares,
                (ev->>'price')::numeric as price,
                (ev->>'sharesProc')::numeric as shares_pct
            FROM api_raw_data,
                jsonb_array_elements(raw_data->'list') as inst,
                jsonb_array_elements(inst->'values') as ev
            WHERE endpoint_name = 'holdings_buyback'
              AND success = true
              AND (ev->>'date') IS NOT NULL
        ),
        buyback_yearly AS (
            SELECT
                instrument_id,
                ev_year as year,
                SUM(ABS(change_shares)) as buyback_total_shares,
                SUM(ABS(change_shares) * price) as buyback_total_amount,
                COUNT(*) as buyback_count,
                MAX(shares_pct) as buyback_shares_pct
            FROM buyback_raw
            WHERE change_shares > 0
            GROUP BY instrument_id, ev_year
        )
        INSERT INTO ml_holdings_features (
            instrument_id, year,
            buyback_total_shares, buyback_total_amount, buyback_count, buyback_shares_pct
        )
        SELECT
            instrument_id, year,
            buyback_total_shares, buyback_total_amount, buyback_count, buyback_shares_pct
        FROM buyback_yearly b
        WHERE NOT EXISTS (
            SELECT 1 FROM ml_holdings_features h
            WHERE h.instrument_id = b.instrument_id AND h.year = b.year
        )
        """

        self.cursor.execute(buyback_insert)
        buyback_inserted = self.cursor.rowcount

        self.conn.commit()
        print(f"  ✓ Buyback features: {buyback_updated} updated, {buyback_inserted} new rows")

        # Summary
        self.cursor.execute("SELECT COUNT(*) FROM ml_holdings_features")
        total = self.cursor.fetchone()[0]
        print(f"✓ Holdings features total: {total} (instrument, year) rows")

    def calculate_monthly_targets(self, instrument_id: int = None):
        """
        Calculate monthly stock return targets from ml_stock_prices.

        For each (instrument, year, month), compute:
        - month_end_price: closing price on the last trading day of the month
        - next_month_return: % return from this month-end to next month-end
        - market_median_monthly_return: median of all stocks' next_month_return
        - next_month_excess_return: stock return minus median
        """
        print(f"Calculating monthly targets{f' for instrument {instrument_id}' if instrument_id else ''}...")

        where_clause = ""
        if instrument_id:
            where_clause = f"AND me.instrument_id = {instrument_id}"

        query = f"""
        WITH month_end_prices AS (
            -- Get the last trading day's close for each (instrument, year, month)
            SELECT DISTINCT ON (instrument_id, EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date))
                instrument_id,
                EXTRACT(YEAR FROM date)::integer as year,
                EXTRACT(MONTH FROM date)::integer as month,
                date as month_end_date,
                close as month_end_price
            FROM ml_stock_prices
            WHERE close IS NOT NULL AND close > 0
            ORDER BY instrument_id, EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date), date DESC
        ),
        with_next AS (
            SELECT
                me.*,
                LEAD(month_end_price) OVER (
                    PARTITION BY instrument_id ORDER BY year, month
                ) as next_month_price,
                LEAD(year) OVER (
                    PARTITION BY instrument_id ORDER BY year, month
                ) as next_year,
                LEAD(month) OVER (
                    PARTITION BY instrument_id ORDER BY year, month
                ) as next_month
            FROM month_end_prices me
        ),
        returns AS (
            SELECT
                instrument_id, year, month, month_end_date, month_end_price,
                CASE
                    -- Only calculate return if next month is actually the consecutive month
                    WHEN next_month_price IS NOT NULL
                        AND (year * 12 + month + 1) = (next_year * 12 + next_month)
                    THEN ((next_month_price - month_end_price) / month_end_price) * 100
                END as next_month_return
            FROM with_next
            WHERE 1=1 {where_clause}
        ),
        market_median AS (
            SELECT
                year, month,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY next_month_return) as median_return
            FROM returns
            WHERE next_month_return IS NOT NULL
            GROUP BY year, month
        )
        INSERT INTO ml_monthly_targets (
            instrument_id, year, month,
            month_end_date, month_end_price,
            next_month_return, market_median_monthly_return, next_month_excess_return
        )
        SELECT
            r.instrument_id, r.year, r.month,
            r.month_end_date, r.month_end_price,
            r.next_month_return,
            m.median_return,
            CASE WHEN r.next_month_return IS NOT NULL AND m.median_return IS NOT NULL
                THEN r.next_month_return - m.median_return ELSE NULL END
        FROM returns r
        LEFT JOIN market_median m ON r.year = m.year AND r.month = m.month
        ON CONFLICT (instrument_id, year, month) DO UPDATE SET
            month_end_date = EXCLUDED.month_end_date,
            month_end_price = EXCLUDED.month_end_price,
            next_month_return = EXCLUDED.next_month_return,
            market_median_monthly_return = EXCLUDED.market_median_monthly_return,
            next_month_excess_return = EXCLUDED.next_month_excess_return
        """

        self.cursor.execute(query)
        rows = self.cursor.rowcount
        self.conn.commit()
        print(f"✓ Monthly targets: {rows} (instrument, year, month) rows")

    def calculate_monthly_price_features(self, instrument_id: int = None):
        """
        Calculate price-based features as of each month-end.

        Same features as pre-report features but anchored to month-end dates:
        - Momentum: price change over 5/10/20/30 days
        - Volume: ratio of recent to longer-term average
        - Volatility: std dev of daily returns
        - Trend: was price rising?
        - Relative position: distance from 20-day high/low
        """
        print(f"Calculating monthly price features{f' for instrument {instrument_id}' if instrument_id else ''}...")

        where_clause = ""
        if instrument_id:
            where_clause = f"AND mt.instrument_id = {instrument_id}"

        query = f"""
        WITH month_ends AS (
            SELECT DISTINCT instrument_id, year, month, month_end_date
            FROM ml_monthly_targets
            WHERE month_end_date IS NOT NULL
            {'AND instrument_id = ' + str(instrument_id) if instrument_id else ''}
        ),
        price_windows AS (
            SELECT
                me.instrument_id,
                me.year,
                me.month,
                me.month_end_date,
                sp.date as price_date,
                sp.close,
                sp.volume,
                sp.daily_return,
                me.month_end_date - sp.date as days_before
            FROM month_ends me
            JOIN ml_stock_prices sp ON me.instrument_id = sp.instrument_id
            WHERE sp.date >= me.month_end_date - INTERVAL '40 days'
              AND sp.date <= me.month_end_date
              AND sp.close IS NOT NULL AND sp.close > 0
        ),
        aggregated AS (
            SELECT
                instrument_id, year, month, month_end_date,

                -- Price at month-end (days_before = 0)
                MAX(CASE WHEN days_before = 0 THEN close END) as price_at_end,

                -- Prices at N days before (closest available day in range)
                MAX(CASE WHEN days_before BETWEEN 4 AND 7 THEN close END) as price_5d_before,
                MAX(CASE WHEN days_before BETWEEN 9 AND 14 THEN close END) as price_10d_before,
                MAX(CASE WHEN days_before BETWEEN 18 AND 25 THEN close END) as price_20d_before,
                MAX(CASE WHEN days_before BETWEEN 28 AND 35 THEN close END) as price_30d_before,

                -- Volume averages
                AVG(CASE WHEN days_before <= 7 THEN volume END) as avg_volume_5d,
                AVG(CASE WHEN days_before <= 25 THEN volume END) as avg_volume_20d,

                -- Volatility (std dev of daily returns)
                STDDEV(CASE WHEN days_before <= 7 THEN daily_return END) as volatility_5d,
                STDDEV(CASE WHEN days_before <= 25 THEN daily_return END) as volatility_20d,

                -- 20-day high/low
                MAX(CASE WHEN days_before <= 25 THEN close END) as high_20d,
                MIN(CASE WHEN days_before <= 25 THEN close END) as low_20d

            FROM price_windows
            GROUP BY instrument_id, year, month, month_end_date
        )
        INSERT INTO ml_monthly_price_features (
            instrument_id, year, month, month_end_date,
            price_change_5d, price_change_10d, price_change_20d, price_change_30d,
            volume_ratio_5d_20d,
            volatility_5d, volatility_20d,
            was_rising_5d, was_rising_10d, was_rising_20d,
            pct_from_20d_high, pct_from_20d_low
        )
        SELECT
            instrument_id, year, month, month_end_date,

            CASE WHEN price_5d_before > 0
                THEN ((price_at_end - price_5d_before) / price_5d_before) * 100 END,
            CASE WHEN price_10d_before > 0
                THEN ((price_at_end - price_10d_before) / price_10d_before) * 100 END,
            CASE WHEN price_20d_before > 0
                THEN ((price_at_end - price_20d_before) / price_20d_before) * 100 END,
            CASE WHEN price_30d_before > 0
                THEN ((price_at_end - price_30d_before) / price_30d_before) * 100 END,

            CASE WHEN avg_volume_20d > 0
                THEN avg_volume_5d / avg_volume_20d END,

            volatility_5d,
            volatility_20d,

            price_at_end > price_5d_before,
            price_at_end > price_10d_before,
            price_at_end > price_20d_before,

            CASE WHEN high_20d > 0
                THEN ((price_at_end - high_20d) / high_20d) * 100 END,
            CASE WHEN low_20d > 0
                THEN ((price_at_end - low_20d) / low_20d) * 100 END

        FROM aggregated
        WHERE price_at_end IS NOT NULL
        ON CONFLICT (instrument_id, year, month) DO UPDATE SET
            month_end_date = EXCLUDED.month_end_date,
            price_change_5d = EXCLUDED.price_change_5d,
            price_change_10d = EXCLUDED.price_change_10d,
            price_change_20d = EXCLUDED.price_change_20d,
            price_change_30d = EXCLUDED.price_change_30d,
            volume_ratio_5d_20d = EXCLUDED.volume_ratio_5d_20d,
            volatility_5d = EXCLUDED.volatility_5d,
            volatility_20d = EXCLUDED.volatility_20d,
            was_rising_5d = EXCLUDED.was_rising_5d,
            was_rising_10d = EXCLUDED.was_rising_10d,
            was_rising_20d = EXCLUDED.was_rising_20d,
            pct_from_20d_high = EXCLUDED.pct_from_20d_high,
            pct_from_20d_low = EXCLUDED.pct_from_20d_low
        """

        self.cursor.execute(query)
        rows = self.cursor.rowcount
        self.conn.commit()
        print(f"✓ Monthly price features: {rows} (instrument, year, month) rows")

    def run_migrations(self):
        """Run SQL migrations to ensure schema is up to date."""
        base_dir = os.path.dirname(__file__)
        migrations = ['migration_gap_fix.sql', 'migration_monthly.sql']
        for migration_file in migrations:
            migration_path = os.path.join(base_dir, migration_file)
            if not os.path.exists(migration_path):
                continue
            print(f"Running migration: {migration_file}...")
            with open(migration_path, 'r') as f:
                sql = f.read()
            self.cursor.execute(sql)
            self.conn.commit()
            print(f"✓ {migration_file} applied")

    def run(self, instrument_id: int = None):
        """Run full transformation pipeline."""
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"Starting ML Transformation")
        print(f"{'='*60}\n")

        try:
            self.connect()
            self.run_migrations()

            # Transform in order
            self.transform_kpi_data(instrument_id)
            self.calculate_targets(instrument_id)
            self.transform_stock_prices(instrument_id)
            self.calculate_return_targets(instrument_id)
            self.calculate_pre_report_features(instrument_id)
            self.transform_holdings_data(instrument_id)

            # Monthly tables for monthly walk-forward model
            self.calculate_monthly_targets(instrument_id)
            self.calculate_monthly_price_features(instrument_id)

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

            self.cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(next_year_dividend_growth) as has_dividend,
                    COUNT(next_year_return) as has_return,
                    COUNT(next_year_excess_return) as has_excess_return,
                    COUNT(next_year_outperformed) as has_outperformed
                FROM ml_targets
            """)
            t = self.cursor.fetchone()
            print(f"ML Targets: {t[0]} total rows, {t[1]} with dividend targets, "
                  f"{t[2]} with return targets, {t[3]} with excess return, {t[4]} with outperformance")

            self.cursor.execute("SELECT COUNT(*) FROM ml_stock_prices")
            price_count = self.cursor.fetchone()[0]
            print(f"Stock Prices: {price_count} daily records")

            self.cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN was_rising_5d THEN 1 ELSE 0 END)
                FROM ml_pre_report_features
            """)
            prereport_stats = self.cursor.fetchone()
            if prereport_stats[0] > 0:
                rising_pct = (prereport_stats[1] / prereport_stats[0]) * 100
                print(f"Pre-Report Features: {prereport_stats[0]} reports, {rising_pct:.1f}% had rising prices before report")

            self.cursor.execute("""
                SELECT COUNT(*),
                       COUNT(insider_net_shares) as has_insider,
                       COUNT(buyback_count) as has_buyback
                FROM ml_holdings_features
            """)
            h = self.cursor.fetchone()
            print(f"Holdings Features: {h[0]} total rows, {h[1]} with insider data, {h[2]} with buyback data")

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
