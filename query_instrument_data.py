#!/usr/bin/env python3
"""
Extract ALL raw data per instrument into a single flat CSV.
One row per instrument, with columns for every KPI/year combination,
stock price summary, and report financials.

Usage:
    python query_instrument_data.py
    python query_instrument_data.py --out data_dump.csv
"""
import psycopg2
import csv
import os
import argparse
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# KPI ID -> short column name (matches transform_to_ml.py)
KPI_MAP = {
    1: 'dividend_yield', 2: 'pe_ratio', 3: 'ps_ratio', 4: 'pb_ratio',
    10: 'ev_ebit', 11: 'ev_ebitda', 13: 'ev_fcf', 15: 'ev_sales', 19: 'peg_ratio',
    20: 'dividend_payout',
    23: 'fcf_per_share', 24: 'fcf_margin_pct', 27: 'earnings_fcf',
    28: 'gross_margin', 29: 'operating_margin', 30: 'net_margin',
    31: 'fcf_margin', 32: 'ebitda_margin',
    33: 'roe', 34: 'roa', 36: 'roc', 37: 'roic',
    39: 'equity_ratio', 40: 'debt_equity', 41: 'net_debt_pct',
    42: 'net_debt_ebitda', 44: 'current_ratio', 46: 'cash_pct',
    49: 'enterprise_value', 50: 'market_cap',
    51: 'ocf_margin', 53: 'revenue', 54: 'ebitda', 56: 'earnings',
    57: 'total_assets', 58: 'total_equity', 60: 'net_debt', 61: 'num_shares',
    62: 'ocf', 63: 'fcf', 64: 'capex',
    5: 'revenue_per_share', 6: 'eps', 7: 'dividend_per_share',
    8: 'book_value_per_share', 68: 'ocf_per_share',
    94: 'revenue_growth', 96: 'ebit_growth', 97: 'earnings_growth',
    98: 'dividend_growth', 99: 'book_value_growth', 100: 'assets_growth',
}


def main():
    parser = argparse.ArgumentParser(description='Extract all instrument data to CSV')
    parser.add_argument('--out', default='all_instrument_data.csv', help='Output CSV file')
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'borsdata'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'yourpass'),
        port=os.getenv('DB_PORT', 5432)
    )
    cur = conn.cursor()

    # ── 1. Instrument metadata ──────────────────────────────────────────
    print("Loading instrument metadata...")
    cur.execute("""
        SELECT raw_data->'instruments'
        FROM api_raw_data
        WHERE endpoint_name = 'instruments' AND success = true
        ORDER BY fetch_timestamp DESC LIMIT 1
    """)
    instruments_json = cur.fetchone()[0]
    meta = {}
    for inst in instruments_json:
        iid = inst.get('insId')
        meta[iid] = {
            'company_name': inst.get('name', ''),
            'ticker': inst.get('ticker', ''),
            'isin': inst.get('isin', ''),
            'instrument_type': inst.get('instrument', ''),
        }
    print(f"  {len(meta)} instruments in metadata")

    # ── 2. KPI values (from batch endpoints) ────────────────────────────
    # Each batch row: {kpiId: N, kpisList: [{instrument: X, values: [{y, p, v}]}]}
    print("Loading KPI data from batch endpoints...")
    cur.execute("""
        SELECT (raw_data->>'kpiId')::integer as kpi_id,
               raw_data->'kpisList' as kpis_list
        FROM api_raw_data
        WHERE endpoint_name LIKE 'kpi_%%_batch' AND success = true
    """)
    batch_rows = cur.fetchall()

    # instrument_id -> {col_name_year: value}
    kpi_data = defaultdict(dict)
    all_kpi_year_cols = set()

    for kpi_id, kpis_list in batch_rows:
        col_base = KPI_MAP.get(kpi_id)
        if not col_base:
            continue
        for entry in kpis_list:
            iid = entry.get('instrument')
            if iid is None:
                continue
            for val in entry.get('values', []):
                year = val.get('y')
                period = val.get('p')
                v = val.get('v')
                if year is None or v is None:
                    continue
                # Only full-year (period=5) for clean data; include quarterly marker if period != 5
                if period == 5:
                    col = f"{col_base}_{year}"
                else:
                    col = f"{col_base}_{year}_q{period}"
                kpi_data[iid][col] = v
                all_kpi_year_cols.add(col)

    print(f"  {len(kpi_data)} instruments with KPI data")
    print(f"  {len(all_kpi_year_cols)} KPI/year columns")

    # ── 3. Stock prices summary ─────────────────────────────────────────
    print("Loading stock prices...")
    cur.execute("""
        SELECT DISTINCT ON (instrument_id)
            instrument_id,
            raw_data->'stockPricesList' as prices
        FROM api_raw_data
        WHERE endpoint_name LIKE 'stockprices_%%'
          AND endpoint_name NOT LIKE '%%array%%'
          AND endpoint_name NOT LIKE '%%last%%'
          AND endpoint_name NOT LIKE '%%date%%'
          AND instrument_id IS NOT NULL
          AND success = true
        ORDER BY instrument_id, fetch_timestamp DESC
    """)
    price_data = {}
    for iid, prices in cur.fetchall():
        if not prices:
            continue
        dates = [p['d'] for p in prices if p.get('d')]
        closes = [p['c'] for p in prices if p.get('c') is not None]
        price_data[iid] = {
            'price_days': len(prices),
            'price_from': min(dates) if dates else None,
            'price_to': max(dates) if dates else None,
            'price_latest_close': closes[-1] if closes else None,
            'price_min_close': min(closes) if closes else None,
            'price_max_close': max(closes) if closes else None,
        }
    print(f"  {len(price_data)} instruments with stock prices")

    # ── 4. Yearly report financials ─────────────────────────────────────
    print("Loading yearly reports...")
    cur.execute("""
        SELECT DISTINCT ON (instrument_id)
            instrument_id,
            raw_data->'reports' as reports
        FROM api_raw_data
        WHERE endpoint_name LIKE 'reports_year_%%'
          AND instrument_id IS NOT NULL
          AND success = true
        ORDER BY instrument_id, fetch_timestamp DESC
    """)
    report_fields = [
        'revenues', 'earnings_Per_Share', 'dividend', 'operating_Income',
        'profit_Before_Tax', 'profit_To_Equity_Holders', 'total_Assets',
        'total_Equity', 'net_Debt', 'free_Cash_Flow', 'number_Of_Shares',
        'cash_And_Equivalents', 'current_Assets', 'non_Current_Assets',
        'current_Liabilities', 'non_Current_Liabilities',
        'cash_Flow_From_Operating_Activities', 'cash_Flow_From_Investing_Activities',
        'cash_Flow_From_Financing_Activities', 'gross_Income', 'net_Sales',
    ]
    report_data = defaultdict(dict)
    all_report_cols = set()

    for iid, reports in cur.fetchall():
        if not reports:
            continue
        for rpt in reports:
            year = rpt.get('year')
            period = rpt.get('period')
            if year is None or period != 5:  # full-year only
                continue
            for field in report_fields:
                val = rpt.get(field)
                if val is not None:
                    col = f"rpt_{field}_{year}"
                    report_data[iid][col] = val
                    all_report_cols.add(col)
    print(f"  {len(report_data)} instruments with yearly reports")
    print(f"  {len(all_report_cols)} report/year columns")

    # ── 5. Build CSV ────────────────────────────────────────────────────
    # Collect all instrument IDs
    all_ids = sorted(set(meta.keys()) | set(kpi_data.keys()) | set(price_data.keys()) | set(report_data.keys()))

    # Build column order
    fixed_cols = ['instrument_id', 'company_name', 'ticker', 'isin', 'instrument_type']
    price_cols = ['price_days', 'price_from', 'price_to', 'price_latest_close', 'price_min_close', 'price_max_close']
    sorted_kpi_cols = sorted(all_kpi_year_cols)
    sorted_report_cols = sorted(all_report_cols)
    all_cols = fixed_cols + price_cols + sorted_kpi_cols + sorted_report_cols

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()

        for iid in all_ids:
            row = {'instrument_id': iid}
            # Metadata
            m = meta.get(iid, {})
            row.update(m)
            # Prices
            row.update(price_data.get(iid, {}))
            # KPIs
            row.update(kpi_data.get(iid, {}))
            # Reports
            row.update(report_data.get(iid, {}))
            writer.writerow(row)

    print(f"\n{'=' * 60}")
    print(f"DONE: {out_path}")
    print(f"  {len(all_ids)} instruments (rows)")
    print(f"  {len(all_cols)} columns total")
    print(f"    - {len(fixed_cols)} identity columns")
    print(f"    - {len(price_cols)} price columns")
    print(f"    - {len(sorted_kpi_cols)} KPI value columns")
    print(f"    - {len(sorted_report_cols)} report value columns")

    conn.close()


if __name__ == '__main__':
    main()
