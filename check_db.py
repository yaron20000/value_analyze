#!/usr/bin/env python3
"""Quick check: what data do we actually have in api_raw_data?"""
import psycopg2, os, sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'borsdata'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'yourpass'),
        port=os.getenv('DB_PORT', 5432)
    )
except Exception as e:
    print(f"Cannot connect to DB: {e}")
    sys.exit(1)

cur = conn.cursor()

print("=" * 70)
print("WHAT DATA DO WE HAVE IN api_raw_data?")
print("=" * 70)

# 1. Endpoints overview
cur.execute("""
    SELECT endpoint_name,
           COUNT(*) as total,
           COUNT(*) FILTER (WHERE success) as ok,
           COUNT(*) FILTER (WHERE NOT success) as failed,
           MIN(fetch_timestamp)::date as first_fetch,
           MAX(fetch_timestamp)::date as last_fetch
    FROM api_raw_data
    GROUP BY endpoint_name
    ORDER BY endpoint_name
""")
rows = cur.fetchall()
if not rows:
    print("\n*** api_raw_data is EMPTY - no data fetched yet ***")
    conn.close()
    sys.exit(0)

print(f"\n{'Endpoint':<40} {'Total':>5} {'OK':>5} {'Fail':>5} {'First':>12} {'Last':>12}")
print("-" * 85)
for r in rows:
    print(f"{r[0]:<40} {r[1]:>5} {r[2]:>5} {r[3]:>5} {str(r[4]):>12} {str(r[5]):>12}")
print(f"\nTotal rows: {sum(r[1] for r in rows)}")

# 2. How many distinct instruments have data?
cur.execute("""
    SELECT COUNT(DISTINCT instrument_id)
    FROM api_raw_data
    WHERE instrument_id IS NOT NULL AND success = true
""")
print(f"\nDistinct instruments (per-instrument endpoints): {cur.fetchone()[0]}")

# 3. How many instruments in batch KPI data?
cur.execute("""
    SELECT COUNT(DISTINCT (inst_data->>'instrument')::integer)
    FROM api_raw_data,
         jsonb_array_elements(raw_data->'kpisList') AS inst_data
    WHERE endpoint_name LIKE 'kpi_%%_batch' AND success = true
""")
print(f"Distinct instruments (batch KPIs): {cur.fetchone()[0]}")

# 4. Sample: pick one instrument and show what we have
cur.execute("""
    SELECT instrument_id FROM api_raw_data
    WHERE instrument_id IS NOT NULL AND success = true
    LIMIT 1
""")
sample = cur.fetchone()
if sample:
    sid = sample[0]
    print(f"\n{'=' * 70}")
    print(f"SAMPLE: instrument_id = {sid}")
    print(f"{'=' * 70}")

    # What endpoints for this instrument?
    cur.execute("""
        SELECT endpoint_name, success,
               CASE WHEN raw_data IS NOT NULL THEN jsonb_typeof(raw_data) ELSE 'NULL' END,
               pg_column_size(raw_data) as data_bytes
        FROM api_raw_data
        WHERE instrument_id = %s
        ORDER BY endpoint_name
    """, (sid,))
    for r in cur.fetchall():
        print(f"  {r[0]:<35} success={r[1]}  type={r[2]:<8} size={r[3] or 0} bytes")

    # Show a snippet of stock prices JSON structure
    cur.execute("""
        SELECT raw_data
        FROM api_raw_data
        WHERE instrument_id = %s AND endpoint_name LIKE 'stockprices_%%' AND success = true
        LIMIT 1
    """, (sid,))
    r = cur.fetchone()
    if r and r[0]:
        import json
        data = r[0]
        keys = list(data.keys()) if isinstance(data, dict) else ['(array)']
        print(f"\n  Stock prices JSON keys: {keys}")
        if 'stockPricesList' in data:
            prices = data['stockPricesList']
            print(f"  stockPricesList: {len(prices)} entries")
            if prices:
                print(f"  First entry: {json.dumps(prices[0])}")
                print(f"  Last entry:  {json.dumps(prices[-1])}")

    # Show a snippet of yearly reports JSON structure
    cur.execute("""
        SELECT raw_data
        FROM api_raw_data
        WHERE instrument_id = %s AND endpoint_name LIKE 'reports_year_%%' AND success = true
        LIMIT 1
    """, (sid,))
    r = cur.fetchone()
    if r and r[0]:
        import json
        data = r[0]
        keys = list(data.keys()) if isinstance(data, dict) else ['(array)']
        print(f"\n  Yearly reports JSON keys: {keys}")
        if 'reports' in data:
            reports = data['reports']
            print(f"  reports: {len(reports)} entries")
            if reports:
                print(f"  First report keys: {list(reports[0].keys())}")
                print(f"  First report: {json.dumps(reports[0], indent=2)[:500]}")

# 5. Show batch KPI sample
cur.execute("""
    SELECT endpoint_name, kpi_name, raw_data
    FROM api_raw_data
    WHERE endpoint_name LIKE 'kpi_%%_batch' AND success = true
    LIMIT 1
""")
r = cur.fetchone()
if r:
    import json
    data = r[2]
    print(f"\n{'=' * 70}")
    print(f"SAMPLE BATCH KPI: {r[0]} (kpi_name={r[1]})")
    print(f"{'=' * 70}")
    print(f"  JSON keys: {list(data.keys())}")
    kpis_list = data.get('kpisList', [])
    print(f"  kpisList: {len(kpis_list)} instruments")
    if kpis_list:
        first = kpis_list[0]
        print(f"  First instrument entry keys: {list(first.keys())}")
        print(f"  instrument={first.get('instrument')}, values count={len(first.get('values', []))}")
        vals = first.get('values', [])
        if vals:
            print(f"  First value: {json.dumps(vals[0])}")
            print(f"  Last value:  {json.dumps(vals[-1])}")

conn.close()
print("\nDone.")
