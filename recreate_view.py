"""Recreate the ml_training_data view without dropping any tables.
Safe to run anytime the view gets out of sync with schema_ml.sql."""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    database=os.getenv('DB_NAME', 'borsdata'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD'),
    port=int(os.getenv('DB_PORT', 5432))
)
conn.autocommit = True
cur = conn.cursor()

# Extract only the view definition from schema_ml.sql
schema_path = os.path.join(os.path.dirname(__file__), 'schema_ml.sql')
with open(schema_path, 'r') as f:
    sql = f.read()

# Find the view creation block
start = sql.index('DROP VIEW IF EXISTS ml_training_data')
end = sql.index(';', sql.index('ORDER BY f.instrument_id, f.year', start)) + 1
view_sql = sql[start:end]

cur.execute(view_sql)
print("ml_training_data view recreated successfully")

# Verify columns
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'ml_training_data'
    ORDER BY ordinal_position
""")
cols = [row[0] for row in cur.fetchall()]
print(f"View has {len(cols)} columns")

expected = ['next_year_excess_return', 'market_median_return']
for col in expected:
    if col in cols:
        print(f"  OK: {col}")
    else:
        print(f"  MISSING: {col}")

cur.close()
conn.close()
