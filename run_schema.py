"""Run schema_ml.sql against the database using psycopg2."""
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

with open(os.path.join(os.path.dirname(__file__), 'schema_ml.sql'), 'r') as f:
    sql = f.read()

cur = conn.cursor()
cur.execute(sql)
print("schema_ml.sql executed successfully")

# Verify the view has the expected column
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'ml_targets' ORDER BY ordinal_position")
cols = [row[0] for row in cur.fetchall()]
print(f"ml_targets columns: {cols}")
assert 'next_year_excess_return' in cols, "ERROR: next_year_excess_return not found in ml_targets!"
print("Verified: next_year_excess_return column exists")

cur.close()
conn.close()
