#!/usr/bin/env python3
"""
Run SQL migrations using Python/psycopg2.

Usage:
    python run_migration.py migration_add_prereport_features.sql
    python run_migration.py schema_ml.sql --recreate
"""

import psycopg2
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def run_migration(sql_file: str, db_config: dict):
    """Execute a SQL file against the database."""

    if not os.path.exists(sql_file):
        print(f"Error: File not found: {sql_file}")
        sys.exit(1)

    print(f"Reading {sql_file}...")
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql = f.read()

    print(f"Connecting to {db_config['database']}@{db_config['host']}...")
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    try:
        print(f"Executing migration...")
        cursor.execute(sql)
        conn.commit()
        print(f"✓ Migration complete!")

        # Show what tables exist now
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'ml_%'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        if tables:
            print(f"\nML tables in database:")
            for (table,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count} rows")

    except Exception as e:
        conn.rollback()
        print(f"✗ Error: {e}")
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Run SQL migrations')
    parser.add_argument('sql_file', help='SQL file to execute')
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', type=int, default=int(os.getenv('DB_PORT', 5432)))

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

    run_migration(args.sql_file, db_config)


if __name__ == '__main__':
    main()
