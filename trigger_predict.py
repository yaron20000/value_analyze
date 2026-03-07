#!/usr/bin/env python3
"""
Event-Triggered Re-Prediction
==============================
Checks whether new insider transactions or report dates have arrived since the
last prediction run.  If so (or if --force is given), it:
  1. Re-transforms holdings data (ml_holdings_features) via transform_to_ml.py
  2. Re-runs current-month prediction via train_model.py --predict-only

Intended to be run after fetch_and_store.py has ingested fresh data.

Usage:
    python trigger_predict.py
    python trigger_predict.py --force
    python trigger_predict.py --db-password mypass --output-dir results
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

LAST_PREDICTION_FILE = 'last_prediction.json'


def load_last_prediction_time(output_dir: str):
    """Return the datetime of the last prediction run, or None if not found."""
    filepath = os.path.join(output_dir, LAST_PREDICTION_FILE)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
        return datetime.fromisoformat(data['timestamp'])
    except Exception:
        return None


def get_latest_insider_date(conn):
    """Return the most recent transactionDate across all insider records."""
    query = """
        SELECT MAX((tx->>'transactionDate')::date)
        FROM api_raw_data,
             jsonb_array_elements(raw_data->'insider') AS tx
        WHERE endpoint_name = 'holdings_insider'
          AND (tx->>'transactionDate') IS NOT NULL
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return row[0] if row and row[0] else None
    except Exception as e:
        print(f"  Warning: could not query insider dates: {e}")
        conn.rollback()
        return None


def get_latest_report_date(conn):
    """Return the most recent report_date in ml_pre_report_features."""
    query = "SELECT MAX(report_date) FROM ml_pre_report_features"
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return row[0] if row and row[0] else None
    except Exception as e:
        print(f"  Warning: could not query report dates: {e}")
        conn.rollback()
        return None


def run_subprocess(cmd: list, description: str) -> bool:
    """Run a subprocess command, stream output, return True on success."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"  Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n  ERROR: {description} exited with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Trigger current-month re-prediction when new insider or report data arrives')
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', type=int, default=int(os.getenv('DB_PORT', 5432)))
    parser.add_argument('--output-dir', default='results',
                        help='Output directory (must match train_model.py --output-dir)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Passed through to train_model.py --top-n')
    parser.add_argument('--force', action='store_true',
                        help='Re-predict even if no new data is detected')
    args = parser.parse_args()

    if not args.db_password:
        print("Error: DB password required (--db-password or DB_PASSWORD env var)")
        sys.exit(1)

    # --- Check last prediction timestamp ---
    last_run = load_last_prediction_time(args.output_dir)
    if last_run:
        print(f"Last prediction run: {last_run.isoformat()}")
    else:
        print("No previous prediction timestamp found — will run prediction.")

    # --- Connect and query latest data dates ---
    conn = psycopg2.connect(
        host=args.db_host, database=args.db_name,
        user=args.db_user, password=args.db_password, port=args.db_port
    )
    try:
        latest_insider = get_latest_insider_date(conn)
        latest_report = get_latest_report_date(conn)
    finally:
        conn.close()

    print(f"Latest insider transaction date in DB: {latest_insider}")
    print(f"Latest report date in DB:              {latest_report}")

    # --- Decide whether to re-predict ---
    trigger_reason = None

    if args.force:
        trigger_reason = 'forced (--force flag)'
    elif last_run is None:
        trigger_reason = 'no previous prediction found'
    else:
        last_run_date = last_run.date()
        if latest_insider and latest_insider > last_run_date:
            trigger_reason = f'new insider data ({latest_insider} > last run {last_run_date})'
        elif latest_report and latest_report > last_run_date:
            trigger_reason = f'new report filed ({latest_report} > last run {last_run_date})'

    if trigger_reason is None:
        print(f"\nNo new data since last prediction ({last_run}). Nothing to do.")
        sys.exit(0)

    print(f"\nTrigger: {trigger_reason}")

    # --- Step 1: refresh ml_holdings_features ---
    db_args = [
        '--db-host', args.db_host,
        '--db-name', args.db_name,
        '--db-user', args.db_user,
        '--db-password', args.db_password,
        '--db-port', str(args.db_port),
    ]
    transform_cmd = [sys.executable, 'transform_to_ml.py', '--holdings-only'] + db_args
    ok = run_subprocess(transform_cmd, 'transform_to_ml.py --holdings-only')
    if not ok:
        print("Aborting: holdings transform failed.")
        sys.exit(1)

    # --- Step 2: run current-month prediction ---
    predict_cmd = [
        sys.executable, 'train_model.py',
        '--predict-only',
        '--output-dir', args.output_dir,
        '--top-n', str(args.top_n),
    ] + db_args
    ok = run_subprocess(predict_cmd, 'train_model.py --predict-only')
    if not ok:
        print("Prediction failed.")
        sys.exit(1)

    print(f"\nDone. Prediction triggered by: {trigger_reason}")


if __name__ == '__main__':
    main()
