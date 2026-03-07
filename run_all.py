#!/usr/bin/env python3
"""
Full pipeline runner
=====================
1. Ensures the postgres-borsdata Docker container is running
2. fetch_and_store.py   – pulls fresh data from Borsdata API
3. trigger_predict.py   – re-predicts if new data detected (or --force)
4. train_model.py       – full walk-forward backtest / training

Usage:
    python run_all.py
    python run_all.py --force
    python run_all.py --api-key YOUR_KEY --db-password mypass --nordics
    python run_all.py --skip-fetch          # trigger + train only
    python run_all.py --skip-train          # fetch + trigger only
"""

import argparse
import subprocess
import sys
import time
import os

from dotenv import load_dotenv

load_dotenv()


def ensure_docker_db():
    """Start postgres-borsdata if it isn't already running."""
    result = subprocess.run(
        ['docker', 'inspect', '--format', '{{.State.Running}}', 'postgres-borsdata'],
        capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip() == 'true':
        print("[docker] postgres-borsdata is already running.")
        return

    print("[docker] Starting postgres-borsdata container...")
    r = subprocess.run(['docker', 'start', 'postgres-borsdata'], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[docker] ERROR: {r.stderr.strip()}")
        sys.exit(1)

    print("[docker] Waiting for PostgreSQL to be ready...")
    time.sleep(5)
    print("[docker] PostgreSQL ready.")


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: fetch → trigger → train")
    # Shared DB args
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', default=os.getenv('DB_PORT', '5432'))
    # fetch_and_store args
    parser.add_argument('--api-key', default=os.getenv('BORSDATA_API_KEY'),
                        help='Borsdata API key')
    parser.add_argument('--nordics', action='store_true',
                        help='Fetch all Nordic instruments')
    parser.add_argument('--instruments',
                        help='Comma-separated instrument IDs to fetch')
    parser.add_argument('--force-refetch', action='store_true',
                        help='Force re-fetch even if data exists')
    # trigger_predict args
    parser.add_argument('--force', action='store_true',
                        help='Force re-prediction even if no new data')
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--output-dir', default='results')
    # train_model args
    parser.add_argument('--min-train-months', type=int, default=36)
    # Pipeline control
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetch_and_store.py')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip train_model.py (full backtest)')
    args = parser.parse_args()

    if not args.db_password:
        print("Error: DB password required (--db-password or DB_PASSWORD env var)")
        sys.exit(1)

    ensure_docker_db()

    # ── 1. fetch_and_store ───────────────────────────────────────────────────
    if not args.skip_fetch:
        if not args.api_key:
            print("Error: API key required (--api-key or BORSDATA_API_KEY env var)")
            sys.exit(1)
        fetch_cmd = [
            sys.executable, 'fetch_and_store.py',
            args.api_key,
            '--db-host', args.db_host,
            '--db-name', args.db_name,
            '--db-user', args.db_user,
            '--db-password', args.db_password,
            '--db-port', str(args.db_port),
        ]
        if args.nordics:
            fetch_cmd.append('--nordics')
        if args.instruments:
            fetch_cmd.extend(['--instruments', args.instruments])
        if args.force_refetch:
            fetch_cmd.append('--force-refetch')
        run(fetch_cmd, "fetch_and_store.py")

    # ── 2. trigger_predict ───────────────────────────────────────────────────
    trigger_cmd = [
        sys.executable, 'trigger_predict.py',
        '--db-host', args.db_host,
        '--db-name', args.db_name,
        '--db-user', args.db_user,
        '--db-password', args.db_password,
        '--db-port', str(args.db_port),
        '--output-dir', args.output_dir,
        '--top-n', str(args.top_n),
    ]
    if args.force:
        trigger_cmd.append('--force')
    run(trigger_cmd, "trigger_predict.py")

    # ── 3. train_model (full backtest) ───────────────────────────────────────
    if not args.skip_train:
        train_cmd = [
            sys.executable, 'train_model.py',
            '--db-host', args.db_host,
            '--db-name', args.db_name,
            '--db-user', args.db_user,
            '--db-password', args.db_password,
            '--db-port', str(args.db_port),
            '--output-dir', args.output_dir,
            '--top-n', str(args.top_n),
            '--min-train-months', str(args.min_train_months),
        ]
        run(train_cmd, "train_model.py")

    print("\nAll steps completed successfully.")


if __name__ == '__main__':
    main()
