#!/usr/bin/env python3
"""
Trigger-only runner
====================
1. Ensures the postgres-borsdata Docker container is running
2. Runs trigger_predict.py (re-predicts if new data detected, or --force)

Usage:
    python run_trigger.py
    python run_trigger.py --force
    python run_trigger.py --db-password mypass --force
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
    parser = argparse.ArgumentParser(description="Start DB and run trigger_predict.py")
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', default=os.getenv('DB_PORT', '5432'))
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--force', action='store_true',
                        help='Force re-prediction even if no new data')
    args = parser.parse_args()

    if not args.db_password:
        print("Error: DB password required (--db-password or DB_PASSWORD env var)")
        sys.exit(1)

    ensure_docker_db()

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
    print("\nDone.")


if __name__ == '__main__':
    main()
