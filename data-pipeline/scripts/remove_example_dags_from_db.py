#!/usr/bin/env python3
"""One-time: remove example_dags from Airflow DB so only project DAGs (dags-folder) remain.
Run from data-pipeline with: python scripts/remove_example_dags_from_db.py
Requires AIRFLOW_HOME=./airflow_home (or set in run_airflow*.sh).
"""
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DB_PATH = PIPELINE_ROOT / "airflow_home" / "airflow.db"

if not DB_PATH.exists():
    print(f"DB not found: {DB_PATH}", file=sys.stderr)
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)
try:
    cur = conn.execute("SELECT COUNT(*) FROM dag WHERE bundle_name = 'example_dags'")
    n = cur.fetchone()[0]
    conn.execute("DELETE FROM dag WHERE bundle_name = 'example_dags'")
    conn.commit()
    print(f"Removed {n} example DAGs. Remaining: 7 project DAGs (dags-folder).")
finally:
    conn.close()
