#!/usr/bin/env bash
# Run Airflow from step 1.2: uses AIRFLOW_HOME=./airflow_home and dags_folder=./dags (set in airflow.cfg).
# From repo root: cd data-pipeline && ./run_airflow.sh
# Or: cd data-pipeline && source .venv/bin/activate && AIRFLOW_HOME="$(pwd)/airflow_home" airflow standalone

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Create venv first: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f "airflow_home/airflow.cfg" ]]; then
  echo "First-time setup: run 'source .venv/bin/activate && AIRFLOW_HOME=\"$(pwd)/airflow_home\" airflow db migrate'"
  echo "Then ensure airflow.cfg has: dags_folder = $(pwd)/dags"
  exit 1
fi

source .venv/bin/activate
export AIRFLOW_HOME="${SCRIPT_DIR}/airflow_home"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
echo "AIRFLOW_HOME=$AIRFLOW_HOME (dags_folder=$SCRIPT_DIR/dags)"
echo ""
echo "  → Open http://localhost:8080 and click **DAGs** in the top nav."
echo "  → If you see 87 DAGs (examples), you're on the WRONG Airflow. Run ./run_airflow_8081.sh and open http://localhost:8081 instead."
echo ""
exec airflow standalone
