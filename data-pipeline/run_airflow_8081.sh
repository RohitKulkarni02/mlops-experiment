#!/usr/bin/env bash
# Run THIS project's Airflow on port 8081 so it doesn't conflict with another Airflow on 8080.
# If you see 87 DAGs at http://localhost:8080, that's the other Airflow — open http://localhost:8081 instead.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Create venv first: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f "airflow_home/airflow.cfg" ]]; then
  echo "First-time setup: run ./setup_airflow.sh"
  exit 1
fi

source .venv/bin/activate
export AIRFLOW_HOME="${SCRIPT_DIR}/airflow_home"
export AIRFLOW__API__PORT=8081
export AIRFLOW__CORE__LOAD_EXAMPLES=false
echo "AIRFLOW_HOME=$AIRFLOW_HOME"
echo "API port: 8081 (so you don't conflict with another Airflow on 8080)"
echo ""
echo "  → Open http://localhost:8081 (not 8080!)"
echo "  → Click **DAGs** in the top nav. You should see only this project's DAGs (~7), including full_pipeline_dag."
echo ""
exec airflow standalone
