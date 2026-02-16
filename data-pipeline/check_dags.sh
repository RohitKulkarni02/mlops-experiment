#!/usr/bin/env bash
# Run from data-pipeline: verifies AIRFLOW_HOME and that full_pipeline_dag is visible to Airflow.
# If this shows full_pipeline_dag but the UI does not, start the UI from here: ./run_airflow.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export AIRFLOW_HOME="${SCRIPT_DIR}/airflow_home"
echo "Using AIRFLOW_HOME=$AIRFLOW_HOME"
echo "DAGs folder in config: $(grep '^dags_folder' airflow_home/airflow.cfg 2>/dev/null || true)"
echo ""
echo "DAGs visible to Airflow (look for full_pipeline_dag):"
airflow dags list 2>/dev/null | grep -E "dag_id|full_pipeline|data_acquisition|preprocessing|validation|bias_detection|evaluation|anomaly" || airflow dags list 2>/dev/null | head -30
echo ""
echo "If full_pipeline_dag appears above but not in the UI, start Airflow from this directory: ./run_airflow.sh"
echo "Then open the DAGs tab and search for 'full' or 'pipeline'."
