#!/usr/bin/env bash
# Step 1.2 + 1.3 prep: create AIRFLOW_HOME, run db migrate, set dags_folder to this repo's dags.
# Run from data-pipeline: ./setup_airflow.sh
# Then start Airflow with: ./run_airflow.sh  (or airflow standalone with AIRFLOW_HOME set)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DAGS_ABSOLUTE="${SCRIPT_DIR}/dags"
AIRFLOW_HOME_DIR="${SCRIPT_DIR}/airflow_home"

if [[ ! -d ".venv" ]]; then
  echo "Create venv first: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
export AIRFLOW_HOME="$AIRFLOW_HOME_DIR"
mkdir -p "$AIRFLOW_HOME_DIR"

if [[ ! -f "$AIRFLOW_HOME_DIR/airflow.cfg" ]]; then
  echo "Running airflow db migrate..."
  airflow db migrate
fi

# Point dags_folder at this repo's dags (so DAGs find scripts/ and data/ via parent path)
if [[ -f "$AIRFLOW_HOME_DIR/airflow.cfg" ]]; then
  if grep -q "dags_folder = $DAGS_ABSOLUTE" "$AIRFLOW_HOME_DIR/airflow.cfg" 2>/dev/null; then
    echo "dags_folder already set to $DAGS_ABSOLUTE"
  else
    # Replace the dags_folder line (works on both macOS and Linux with sed)
    if sed --version 2>/dev/null | grep -q GNU; then
      sed -i "s|^dags_folder = .*|dags_folder = $DAGS_ABSOLUTE|" "$AIRFLOW_HOME_DIR/airflow.cfg"
    else
      sed -i '' "s|^dags_folder = .*|dags_folder = $DAGS_ABSOLUTE|" "$AIRFLOW_HOME_DIR/airflow.cfg"
    fi
    echo "Set dags_folder = $DAGS_ABSOLUTE in $AIRFLOW_HOME_DIR/airflow.cfg"
  fi
fi

echo "Done. Start Airflow with: export AIRFLOW_HOME=$AIRFLOW_HOME_DIR && airflow standalone"
echo "Or run: ./run_airflow.sh"
