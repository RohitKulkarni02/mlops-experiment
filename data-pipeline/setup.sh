#!/usr/bin/env bash
# One-time setup for Airflow in Docker (professor pattern).
# Run from data-pipeline: bash setup.sh
# Then start with: docker compose up
# See README and https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Airflow Docker one-time setup ==="

# Required dirs (dags, scripts, config already exist; ensure logs and data exist)
mkdir -p ./dags ./logs ./plugins ./data/raw ./data/processed

# AIRFLOW_UID so container files are owned by host user (official requirement)
touch .env
grep -q '^AIRFLOW_UID=' .env 2>/dev/null || echo "AIRFLOW_UID=$(id -u)" >> .env

# Shared credentials (same for whole team when using Docker)
if ! grep -q "_AIRFLOW_WWW_USER_USERNAME" .env 2>/dev/null; then
  echo "_AIRFLOW_WWW_USER_USERNAME=airflow" >> .env
  echo "_AIRFLOW_WWW_USER_PASSWORD=airflow" >> .env
  echo "Added default credentials to .env (airflow/airflow)"
fi

# Project dir for mounts (defaults to current dir in compose)
if ! grep -q "AIRFLOW_PROJ_DIR" .env 2>/dev/null; then
  echo "AIRFLOW_PROJ_DIR=${SCRIPT_DIR}" >> .env
fi

echo "Created/updated .env (AIRFLOW_UID and credentials)."

# Clean previous run (optional; uncomment to reset DB and volumes)
# docker compose down -v
# rm -rf ./logs ./plugins  # only if you want a clean logs/plugins

echo "Initializing Airflow database and admin user (may take a minute)..."
docker compose up airflow-init

echo ""
echo "Done. Start Airflow with: docker compose up"
echo "Then open http://localhost:8080 and log in with the credentials in .env"
