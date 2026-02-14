"""
Validation DAG: schema checks (sample rate, duration, format), emotion labels; quality report and JSON schema.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_validation(**kwargs):
    from scripts.validate_schema import run_validation
    from scripts.utils import PROCESSED_DIR
    r = run_validation(data_dir=PIPELINE_ROOT / "data" / "processed", schema_out=PROCESSED_DIR / "audio_schema.json", report_out=PROCESSED_DIR / "quality_report.json")
    if r.get("failed", 0) > 0 and r.get("passed", 0) == 0:
        raise RuntimeError("All validation checks failed")
    return r


default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="validation_dag",
    default_args=default_args,
    description="Data validation and schema generation (Great Expectations style)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "validation"],
) as dag:
    validate = PythonOperator(
        task_id="validate_schema_and_quality",
        python_callable=_run_validation,
    )
