"""
Preprocessing DAG: audio 16kHz mono WAV, loudness normalization, silence trim; stratified split (dev 20%, test 70%, holdout 10%).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_preprocess(**kwargs):
    from scripts.preprocess_audio import run_preprocessing
    ok, fail = run_preprocessing(raw_subdir=None)
    if fail > 0 and ok == 0:
        raise RuntimeError("All preprocessing failed")
    return {"ok": ok, "fail": fail}


def _run_split(**kwargs):
    from scripts.stratified_split import run_split
    return run_split(staged_dir=PIPELINE_ROOT / "data" / "processed" / "staged", out_dir=PIPELINE_ROOT / "data" / "processed")


default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="preprocessing_dag",
    default_args=default_args,
    description="Audio preprocessing and stratified train/dev/holdout split",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "preprocessing"],
) as dag:
    preprocess = PythonOperator(
        task_id="preprocess_audio",
        python_callable=_run_preprocess,
    )
    stratified_split = PythonOperator(
        task_id="stratified_split",
        python_callable=_run_split,
    )
    preprocess >> stratified_split
