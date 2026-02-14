"""
Evaluation DAG: run STT (Chirp 3), translation, emotion detection on test set; compute WER, BLEU, F1; compare to targets.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_evaluation(**kwargs):
    from scripts.evaluate_models import run_evaluation
    return run_evaluation(data_dir=PIPELINE_ROOT / "data" / "processed", metrics_path=PIPELINE_ROOT / "data" / "processed" / "evaluation_metrics.json", use_live_apis=False)


default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="evaluation_dag",
    default_args=default_args,
    description="Model evaluation: WER, BLEU, F1 vs targets",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "evaluation"],
) as dag:
    eval_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=_run_evaluation,
    )
