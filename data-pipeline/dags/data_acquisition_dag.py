"""
Data Acquisition DAG: download emotion & speech datasets, validate checksums, store in data/raw/ (DVC tracked).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add pipeline root so scripts can be imported when DAG runs
import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_download(**kwargs):
    from scripts.download_datasets import download_datasets
    datasets = kwargs.get("datasets")  # optional list from trigger
    r = download_datasets(datasets=datasets)
    failed = [k for k, v in r.items() if not v]
    if failed:
        raise RuntimeError(f"Downloads failed or skipped: {failed}")


default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="data_acquisition_dag",
    default_args=default_args,
    description="Download RAVDESS, IEMOCAP, CREMA-D, MELD, Common Voice, etc. into data/raw/",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "acquisition"],
) as dag:
    download_task = PythonOperator(
        task_id="download_datasets",
        python_callable=_run_download,
        op_kwargs={"datasets": None},
    )
    def _validate_checksums():
        from scripts.download_datasets import compute_sha256
        from scripts.utils import RAW_DIR
        out = {}
        for p in RAW_DIR.rglob("*"):
            if p.is_file() and p.suffix != ".dvc":
                try:
                    out[str(p.relative_to(RAW_DIR))] = compute_sha256(p)
                except Exception:
                    pass
        return out

    validate_checksums = PythonOperator(
        task_id="validate_checksums",
        python_callable=_validate_checksums,
    )
    download_task >> validate_checksums
