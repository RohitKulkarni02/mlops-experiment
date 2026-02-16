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
    import logging
    import traceback
    task_logger = logging.getLogger(__name__)

    def log(msg: str) -> None:
        task_logger.info(msg)
        print(msg, flush=True)

    log("=== data_acquisition_dag: download task starting ===")
    from scripts.utils import CONFIG_DIR, PIPELINE_ROOT
    config_path = CONFIG_DIR / "datasets.yaml"
    log(f"PIPELINE_ROOT={PIPELINE_ROOT}, config path={config_path}, config exists={config_path.exists()}")

    datasets = kwargs.get("datasets")  # optional list from trigger
    log(f"datasets param: {datasets} (None = use all from config)")

    try:
        from scripts.download_datasets import download_datasets
        log("Calling download_datasets() ...")
        r = download_datasets(datasets=datasets)
        log(f"download_datasets() returned: {r}")
    except Exception as e:
        log(f"download_datasets() raised: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        raise

    failed = [k for k, v in r.items() if not v]
    if failed:
        log(f"Download failures (datasets with URL that failed): {failed}")
        raise RuntimeError(f"Downloads failed: {failed}")
    log("All downloads with configured URLs succeeded or were already present. (Datasets with no URL were skipped.)")


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


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=0),
}


# None = download all datasets that have a URL in config/datasets.yaml. Set to e.g. ["RAVDESS", "MELD"] to limit.
DOWNLOAD_DATASETS_FILTER = None


def get_tasks(dag: DAG):
    """Return tasks for this stage (for use in full_pipeline_dag). Last task is the final one in the chain."""
    download_task = PythonOperator(
        task_id="download_datasets",
        python_callable=_run_download,
        op_kwargs={"datasets": DOWNLOAD_DATASETS_FILTER},
        execution_timeout=timedelta(hours=2),
        dag=dag,
    )
    validate_checksums = PythonOperator(
        task_id="validate_checksums",
        python_callable=_validate_checksums,
        dag=dag,
    )
    download_task >> validate_checksums
    return [download_task, validate_checksums]


with DAG(
    dag_id="data_acquisition_dag",
    default_args=_default_args,
    description="Download RAVDESS, IEMOCAP, CREMA-D, MELD, Common Voice, etc. into data/raw/",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "acquisition"],
) as dag:
    _tasks = get_tasks(dag)
