# Iikshana Data Pipeline

Data pipeline for the Iikshana ADA-compliant courtroom visual aid system: data acquisition, preprocessing, validation, bias detection, and model evaluation (WER, BLEU, F1). Built with **Apache Airflow**, **DVC**, and **Great Expectations**-style validation.

## Project Overview

This pipeline supports evaluation of pre-trained models (e.g. Google Gemini 2.0 Flash, Chirp 3) on emotion recognition and multilingual speech datasets. It does **not** train models; it focuses on:

- **Data acquisition**: Download RAVDESS, IEMOCAP, CREMA-D, MELD, TESS, SAVEE, EMO-DB, Common Voice, etc.
- **Preprocessing**: 16 kHz mono WAV, loudness normalization, silence trimming, stratified splits (Dev 20%, Test 70%, Holdout 10%) with **no speaker overlap**.
- **Validation**: Schema checks (sample rate, duration, format), emotion label validation, quality reports.
- **Bias detection**: Slicing by demographics, emotion, language, audio quality; disparity reports and mitigation notes.
- **Evaluation**: STT (Chirp 3), translation, emotion detection → WER, BLEU, F1 vs targets (WER < 10%, BLEU > 0.40, F1 > 0.70).
- **Anomaly detection**: Missing/corrupt files, duration distribution, label imbalance; optional alerts.

## Environment Setup

- **Python**: 3.10+ recommended.
- **Airflow**: Install separately (see below). Use a dedicated venv for Airflow if you prefer.

### 1. Create virtual environment and install dependencies

```bash
cd data-pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Apache Airflow (optional, for DAGs)

```bash
pip install "apache-airflow>=2.7.0"
export AIRFLOW_HOME=/path/to/airflow_home   # e.g. ./airflow_home
airflow db init
airflow standalone   # or configure a scheduler + webserver
```

Point Airflow’s `dags_folder` to this repo’s `data-pipeline/dags` (or copy/symlink the `dags` folder into `AIRFLOW_HOME/dags`).

### 3. DVC (data versioning)

```bash
pip install dvc dvc-gs   # dvc-gs for Google Cloud Storage
cd data-pipeline
dvc init
# Optional: add remote (GCS or local)
# dvc remote add -d storage gs://your-bucket/dvc
```

## Running the Pipeline

### Without Airflow (scripts only)

Run stages in order:

```bash
cd data-pipeline
# 1. Download datasets (configure URLs in config/datasets.yaml)
python scripts/download_datasets.py [RAVDESS MELD ...]

# 2. Preprocess and split
python scripts/preprocess_audio.py
python scripts/stratified_split.py

# 3. Validate
python scripts/validate_schema.py

# 4. Bias report
python scripts/detect_bias.py

# 5. Evaluation (placeholder metrics without live APIs)
python scripts/evaluate_models.py

# 6. Anomaly check
python scripts/anomaly_check.py

# 7. Legal glossary (from repo data/legal_glossary)
python scripts/legal_glossary_prep.py
```

### With DVC

```bash
cd data-pipeline
dvc repro
```

### With Airflow

1. Ensure `data-pipeline` (or its `dags` folder) is in Airflow’s `dags_folder` and that the pipeline root is on `PYTHONPATH` when tasks run (or run tasks from `data-pipeline` as cwd).
2. Trigger DAGs in order (acquisition → preprocessing → validation; bias and evaluation can run after preprocessing):

   ```bash
   airflow dags trigger data_acquisition_dag
   airflow dags trigger preprocessing_dag
   airflow dags trigger validation_dag
   airflow dags trigger bias_detection_dag
   airflow dags trigger evaluation_dag
   ```

3. Use the Airflow UI (Gantt chart) to parallelize independent tasks and optimize bottlenecks (e.g. parallel dataset downloads).

## Data Sources

| Dataset        | Description                    | URL / note                          |
|----------------|--------------------------------|-------------------------------------|
| RAVDESS        | Emotional speech/song          | Zenodo (see `config/datasets.yaml`) |
| IEMOCAP        | Multimodal emotions            | License required                     |
| CREMA-D        | Emotion recognition            | License required                     |
| MELD           | Multimodal EmotionLines        | GitHub                               |
| TESS / SAVEE / EMO-DB | Emotion datasets       | Configure in `config/datasets.yaml`  |
| Common Voice   | Multilingual speech            | Hugging Face / Mozilla               |

Add or update URLs and checksums in `config/datasets.yaml`.

## DVC Commands

- **Track data** (after downloads and processing):

  ```bash
  dvc add data/raw/RAVDESS data/raw/IEMOCAP
  dvc add data/processed/dev data/processed/test data/processed/holdout
  dvc add data/legal_glossary
  git add *.dvc .gitignore
  git commit -m "Track pipeline data with DVC"
  ```

- **Pull data** (e.g. on another machine):

  ```bash
  dvc pull
  ```

- **Reproduce pipeline**:

  ```bash
  dvc repro
  ```

## Testing

```bash
cd data-pipeline
pip install -r requirements.txt
pytest tests/ -v
```

- `tests/test_preprocessing.py`: Audio normalization, resampling, `process_one`, `collect_audio_files`.
- `tests/test_validation.py`: Schema validation, manifest label checks.
- `tests/test_splitting.py`: Stratified split, no speaker overlap, `infer_speaker_id` / `infer_emotion`.

## Folder Structure

```
data-pipeline/
├── dags/
│   ├── data_acquisition_dag.py   # Download datasets
│   ├── preprocessing_dag.py      # Audio preprocessing & stratified split
│   ├── validation_dag.py         # Schema & quality checks
│   ├── evaluation_dag.py         # Model evaluation (WER, BLEU, F1)
│   └── bias_detection_dag.py     # Data slicing & bias report
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_audio.py
│   ├── stratified_split.py
│   ├── validate_schema.py
│   ├── detect_bias.py
│   ├── evaluate_models.py
│   ├── legal_glossary_prep.py
│   ├── anomaly_check.py
│   └── utils.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   └── test_splitting.py
├── data/
│   ├── raw/              # DVC tracked
│   ├── processed/        # dev, test, holdout, reports (DVC tracked)
│   └── legal_glossary/   # DVC tracked
├── config/
│   └── datasets.yaml
├── logs/
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Bias Detection Results

The bias detection step (`scripts/detect_bias.py`, `bias_detection_dag`) produces `data/processed/bias_report.json` with:

- Counts per emotion and per speaker.
- Disparities (e.g. strong class imbalance).
- Recommendations: stratified evaluation, confidence thresholding, re-sampling.

Summarize any findings (e.g. “Female voices in TESS show X% better emotion detection than male in SAVEE”) in this section after running on your data.

## Pipeline Optimization

- **Bottlenecks**: Use Airflow’s Gantt chart to find long-running tasks (e.g. large dataset downloads). Parallelize independent downloads in the acquisition DAG.
- **Preprocessing**: Scripts support batch processing; for very large corpora, consider multiprocessing or chunked runs (e.g. by dataset or speaker).
- **Validation**: Run validation and anomaly checks after preprocessing so failures are caught before evaluation.

## Replicating on Another Machine

1. Clone the repo and go to `data-pipeline`.
2. Create a venv, install dependencies: `pip install -r requirements.txt`.
3. Install Airflow if using DAGs; set `AIRFLOW_HOME` and point `dags_folder` to `data-pipeline/dags`.
4. Configure `config/datasets.yaml` (and optional DVC remote).
5. Pull data: `dvc pull` (if remotes are set).
6. Run pipeline: `dvc repro` or trigger Airflow DAGs in order.
7. Run tests: `pytest tests/ -v`.

## Logging

Scripts use Python `logging`; logs are written under `logs/` with timestamps. Each DAG task logs start, progress, and completion; validation and anomaly results are also logged.

## License

Same as the parent Iikshana repository.
