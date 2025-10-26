# AIoT Homework 3.1 — Spam/Ham Classifier and Visualization

## Demo Site
```bat
[https://5114056011-aiot-hw3.streamlit.app/](https://5114056011-aiot-hw3.streamlit.app/)
```

## Short summary

This repository contains a small machine-learning baseline (spam vs ham classifier), developer utilities and an interactive Streamlit dashboard to explore model predictions and evaluation artifacts. It's designed for reproducible development, easy local runs, and simple CI smoke tests.

What you'll find here

- `data/` — sample datasets (raw and processed). Look for `data/sample_small.csv` and example SMS dataset files.
- `src/` — core scripts for preprocessing, training, inference, and evaluation (e.g., `preprocess.py`, `train.py`, `evaluate.py`, `infer.py`).
- `models/` — model artifacts (joblib), reports, and metadata (not checked into source control by default).
- `tools/streamlit_app.py` — interactive Streamlit app for batch and single-text inference and plotting (ROC/PR/confusion matrix).
- `openspec/` — project specs, change proposals, and the OpenSpec workflow artifacts.
- `tests/` — pytest tests and smoke tests.

## Quickstart (Windows cmd.exe)

Follow these steps to set up and run the full local workflow: create the virtualenv, preprocess the canonical dataset, train a model, run inference, and evaluate results.

1) Create a virtual environment and install dependencies

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Preprocess the canonical dataset

The repo contains a headerless SMS dataset (for example `data/sms_spam_no_header.csv`). Preprocess it into `data/processed/`.

```bat
python src/preprocess.py --input data/sms_spam_no_header.csv --output data/processed/sms_spam_clean.csv
```

If your project uses a different canonical file, replace the `--input` path accordingly.

3) Train the baseline model

Train a logistic regression baseline using the processed CSV. This example saves the model under `models/logreg/` and writes a report JSON.

```bat
python src/train.py --data data/processed/sms_spam_clean.csv --out-dir models/logreg --cv 2 --seed 42
```

Expected artifacts: `models/logreg/model.joblib` and `models/logreg/report.json`.

4) Run inference (batch)

Run bulk inference on a CSV to produce predictions. Adjust flags for input/output paths or column names as needed.

```bat
python src/infer.py --model models/logreg/model.joblib --input data/processed/sms_spam_clean.csv --output models/logreg/predictions.csv
```

5) Evaluate predictions

Evaluate the produced predictions against the ground-truth labels. The tool below expects a predictions CSV and a truth CSV or the processed dataset (which includes labels).

```bat
python src/evaluate.py --pred models/logreg/predictions.csv --truth data/processed/sms_spam_clean.csv
```

This should print/produce evaluation metrics (precision, recall, F1, ROC AUC) and optionally save plots to `models/logreg/reports/` if the evaluator supports it.

6) (Optional) Run the Streamlit dashboard

```bat
streamlit run tools/streamlit_app.py
```

Use the sidebar to select a demo CSV (from `data/` or `data/processed/`), choose text/label columns, and view predictions and ROC/PR/confusion matrix plots.

Notes & debugging tips

- Headerless CSVs (label,text) are supported: the app attempts to detect `text` or `text_clean` columns and will fall back to parsing headerless 2-column CSVs.
- If ROC/PR curves look identical across different CSV selections, check the Streamlit sidebar. The app includes debugging markers for the loaded CSV file and basic prediction-probability stats to help diagnose: loaded path, rows/cols, and score (min/mean/max/unique count).
- If you keep large model files, add them to `.gitignore` and use CI artifacts or an external storage service for distribution.

Contributing

- Branching: follow `feat/`, `fix/`, `docs/` prefixes.
- Tests: add pytest tests under `tests/`. CI should run `pytest` and a smoke ML run using the small sample dataset.
