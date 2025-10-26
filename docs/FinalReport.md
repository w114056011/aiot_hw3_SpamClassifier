# Final Report: AIoT Homework 3.1 — Spam/Ham Classifier & Visualization

Date: 2025-10-26

This final report aggregates the project's OpenSpec documentation, proposals, and a concise review of the implemented artifacts in the repository. It is intended for submission/delivery and as a single-entry summary for graders or maintainers.

## Executive summary

This repository provides a minimal, reproducible machine learning baseline for short text classification (spam vs ham), plus developer-oriented visualization tools (a Streamlit dashboard and static report generation). The goals are reproducibility, clear artifacts for grading, and a lightweight interactive UI for exploration and demoing.

Key deliverables

- Baseline ML pipeline (preprocess → train → evaluate → infer) using scikit-learn (TF-IDF + Logistic Regression).
- Model artifact management conventions (joblib + metadata.json) and an evaluation report (`report.json`) containing precision/recall/F1/ROC AUC and related artifacts.
- Streamlit dashboard (`tools/streamlit_app.py`) for single-item and batch inference with ROC/PR/confusion matrix visualizations.
- OpenSpec proposals and change directories under `openspec/changes/` that document the project intent, tasks, and validation steps.

## Project status and review

The codebase includes:

- `openspec/` — project-level spec (`project.md`) and change proposals for:
  - `add-sms-spam-baseline` — baseline ML pipeline (preprocessing, training, evaluation).
  - `add-visualization-streamlit` — Streamlit UI + report generator.
  - `add-mqtt-device-simulator` — optional simulator for IoT testing.

- `src/` — scripts for model pipelines (preprocess, train, evaluate, infer, report generator).
- `tools/streamlit_app.py` — interactive UI (select CSV from `data/` and `data/processed/`, manual text/label selectors, sampling buttons, ROC/PR plots, threshold slider).
- `data/` — sample datasets such as `data/sample_small.csv` and headerless SMS dataset examples.
- `tests/` — pytest smoke tests validating core functionality.

Known issues / diagnostic notes

- Streamlit session-state and version differences: some APIs like `st.experimental_rerun()` are guarded to avoid breaking older Streamlit installations.
- If you observe identical ROC/PR curves across different CSV selections, likely causes are:
  - Same file selected twice (path not changing),
  - Wrong text/label column mapping so actual inputs are identical,
  - Model outputs constant scores across inputs (unique score count = 1), or
  - Streamlit UI did not rerun to pick up the new dataset. The app includes debugging helpers (loaded CSV path, rows/cols, and `pred_proba` stats) to make this visible.

## OpenSpec excerpts

Below is the project-level OpenSpec content (excerpted from `openspec/project.md`):

---

(Excerpt)

- Baseline pipeline (contract)
  - Inputs: CSV or Parquet with `id`,`text`,`label`.
  - Preprocessing: lowercase, strip HTML, TF-IDF vectorization.
  - Model: scikit-learn LogisticRegression. Output artifacts saved via joblib.

- Training recipe
  - Train/test split with stratification; optional stratified k-fold CV.
  - Pipeline: TfidfVectorizer → LogisticRegression. Grid search over C, max_features, ngram_range.

- Metrics: Primary F1; secondary precision/recall/accuracy/ROC AUC; report confusion matrix and suggested thresholds.

(See full `openspec/project.md` in the repo for complete details.)

---

### Change proposals (summary)

- `add-sms-spam-baseline` — Adds preprocessing, training, evaluation scripts, sample dataset, and CI smoke-test. Produces `data/processed/sms_spam_clean.csv`, `models/<name>/report.json`, and plots.

- `add-visualization-streamlit` — Adds Streamlit UI and `src/report.py` to produce static reports for grading and an interactive dashboard for manual inspection.

- `add-mqtt-device-simulator` — Proposal to add a small MQTT device simulator for IoT testing and CI integration.

(Full proposal texts live under `openspec/changes/*/proposal.md`.)

## How to run (summary)

1) Create virtualenv and install dependencies

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Train a baseline (example)

```bat
python src/train.py --data data/sample_small.csv --out-dir models/logreg --cv 2 --seed 42
```

3) Run Streamlit dashboard

```bat
streamlit run tools/streamlit_app.py
```

4) Run tests

```bat
pytest -q
```

## Artifacts submitted

- This `docs/FinalReport.md` — aggregated final report and OpenSpec summary.
- `README.md` — top-level usage and quickstart (project root).

## Next steps and recommendations

- Finish/validate CI that runs an ML smoke test and publishes `models/` artifacts to CI storage.
- Add `OWNERS` and a small CONTRIBUTING guide with PR/branching rules.
- Optionally add a checksum/hash of loaded CSVs in the Streamlit app to ensure dataset changes are detected reliably.

---

If you want, I can also:
- Add small unit tests that validate the Streamlit app's CSV-loading logic (reading headerless files, detecting text/label columns),
- Add a `models/README.md` describing how to store full model artifacts and how graders can fetch them, or
- Produce an exported static HTML report (via `src/report.py`) that can be submitted with grading.

