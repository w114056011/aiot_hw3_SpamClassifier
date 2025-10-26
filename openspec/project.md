## Machine Learning: Email classification (spam vs ham)

This repository can also host a small machine learning pipeline for classifying emails as spam or ham. Below is a baseline pipeline, metrics to track, and easy local workflows for training and inference using Logistic Regression as the baseline model.

### Baseline pipeline (contract)
- Inputs: CSV or Parquet file with rows: `id`, `text`, `label` (label is `spam` or `ham` or 1/0). Optional columns: `timestamp`, `source`.
- Preprocessing: lowercase, strip HTML, remove headers/footers (optional), tokenize, TF-IDF vectorization (scikit-learn's TfidfVectorizer), optional n-grams (1,2).
- Model: scikit-learn LogisticRegression (solver='liblinear' or 'saga' for larger data), with class weighting if dataset is imbalanced.
- Output: serialized model and vectorizer saved with joblib (e.g., `models/logreg.joblib`, `models/tfidf.joblib`). Also produce an evaluation report (JSON/CSV) with metrics and confusion matrix.

### Training recipe
1. Split data: train/test split (e.g., 80/20) with stratification on label. Optionally use stratified k-fold cross-validation (5 folds) for robust estimates.
2. Build pipeline: TfidfVectorizer -> LogisticRegression (use sklearn.pipeline.Pipeline).
3. Hyperparameter grid (baseline): C in [0.01, 0.1, 1, 10], max_features [None, 20000], ngram_range [(1,1),(1,2)]. Use GridSearchCV with scoring='f1' (or 'roc_auc').
4. Final model: retrain on full training set with best params and evaluate on test set.

### Metrics (report these)
- Primary: F1-score (harmonic mean of precision and recall) — balances false positives and false negatives.
- Secondary: Precision, Recall, Accuracy, ROC AUC.
- Threshold/Calibration: For production use, also report precision-recall curve and suggested probability thresholds to meet recall or precision targets.
- Confusion Matrix: TP/FP/FN/TN counts and normalized rates.
- Stability: cross-validation mean/std for the chosen metric (e.g., F1 mean ± std over folds).

### Quick local workflows (commands)
Prereqs: Python 3.8+, virtualenv, pip. Add `requirements.txt` with: scikit-learn, pandas, joblib, numpy, python-dateutil (if needed), pytest (for tests).

1) Create virtualenv and install deps:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Train (example):

```bat
python src/train.py --data data/emails.csv --out-dir models --cv 5 --seed 42
```

Expected artifacts: `models/logreg.joblib`, `models/tfidf.joblib`, `models/report.json`.

3) Inference (single file or interactive):

```bat
python src/infer.py --model models/logreg.joblib --vectorizer models/tfidf.joblib --input samples/email1.txt
```

Or bulk inference:

```bat
python src/infer.py --model models/logreg.joblib --vectorizer models/tfidf.joblib --input data/unlabeled.csv --output predictions.csv
```

4) Quick evaluation locally (after training):

```bat
python src/evaluate.py --pred predictions.csv --truth data/test_labels.csv
```

### Model storage & reproducibility
- Save both the pipeline components (vectorizer + model) and a `metadata.json` with training data hash, train/test split seed, scikit-learn version, Python version, and hyperparameters.
- Use joblib.dump/load for model files.

### Small experiments & CI
- For quick CI checks, use a tiny sample dataset (100–500 rows) and assert baseline metrics (e.g., test F1 > 0.80 on the tiny held-out set) to detect regressions.
- Add a smoke test that runs `python src/train.py --data data/sample_small.csv --out-dir /tmp/models --cv 2` and ensures model files are created.

### Edge cases and notes
- Imbalanced data: prefer class_weight='balanced' or use resampling methods.
- Non-English or multilingual text: consider language-specific preprocessing or multilingual embeddings.
- Privacy: remove PII when sharing datasets; store only hashes if required.

### Next steps I can take (pick one)
- Scaffold the minimal `src/` Python scripts (`train.py`, `infer.py`, `evaluate.py`), `requirements.txt`, and a small sample dataset so you can run the pipeline locally.
- Or just keep this as documentation and you implement code locally. If you want code, tell me which Python version and I will add runnable files and a brief test.

## Project details, conventions, and repo layout

Below are concrete conventions and paths I suggest we follow so the repository stays consistent and the ML/email work integrates cleanly with the rest of the AIoT project.

- Repository root conventions
	- `data/` — Raw and processed datasets
		- `data/raw/` — immutable original datasets (CSV/Parquet)
		- `data/processed/` — cleaned, tokenized or feature-engineered data used for training
	- `models/` — model artifacts and metadata (not for source-controlled large binaries; use `.gitignore` and store small artifacts or hashes)
	- `src/` — application and ML source code (training, inference, utilities)
	- `tools/` — developer utilities such as device simulator, data collectors, conversion scripts
	- `notebooks/` — exploratory notebooks (keep experiments here; do not rely on them for production workflows)
	- `openspec/` — specs and changes (this folder)
	- `tests/` — unit and integration tests
	- `requirements.txt` or `pyproject.toml` — dependency manifest for Python code

- Data conventions
	- Expect text fields to be UTF-8 and normalized (NFC). Use ISO 8601 timestamps in UTC where applicable.
	- Label column name: `label` using `spam`/`ham` strings or `1`/`0` integers. Document any mapping in `data/README.md`.
	- Small sample dataset for CI: `data/sample_small.csv` (100–500 rows) with balanced labels.

- Model artifacts and metadata
	- Save models under `models/<model-name>/` containing:
		- `model.joblib` — trained model or pipeline
		- `vectorizer.joblib` — TF-IDF or other vectorizer if not part of the pipeline
		- `metadata.json` — {training_date, data_hash, sklearn_version, python_version, seed, hyperparameters}
	- Add `models/.gitkeep` but do NOT commit large model files. Instead, keep them in CI artifacts or a lightweight `models/README.md` pointing to where full models are stored.

- Development commands (cmd.exe examples)
	- Create virtualenv and install

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

	- Run training (parameters shown previously)

```bat
python src/train.py --data data/emails.csv --out-dir models/logreg --cv 5 --seed 42
```

	- Run inference on a CSV

```bat
python src/infer.py --model models/logreg/model.joblib --vectorizer models/logreg/vectorizer.joblib --input data/unlabeled.csv --output predictions.csv
```

- Coding standards
	- Python: black + isort + flake8 (configure via `pyproject.toml` or `.flake8`); target minimal typing using typing module (optional but recommended)
	- Use functions and small modules with clear docstrings; prefer small, testable functions over monolith scripts

- Testing & CI
	- Unit tests: pytest under `tests/unit/`
	- Integration tests: `tests/integration/` which may start local dockerized services (Mosquitto) or use the `data/sample_small.csv` for ML tests
	- Add a GitHub Actions workflow (or similar CI) with jobs: lint, unit-test, ml-smoke-test (runs training on sample dataset and uploads artifacts)

- Git & PR conventions
	- Branches: `main` (protected), `feat/<short-desc>`, `fix/<short-desc>`, `docs/<short-desc>`
	- Commits: Conventional Commits style (e.g., `feat: add train script`) and small PRs
	- Require at least one reviewer and passing CI checks before merge

- Ownership and contacts
	- Add a top-level `OWNERS` or `MAINTAINERS.md` with names/emails of the people responsible for reviews and merges. If you want, I can scaffold this file.

If you'd like, I can now scaffold the minimal `src/` scripts and `requirements.txt` and run a quick smoke test locally in the workspace (small dataset approach). Tell me if you want Python 3.8/3.9/3.10/3.11 (default: 3.9) and I will proceed.
