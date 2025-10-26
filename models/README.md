Model artifact conventions

- Store model artifacts under `models/<model-name>/` (e.g., `models/logreg/`).
- Expected files:
  - `model.joblib` — the trained pipeline or model
  - `report.json` — evaluation metrics (classification report)
  - `reports/` — optional charts: `roc.png`, `pr.png`, `confusion_matrix.png`
  - `metadata.json` — training metadata (seed, data hash, package versions)

Do NOT commit large model binaries to Git; keep small artifacts for examples only. Use CI artifacts or external storage for large models.
