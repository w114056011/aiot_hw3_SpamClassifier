## 1. Implementation
- [ ] 1.1 Create `tools/streamlit_app.py` that:
  - Loads a model pipeline and vectorizer from `models/<model-name>/model.joblib` (or pipeline object)
  - Shows evaluation summary (metrics), displays confusion matrix, ROC/PR plots, and allows single-text and CSV batch inference
  - Allows user to adjust probability threshold and see updated confusion matrix/precision/recall
- [ ] 1.2 Create `src/report.py` to generate static reports (PNG and HTML) from evaluation artifacts and model predictions
- [ ] 1.3 Add `requirements-streamlit.txt` (or include streamlit in `requirements.txt` behind an extras flag)

## 2. Tests
- [ ] 2.1 Unit tests for `src/report.py` generating PNGs from small synthetic inputs
- [ ] 2.2 Smoke test: start streamlit in headless mode and assert the server starts (optional in CI)

## 3. Docs
- [ ] 3.1 Add `docs/streamlit.md` with instructions for running locally and sample screenshots
- [ ] 3.2 Update `ml/README.md` with a `make report` and `make serve` examples

## 4. CI
- [ ] 4.1 Add CI job `ci-report.yml` that runs `python src/report.py --model models/logreg --out reports/` and uploads `reports/` as job artifacts
