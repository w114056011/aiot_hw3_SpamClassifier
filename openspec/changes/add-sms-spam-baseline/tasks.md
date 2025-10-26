## 1. Implementation
- [ ] 1.1 Add `src/preprocess.py` to read `data/sms_spam_no_header.csv`, clean text, and write `data/processed/sms_spam_clean.csv`.
- [ ] 1.2 Add `src/train.py` that trains a scikit-learn Pipeline (TfidfVectorizer -> LogisticRegression/SVC), performs GridSearchCV (small grid), and writes model artifacts to `models/logreg/`.
- [ ] 1.3 Add `src/evaluate.py` to generate metrics report (`models/logreg/report.json`) and charts (`models/logreg/reports/roc.png`, `pr.png`, `confusion_matrix.png`).
- [ ] 1.4 Add `src/infer.py` for single-file and bulk inference.
- [ ] 1.5 Add `data/sample_small.csv` (100–500 rows) and `data/README.md` describing format and label mapping.

## 2. Tests
- [ ] 2.1 Unit tests for preprocessing, small pipeline unit test for training (pytest)
- [ ] 2.2 Integration smoke test: run training on `data/sample_small.csv` and assert artifacts created and F1 greater than a configurable threshold.

## 3. Docs
- [ ] 3.1 Update `openspec/project.md` (already updated) and add a `ml/README.md` with usage examples and commands.

## 4. CI
- [ ] 4.1 Add GitHub Actions workflow `ci-ml-smoke.yml` with job that runs the smoke test on a small dataset.
