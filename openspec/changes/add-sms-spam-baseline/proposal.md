## Why

Students and developers need a reproducible, minimal machine-learning baseline to classify short text messages (spam vs ham). Providing a documented baseline (Preprocess → Train → Predict) reduces onboarding friction, enables CI smoke-tests, and creates a clear artifact for grading and iterative improvement.

This change uses the dataset available at `data/sms_spam_no_header.csv` (if the project intends to classify emails instead of SMS, we will adapt the dataset reference — current proposal assumes the provided SMS dataset is canonical).

## What Changes

- Add a new change: `add-sms-spam-baseline`.
- Add scripts and specs for a baseline classification pipeline that performs:
  - Data pre-processing: cleaning, tokenization, TF-IDF vectorization and output to `data/processed/sms_spam_clean.csv`.
  - Model training: Logistic Regression (optionally SVM) with a small hyperparameter search and cross-validation.
  - Evaluation: compute metrics (precision, recall, F1, accuracy, ROC AUC), produce confusion matrix and plots (ROC/PR curves) and write `models/<name>/report.json` and chart PNGs to `models/<name>/reports/`.
- Add `tasks.md` with a clear implementation checklist and a spec delta under `changes/add-sms-spam-baseline/specs/classification/spec.md` describing requirements and at least one scenario per requirement.

**Breaking changes:** None — additive developer tooling only.

## Impact

- Affected specs: adds test/spec coverage for a `classification` capability under `changes/<id>/specs` (no existing specs are modified).
- Affected code: new scripts under `src/ml/` or `src/` (`preprocess.py`, `train.py`, `evaluate.py`, `infer.py`) and small docs (README and usage examples).
- Rollout: developer-only; no production services modified.

## Owner

- Proposed owner: repository maintainer or course staff. Please assign an owner for approval and review.

## Timeline

- Draft spec & tasks: same day
- Implementation + unit tests + small sample dataset: 1 day
- Validation and CI smoke test: +0.5 day

## Validation

- Run `openspec validate add-sms-spam-baseline --strict` once the spec delta files are in place. The spec delta included with this proposal follows the required formatting (## ADDED Requirements + `#### Scenario:` entries).
