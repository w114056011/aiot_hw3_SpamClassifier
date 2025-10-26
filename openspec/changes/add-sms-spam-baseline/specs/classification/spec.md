## ADDED Requirements

### Requirement: Data pre-processing
The system SHALL provide a reproducible data pre-processing step that ingests `data/sms_spam_no_header.csv`, normalizes and cleans text, tokenizes, and writes the processed output to `data/processed/sms_spam_clean.csv` ready for training.

#### Scenario: Clean and vectorize
- **WHEN** the pre-processing script is run on `data/sms_spam_no_header.csv`
- **THEN** `data/processed/sms_spam_clean.csv` SHALL exist
- **AND** each row SHALL contain `id`, `text_clean`, and `label` columns

### Requirement: Model training
The system SHALL provide a training script that builds a classifier pipeline using TF-IDF vectorization and Logistic Regression (SVM optional), supports cross-validation and a small hyperparameter grid, and writes model artifacts to `models/<model-name>/`.

#### Scenario: Train baseline model
- **WHEN** `src/train.py` is executed with `--data data/processed/sms_spam_clean.csv` and `--out-dir models/logreg`
- **THEN** `models/logreg/model.joblib` and `models/logreg/vectorizer.joblib` SHALL be produced
- **AND** `models/logreg/report.json` SHALL contain evaluation metrics on the test split (precision, recall, f1, accuracy, roc_auc)

### Requirement: Evaluation and reporting
The system SHALL produce evaluation artifacts: confusion matrix, ROC curve and PR curve images, and a JSON/CSV report with metrics and cross-validation statistics.

#### Scenario: Produce evaluation artifacts
- **WHEN** `src/evaluate.py` is run against the trained model and test set
- **THEN** `models/logreg/reports/confusion_matrix.png`, `roc.png`, `pr.png` and `models/logreg/report.json` SHALL be created with numeric metrics
