# ML: SMS/Email Spam Baseline

This folder contains scripts and guidance for the baseline spam/ham classification pipeline.

Usage examples (from repository root):

```bat
python src/preprocess.py --input data/sms_spam_no_header.csv --output data/processed/sms_spam_clean.csv
python src/train.py --data data/processed/sms_spam_clean.csv --out-dir models/logreg --cv 3 --seed 42
python src/infer.py --model models/logreg/model.joblib --input data/processed/sms_spam_clean.csv --output models/logreg/predictions.csv
python src/evaluate.py --pred models/logreg/predictions.csv --truth data/processed/sms_spam_clean.csv --out-dir models/logreg/reports
```

Files created by scripts will be placed in `data/processed/` and `models/logreg/`.

The baseline uses TF-IDF + Logistic Regression. If you prefer SVM or a different pipeline, edit `src/train.py`.
