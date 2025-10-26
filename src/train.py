import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/sms_spam_clean.csv")
    p.add_argument("--out-dir", default="models/logreg")
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    X = df["text_clean"].astype(str)
    y = df["label"].map(lambda s: 1 if str(s).strip().lower() in ("spam","1","true") else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000)),
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1, 10],
    }

    gs = GridSearchCV(pipe, param_grid, cv=args.cv, scoring="f1", n_jobs=1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    y_pred = best.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print("Test F1:", f1)

    report = classification_report(y_test, y_pred, output_dict=True)
    import json

    with open(outdir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # save model and vectorizer if pipeline
    joblib.dump(best, outdir / "model.joblib")
    print(f"Saved model to {outdir / 'model.joblib'}")


if __name__ == "__main__":
    main()
