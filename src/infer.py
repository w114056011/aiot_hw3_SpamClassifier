import argparse
from pathlib import Path
import joblib
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=False)
    args = p.parse_args()

    model = joblib.load(args.model)

    inp = Path(args.input)
    if inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)
        text_col = "text_clean" if "text_clean" in df.columns else "text"
        df["pred_proba"] = model.predict_proba(df[text_col].astype(str))[:, 1]
        df["pred_label"] = model.predict(df[text_col].astype(str))
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Wrote predictions to {args.output}")
        else:
            print(df[[text_col, "pred_label", "pred_proba"]].head())
    else:
        # assume single text file
        text = inp.read_text(encoding="utf-8")
        proba = model.predict_proba([text])[0][1]
        label = model.predict([text])[0]
        print({"pred_label": int(label), "pred_proba": float(proba)})


if __name__ == "__main__":
    main()
