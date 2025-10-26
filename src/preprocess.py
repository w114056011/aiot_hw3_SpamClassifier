import argparse
import pandas as pd
import re
from pathlib import Path


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/sms_spam_no_header.csv")
    p.add_argument("--output", default="data/processed/sms_spam_clean.csv")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Try to read CSV with headers; if expected columns are missing,
    # attempt to read a headerless CSV in the common (label, text) order
    df = pd.read_csv(inp)
    if "text" not in df.columns or "label" not in df.columns:
        # Attempt headerless parse: many SMS datasets are `label,text` without headers
        df = pd.read_csv(inp, header=None, names=["label", "text"])
        if "text" not in df.columns or "label" not in df.columns:
            raise SystemExit("Input CSV must contain 'text' and 'label' columns (or be a two-column label,text CSV)")

    df["text_clean"] = df["text"].astype(str).map(clean_text)
    # keep id if present
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    df_out = df[["id", "text_clean", "label"]]
    df_out.to_csv(out, index=False)
    print(f"Wrote processed data to {out}")


if __name__ == "__main__":
    main()
