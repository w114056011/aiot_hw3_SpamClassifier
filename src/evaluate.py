import argparse
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--out-dir", default="models/logreg/reports")
    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfp = pd.read_csv(args.pred)
    dft = pd.read_csv(args.truth)

    # Determine join key: prefer 'id', then 'text_clean', then 'text', else align by index if lengths match
    if "id" in dfp.columns and "id" in dft.columns:
        merged = dfp.merge(dft, on="id")
    elif "text_clean" in dfp.columns and "text_clean" in dft.columns:
        merged = dfp.merge(dft, on="text_clean")
    elif "text" in dfp.columns and "text" in dft.columns:
        merged = dfp.merge(dft, on="text")
    else:
        # fallback: if lengths equal, align by row order
        if len(dfp) == len(dft):
            merged = pd.concat([dfp.reset_index(drop=True), dft.reset_index(drop=True)], axis=1)
        else:
            raise SystemExit(
                "Cannot determine join key for predictions and truth. Ensure both files share 'id' or 'text_clean' or have same order/length."
            )

    # Find truth label column
    label_col = None
    for c in ("label", "Label", "truth", "y", "target", "class"):
        if c in merged.columns:
            label_col = c
            break
    if label_col is None:
        # try to heuristically find a short categorical/text column
        for c in merged.columns:
            if merged[c].dtype == object and merged[c].nunique() <= 3:
                label_col = c
                break
    if label_col is None:
        raise SystemExit("Could not find a label column in the truth data. Expected 'label' or similar.")

    y_true = merged[label_col].map(lambda s: 1 if str(s).strip().lower() in ("spam", "1", "true") else 0)

    # Prediction columns: label and probability (optional)
    if "pred_label" in merged.columns:
        y_pred = merged["pred_label"]
    elif "pred" in merged.columns:
        y_pred = merged["pred"]
    elif "prediction" in merged.columns:
        y_pred = merged["prediction"]
    else:
        raise SystemExit("Predictions file must contain a predicted label column (pred_label/pred/prediction)")

    if "pred_proba" in merged.columns:
        y_proba = merged["pred_proba"]
    elif "pred_probability" in merged.columns:
        y_proba = merged["pred_probability"]
    elif "proba" in merged.columns:
        y_proba = merged["proba"]
    else:
        y_proba = None

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    roc = None
    try:
        roc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    report = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm,
    }

    with open(Path(args.out_dir) / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Wrote evaluation report to", Path(args.out_dir) / "eval_report.json")

    # Create plots: ROC, Precision-Recall, Confusion Matrix
    try:
        # Confusion matrix
        cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["ham", "spam"])
        fig_cm = cm_display.figure_
        fig_cm.savefig(Path(args.out_dir) / "confusion_matrix.png")
        plt.close(fig_cm)
    except Exception:
        pass

    if y_proba is not None:
        try:
            roc_disp = RocCurveDisplay.from_predictions(y_true, y_proba)
            fig_roc = roc_disp.figure_
            fig_roc.savefig(Path(args.out_dir) / "roc.png")
            plt.close(fig_roc)
        except Exception:
            pass

        try:
            pr_disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba)
            fig_pr = pr_disp.figure_
            fig_pr.savefig(Path(args.out_dir) / "pr.png")
            plt.close(fig_pr)
        except Exception:
            pass


if __name__ == "__main__":
    main()
