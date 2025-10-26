import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model folder (e.g. models/logreg)")
    p.add_argument("--out", required=True, help="Output directory for the report")
    args = p.parse_args()

    model_dir = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_file = model_dir / "report.json"
    if not report_file.exists():
        raise SystemExit(f"Report file not found: {report_file}")

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    # copy JSON to out
    shutil.copy(report_file, out_dir / "eval_report.json")

    # copy any images from model reports folder
    model_reports = model_dir / "reports"
    if model_reports.exists() and model_reports.is_dir():
        for img in model_reports.glob("*.png"):
            shutil.copy(img, out_dir / img.name)

    # create a lightweight HTML summary
    html = ["<html><head><meta charset='utf-8'><title>Model Report</title></head><body>"]
    html.append(f"<h1>Model report: {model_dir.name}</h1>")
    html.append("<h2>Metrics</h2>")
    html.append("<ul>")
    for k, v in (report or {}).items():
        # if it's a nested dict (classification_report), skip verbose
        if isinstance(v, dict):
            html.append(f"<li>{k}: (see JSON)</li>")
        else:
            html.append(f"<li>{k}: {v}</li>")
    html.append("</ul>")

    # embed images if present
    for img_name in ("confusion_matrix.png", "roc.png", "pr.png"):
        img_path = out_dir / img_name
        if img_path.exists():
            html.append(f"<h3>{img_name}</h3>")
            html.append(f"<img src=\"{img_name}\" style=\"max-width:800px;\"/>")

    html.append("</body></html>")

    with open(out_dir / "report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Wrote report to {out_dir / 'report.html'} and copied images/json")


if __name__ == "__main__":
    main()
