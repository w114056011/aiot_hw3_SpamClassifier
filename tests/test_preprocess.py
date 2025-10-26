import subprocess
import sys
from pathlib import Path


def test_preprocess_creates_processed(tmp_path):
    # Use project sample file
    proj_root = Path(__file__).resolve().parents[1]
    input_csv = proj_root / "data" / "sample_small.csv"
    out_csv = tmp_path / "sms_spam_clean.csv"

    cmd = [sys.executable, str(proj_root / "src" / "preprocess.py"),
           "--input", str(input_csv), "--output", str(out_csv)]
    subprocess.run(cmd, check=True)
    assert out_csv.exists()
    df = out_csv.read_text(encoding="utf-8")
    assert "text_clean" in df
