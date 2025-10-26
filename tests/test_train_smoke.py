import subprocess
import sys
from pathlib import Path


def test_train_smoke_creates_model(tmp_path):
    proj_root = Path(__file__).resolve().parents[1]
    # Preprocess sample into tmp dir first
    processed = tmp_path / "processed.csv"
    cmd_pre = [sys.executable, str(proj_root / "src" / "preprocess.py"),
              "--input", str(proj_root / "data" / "sample_small.csv"), "--output", str(processed)]
    subprocess.run(cmd_pre, check=True)

    outdir = tmp_path / "models"
    cmd_train = [sys.executable, str(proj_root / "src" / "train.py"),
                 "--data", str(processed), "--out-dir", str(outdir), "--cv", "2", "--seed", "42"]
    subprocess.run(cmd_train, check=True)

    model_file = outdir / "model.joblib"
    assert model_file.exists()
