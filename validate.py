import json
import subprocess
from pathlib import Path
import pandas as pd

REQUIRED_JSON = [
    "artifacts/reports/data_validation_report.json",
    "artifacts/reports/preprocessing_preview.json",
    "artifacts/reports/split_report.json",
    "artifacts/reports/metrics.json",
    "artifacts/reports/model_selection_report.json",
    "artifacts/reports/error_analysis.json",
    "artifacts/reports/run_manifest.json",
    "artifacts/reports/safeguards_report.json",
]

REQUIRED_FILES = REQUIRED_JSON + [
    "artifacts/reports/test_predictions.csv",
    "artifacts/models/vectorizer.joblib",
    "artifacts/models/winning_model.joblib",
]


def main():
    failures = []

    for path in REQUIRED_FILES:
        if not Path(path).exists():
            failures.append(f"Missing artifact: {path}")

    for path in REQUIRED_JSON:
        if Path(path).exists():
            try:
                json.load(open(path, "r", encoding="utf-8"))
            except Exception as exc:
                failures.append(f"Invalid JSON: {path}: {exc}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    for col in ["id", "text", "label"]:
        if col not in train.columns:
            failures.append(f"train.csv missing column {col}")
    for col in ["id", "text"]:
        if col not in test.columns:
            failures.append(f"test.csv missing column {col}")

    if Path("artifacts/reports/metrics.json").exists():
        metrics = json.load(open("artifacts/reports/metrics.json", "r", encoding="utf-8"))
        if len(metrics) < 3:
            failures.append("At least 3 models must be trained and evaluated")

    if Path("artifacts/reports/test_predictions.csv").exists():
        preds = pd.read_csv("artifacts/reports/test_predictions.csv")
        if len(preds) != len(test):
            failures.append("Predictions were not generated for all test rows")

    try:
        result = subprocess.run(
            ["python", "predict.py", "--text", "The app is easy to use"],
            capture_output=True,
            text=True,
            check=True,
        )
        if "predicted_label" not in result.stdout:
            failures.append("CLI did not return predicted_label")
    except Exception as exc:
        failures.append(f"CLI inference failed: {exc}")

    if failures:
        print("VALIDATION FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
