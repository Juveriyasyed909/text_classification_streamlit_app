from src.utils import save_json

REQUIRED_TRAIN_COLUMNS = ["id", "text", "label"]
REQUIRED_TEST_COLUMNS = ["id", "text"]


def validate_data(train_df, test_df, report_path="artifacts/reports/data_validation_report.json"):
    errors = []
    warnings = []

    for col in REQUIRED_TRAIN_COLUMNS:
        if col not in train_df.columns:
            errors.append(f"train.csv missing required column: {col}")

    for col in REQUIRED_TEST_COLUMNS:
        if col not in test_df.columns:
            errors.append(f"test.csv missing required column: {col}")

    if not errors:
        if train_df["label"].nunique() < 2:
            errors.append("train.csv must contain at least 2 distinct labels")

        if train_df["id"].duplicated().any():
            errors.append("train.csv contains duplicate IDs")

        if test_df["id"].duplicated().any():
            errors.append("test.csv contains duplicate IDs")

        if train_df["text"].astype(str).str.strip().eq("").any():
            errors.append("train.csv contains empty text values after stripping whitespace")

        if test_df["text"].astype(str).str.strip().eq("").any():
            errors.append("test.csv contains empty text values after stripping whitespace")

    report = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
    }
    save_json(report_path, report)

    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))

    return report
