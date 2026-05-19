from src.utils import save_json


def run_safeguards(train_split_df, val_split_df, report_path="artifacts/reports/safeguards_report.json"):
    findings = []

    label_ratio = train_split_df["label"].value_counts(normalize=True).to_dict()
    if any(v > 0.8 for v in label_ratio.values()):
        findings.append({"type": "class_imbalance", "message": "One class exceeds 80% of training split."})

    missing_val_classes = set(train_split_df["label"].unique()) - set(val_split_df["label"].unique())
    if missing_val_classes:
        findings.append({"type": "missing_validation_class", "classes": sorted(list(missing_val_classes))})

    duplicates = set(train_split_df["processed_text"]).intersection(set(val_split_df["processed_text"]))
    if duplicates:
        findings.append({"type": "duplicate_text_leakage", "count": len(duplicates)})

    report = {"findings": findings, "status": "warnings_found" if findings else "passed"}
    save_json(report_path, report)
    return report
