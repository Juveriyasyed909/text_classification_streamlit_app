import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from src.utils import ensure_dirs, save_json, load_json, log_stage, STAGES, timestamp
from src.preprocessing import preprocess_text
from src.data_validation import validate_data
from src.modeling import build_vectorizer, build_model, confidence_or_score
from src.selection import select_winner
from src.safeguards import run_safeguards

DATA_DIR = Path("data")
REPORT_DIR = Path("artifacts/reports")
MODEL_DIR = Path("artifacts/models")


def main():
    ensure_dirs()
    log_stage("INIT")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    config = load_json(DATA_DIR / "config.json")
    log_stage("DATA_LOADED")

    validate_data(train_df, test_df)
    log_stage("DATA_VALIDATED")

    train_df["processed_text"] = train_df["text"].apply(preprocess_text)
    test_df["processed_text"] = test_df["text"].apply(preprocess_text)
    preview = train_df[["id", "text", "processed_text"]].head(10).to_dict(orient="records")
    save_json(REPORT_DIR / "preprocessing_preview.json", preview)
    log_stage("TEXT_PREPROCESSED")

    seed = config.get("random_seed", 42)
    validation_split = config.get("validation_split", 0.2)
    stratify = train_df["label"] if train_df["label"].value_counts().min() >= 2 else None

    train_split, val_split = train_test_split(
        train_df,
        test_size=validation_split,
        random_state=seed,
        stratify=stratify,
    )

    split_report = {
        "random_seed": seed,
        "train_size": int(len(train_split)),
        "validation_size": int(len(val_split)),
        "labels": {}
    }
    labels = sorted(train_df["label"].unique())
    for label in labels:
        split_report["labels"][str(label)] = {
            "train": int((train_split["label"] == label).sum()),
            "validation": int((val_split["label"] == label).sum())
        }
    save_json(REPORT_DIR / "split_report.json", split_report)
    log_stage("SPLIT_CREATED")

    safeguards_report = run_safeguards(train_split, val_split)

    vectorizer = build_vectorizer(config)
    X_train = vectorizer.fit_transform(train_split["processed_text"])
    X_val = vectorizer.transform(val_split["processed_text"])
    X_test = vectorizer.transform(test_df["processed_text"])
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    log_stage("FEATURES_FIT")

    metrics = {}
    trained_models = {}
    model_names = config.get("models", ["logistic_regression", "linear_svm", "naive_bayes"])

    cv_report = {}
    use_cv = config.get("cross_validation", {}).get("enabled", False)
    cv_folds = config.get("cross_validation", {}).get("folds", 3)

    for model_name in model_names:
        model = build_model(model_name, seed)
        model.fit(X_train, train_split["label"])
        trained_models[model_name] = model
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")

        if use_cv:
            pipe = Pipeline([
                ("vectorizer", build_vectorizer(config)),
                ("model", build_model(model_name, seed))
            ])
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            scores = cross_val_score(pipe, train_df["processed_text"], train_df["label"], cv=cv, scoring="f1_macro")
            cv_report[model_name] = {
                "macro_f1_mean": float(scores.mean()),
                "macro_f1_std": float(scores.std()),
                "folds": cv_folds
            }

    if use_cv:
        save_json(REPORT_DIR / "cross_validation_report.json", cv_report)

    log_stage("MODELS_TRAINED")

    for model_name, model in trained_models.items():
        preds = model.predict(X_val)
        metrics[model_name] = {
            "accuracy": float(accuracy_score(val_split["label"], preds)),
            "macro_precision": float(precision_score(val_split["label"], preds, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(val_split["label"], preds, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(val_split["label"], preds, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(val_split["label"], preds, labels=labels).tolist(),
            "labels": [str(x) for x in labels],
            "classification_report": classification_report(val_split["label"], preds, zero_division=0, output_dict=True)
        }
    save_json(REPORT_DIR / "metrics.json", metrics)
    log_stage("MODELS_EVALUATED")

    winner, ranking = select_winner(metrics, config.get("selection_metric", "macro_f1"))
    winner_name = winner["model_name"]
    selection_report = {
        "selection_metric": config.get("selection_metric", "macro_f1"),
        "winner": winner_name,
        "ranking": ranking,
        "reason": f"{winner_name} selected using deterministic sorting: highest selection metric, then macro precision, then alphabetical model name."
    }
    save_json(REPORT_DIR / "model_selection_report.json", selection_report)
    joblib.dump(trained_models[winner_name], MODEL_DIR / "winning_model.joblib")
    save_json(MODEL_DIR / "winning_model_metadata.json", {"winner": winner_name})
    log_stage("WINNER_SELECTED")

    log_stage("ARTIFACTS_SAVED")

    winner_model = trained_models[winner_name]
    val_preds = winner_model.predict(X_val)
    val_scores = confidence_or_score(winner_model, X_val)
    errors = []
    top_k = config.get("top_k_error_examples", 10)
    for row, pred, score in zip(val_split.to_dict(orient="records"), val_preds, val_scores):
        if row["label"] != pred:
            errors.append({
                "id": row["id"],
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": pred,
                "confidence_or_score": None if score is None else float(score),
                "reason": "Misclassified validation example useful for inspecting ambiguous wording, weak features, or class overlap."
            })
    save_json(REPORT_DIR / "error_analysis.json", errors[:top_k])

    test_preds = winner_model.predict(X_test)
    pred_df = pd.DataFrame({"id": test_df["id"], "predicted_label": test_preds})
    pred_df.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    log_stage("TEST_PREDICTIONS_GENERATED")

    run_manifest = {
        "timestamp": timestamp(),
        "random_seed": seed,
        "files_read": ["data/train.csv", "data/test.csv", "data/config.json"],
        "models_trained": model_names,
        "winning_model": winner_name,
        "key_metrics": metrics[winner_name],
        "output_artifact_paths": {
            "vectorizer": "artifacts/models/vectorizer.joblib",
            "winning_model": "artifacts/models/winning_model.joblib",
            "metrics": "artifacts/reports/metrics.json",
            "test_predictions": "artifacts/reports/test_predictions.csv",
            "safeguards": "artifacts/reports/safeguards_report.json"
        },
        "pipeline_stages": STAGES
    }
    save_json(REPORT_DIR / "run_manifest.json", run_manifest)
    log_stage("REPORT_EXPORTED")

    print("\nPipeline completed successfully.")
    print(f"Winning model: {winner_name}")
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
