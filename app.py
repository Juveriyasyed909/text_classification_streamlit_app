import json
from pathlib import Path
import subprocess
import pandas as pd
import streamlit as st
import joblib

from src.preprocessing import preprocess_text
from src.modeling import confidence_or_score

st.set_page_config(page_title="Text Classification Pipeline", layout="wide")

st.title("Applied AI Text Classification Pipeline")
st.caption("Replayable local ML pipeline with validation, preprocessing, training, evaluation, winner selection, artifacts, CLI, and inference.")

DATA_DIR = Path("data")
REPORT_DIR = Path("artifacts/reports")
MODEL_DIR = Path("artifacts/models")

with st.sidebar:
    st.header("Actions")
    run_pipeline = st.button("Run Full Pipeline")
    run_validation = st.button("Run validate.py")

if run_pipeline:
    with st.spinner("Running pipeline..."):
        result = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True)
    st.subheader("Pipeline Output")
    st.code(result.stdout + result.stderr)
    if result.returncode == 0:
        st.success("Pipeline completed successfully.")
    else:
        st.error("Pipeline failed.")

if run_validation:
    with st.spinner("Running validation..."):
        result = subprocess.run(["python", "validate.py"], capture_output=True, text=True)
    st.subheader("Validation Output")
    st.code(result.stdout + result.stderr)
    if result.returncode == 0:
        st.success("Validation passed.")
    else:
        st.error("Validation failed.")

st.header("Dataset Preview")
col1, col2 = st.columns(2)
with col1:
    if (DATA_DIR / "train.csv").exists():
        st.subheader("train.csv")
        st.dataframe(pd.read_csv(DATA_DIR / "train.csv"), use_container_width=True)
with col2:
    if (DATA_DIR / "test.csv").exists():
        st.subheader("test.csv")
        st.dataframe(pd.read_csv(DATA_DIR / "test.csv"), use_container_width=True)

st.header("Single Text Inference")
text = st.text_area("Enter text", "The app is easy to use and very helpful")
if st.button("Predict"):
    try:
        vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
        model = joblib.load(MODEL_DIR / "winning_model.joblib")
        processed = preprocess_text(text)
        X = vectorizer.transform([processed])
        pred = model.predict(X)[0]
        score = confidence_or_score(model, X)[0]
        st.success(f"Predicted label: {pred}")
        st.write({
            "processed_text": processed,
            "confidence_or_score": None if score is None else float(score)
        })
    except Exception as exc:
        st.warning("Run the pipeline first before inference.")
        st.exception(exc)

st.header("Reports")
report_files = [
    "data_validation_report.json",
    "preprocessing_preview.json",
    "split_report.json",
    "metrics.json",
    "model_selection_report.json",
    "error_analysis.json",
    "safeguards_report.json",
    "run_manifest.json",
]

for file_name in report_files:
    path = REPORT_DIR / file_name
    if path.exists():
        with st.expander(file_name):
            st.json(json.load(open(path, "r", encoding="utf-8")))

pred_path = REPORT_DIR / "test_predictions.csv"
if pred_path.exists():
    st.subheader("test_predictions.csv")
    st.dataframe(pd.read_csv(pred_path), use_container_width=True)
