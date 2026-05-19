# Applied AI Text Classification Pipeline

This project implements a replayable local experimentation pipeline for text classification using Python, scikit-learn, and Streamlit.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run full pipeline

```bash
python pipeline.py
```

## Run validation

```bash
python validate.py
```

## Run CLI inference

```bash
python predict.py --text "The app is easy to use"
```

## Run Streamlit app

```bash
streamlit run app.py
```

## Main artifacts

- artifacts/reports/data_validation_report.json
- artifacts/reports/preprocessing_preview.json
- artifacts/reports/split_report.json
- artifacts/reports/metrics.json
- artifacts/reports/model_selection_report.json
- artifacts/reports/error_analysis.json
- artifacts/reports/test_predictions.csv
- artifacts/models/vectorizer.joblib
- artifacts/models/winning_model.joblib
