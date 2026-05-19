# Text Classification Experimentation Pipeline

A replayable local experimentation pipeline for text classification built using Python and scikit-learn. The project trains multiple baseline models, evaluates them consistently, selects a deployable winner using deterministic logic, generates reusable artifacts, and supports prediction through both CLI inference and an optional Streamlit interface.

This implementation focuses on Applied AI engineering principles including reproducibility, modular design, validation, model comparison, artifact generation, and separation of training from inference.

---

## Features

- Data loading and validation
- Deterministic text preprocessing
- Train/validation split with reproducibility
- TF-IDF feature extraction
- Multiple baseline models:
  - Logistic Regression
  - Linear SVM
  - Multinomial Naive Bayes
- Automated model evaluation
- Deterministic winner selection
- Error analysis
- Safeguard checks
- Cross-validation support
- Saved reusable model artifacts
- CLI inference support
- Validation script
- Optional Streamlit interface for visualization

---

## Pipeline Stages

The pipeline follows explicit stages:

```text
INIT
→ DATA_LOADED
→ DATA_VALIDATED
→ TEXT_PREPROCESSED
→ SPLIT_CREATED
→ FEATURES_FIT
→ MODELS_TRAINED
→ MODELS_EVALUATED
→ WINNER_SELECTED
→ ARTIFACTS_SAVED
→ TEST_PREDICTIONS_GENERATED
→ REPORT_EXPORTED
```

---

## Project Structure

```text
text_classification_streamlit_app/

src/
    data_loader.py
    validator.py
    preprocessing.py
    splitter.py
    trainer.py
    evaluator.py
    selector.py
    safeguards.py
    error_analysis.py

artifacts/

    models/
        vectorizer.joblib
        winning_model.joblib
        logistic_regression.joblib
        linear_svm.joblib
        naive_bayes.joblib

    reports/
        data_validation_report.json
        preprocessing_preview.json
        split_report.json
        metrics.json
        model_selection_report.json
        error_analysis.json
        safeguards_report.json
        run_manifest.json
        cross_validation_report.json

pipeline.py
predict.py
validate.py
app.py
requirements.txt
config.json
train.csv
test.csv
test_predictions.csv
README.md
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/Juveriyasyed909/text_classification_streamlit_app.git
```

Move into project:

```bash
cd text_classification_streamlit_app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Training Pipeline

```bash
python pipeline.py
```

Expected execution:

```text
STAGE: INIT
STAGE: DATA_LOADED
STAGE: DATA_VALIDATED
STAGE: TEXT_PREPROCESSED
STAGE: SPLIT_CREATED
STAGE: FEATURES_FIT
STAGE: MODELS_TRAINED
STAGE: MODELS_EVALUATED
STAGE: WINNER_SELECTED
STAGE: ARTIFACTS_SAVED
STAGE: TEST_PREDICTIONS_GENERATED
STAGE: REPORT_EXPORTED
```

---

## Run Validation

```bash
python validate.py
```

Expected:

```text
Validation PASSED
```

---

## CLI Inference

Run prediction on a single input:

```bash
python predict.py --text "The app is easy to use"
```

Example output:

```text
Prediction: positive
Confidence: 0.91
```

---

## Launch Streamlit Interface

A lightweight Streamlit interface is included for testing and visualization.

```bash
streamlit run app.py
```

---

## Model Selection Logic

Models are ranked using:

1. Highest Macro F1
2. Higher Macro Precision
3. Alphabetical order (tie breaker)

This ensures deterministic winner selection.

---

## Reports Generated

Generated JSON reports:

- data_validation_report.json
- preprocessing_preview.json
- split_report.json
- metrics.json
- model_selection_report.json
- error_analysis.json
- safeguards_report.json
- run_manifest.json
- cross_validation_report.json

Prediction output:

```text
test_predictions.csv
```

---

## Tech Stack

- Python
- scikit-learn
- pandas
- numpy
- joblib
- Streamlit
- argparse

---

## Notes

The assessment requirements did not require a web application. Streamlit was added as an optional visualization layer while preserving a pipeline-first architecture with CLI-based inference and reusable artifacts.
