import argparse
import joblib
from src.preprocessing import preprocess_text
from src.modeling import confidence_or_score


def main():
    parser = argparse.ArgumentParser(description="Single text inference CLI")
    parser.add_argument("--text", required=True, help="Input text to classify")
    args = parser.parse_args()

    vectorizer = joblib.load("artifacts/models/vectorizer.joblib")
    model = joblib.load("artifacts/models/winning_model.joblib")

    processed = preprocess_text(args.text)
    X = vectorizer.transform([processed])
    pred = model.predict(X)[0]
    score = confidence_or_score(model, X)[0]

    print({
        "text": args.text,
        "processed_text": processed,
        "predicted_label": str(pred),
        "confidence_or_score": None if score is None else float(score)
    })


if __name__ == "__main__":
    main()
