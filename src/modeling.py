import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


def build_vectorizer(config):
    vectorizer_config = config.get("vectorizer", {})
    ngram_range = tuple(vectorizer_config.get("ngram_range", [1, 1]))
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=vectorizer_config.get("max_features", 5000),
        min_df=vectorizer_config.get("min_df", 1),
    )


def build_model(name, random_seed):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_seed)
    if name == "linear_svm":
        return LinearSVC(random_state=random_seed)
    if name == "naive_bayes":
        return MultinomialNB()
    raise ValueError(f"Unsupported model in config: {name}")


def confidence_or_score(model, X):
    if hasattr(model, "predict_proba"):
        return np.max(model.predict_proba(X), axis=1)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if len(scores.shape) == 1:
            return np.abs(scores)
        return np.max(scores, axis=1)
    return [None] * X.shape[0]
