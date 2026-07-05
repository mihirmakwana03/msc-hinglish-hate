"""Statistical baseline: TF-IDF + Logistic Regression.

This is the safety-net result. It is intentionally simple and defensible:
word 1-2 grams, class_weight balanced to handle the hate/not imbalance, and
macro-F1 as the headline metric so both classes count equally.

Two evaluation modes:
  evaluate_within  - stratified split of one corpus (the safety-net number)
  evaluate_cross   - train on one corpus, test on another (the generalisation
                     drop that the dissertation is actually about)
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import clean_text

RANDOM_STATE = 42


def build_pipeline(
    ngram_range=(1, 2),
    min_df=2,
    C=1.0,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,          # clean_text already lowercases
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=C,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "hate_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(
            y_true, y_pred, target_names=["not", "hate"], zero_division=0, digits=3
        ),
    }


def evaluate_within(df: pd.DataFrame, test_size=0.2, **pipe_kwargs) -> dict:
    """Stratified train/test split within one corpus -> the safety-net F1."""
    X, y = df["text"].tolist(), df["label"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    pipe = build_pipeline(**pipe_kwargs)
    pipe.fit(X_tr, y_tr)
    out = _metrics(y_te, pipe.predict(X_te))
    out["n_train"], out["n_test"] = len(X_tr), len(X_te)
    out["pipeline"] = pipe
    return out


def evaluate_cross(train_df: pd.DataFrame, test_df: pd.DataFrame, **pipe_kwargs) -> dict:
    """Train on one corpus, evaluate on another -> generalisation number."""
    pipe = build_pipeline(**pipe_kwargs)
    pipe.fit(train_df["text"].tolist(), train_df["label"].tolist())
    out = _metrics(test_df["label"].tolist(), pipe.predict(test_df["text"].tolist()))
    out["n_train"], out["n_test"] = len(train_df), len(test_df)
    out["pipeline"] = pipe
    return out


def summarise(name: str, res: dict) -> str:
    return (
        f"[{name}]  macro-F1 = {res['macro_f1']:.3f}   "
        f"hate-F1 = {res['hate_f1']:.3f}   acc = {res['accuracy']:.3f}   "
        f"(train {res['n_train']}, test {res['n_test']})"
    )
