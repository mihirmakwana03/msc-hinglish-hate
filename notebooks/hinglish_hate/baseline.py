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


# --------------------------------------------------------------------------
# Richer models: full feature union (word + char + lexicon) and alternatives
# --------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

from .features import build_feature_union


def build_full_pipeline(lexicon_terms, clf="logreg", C=1.0, class_weight="balanced",
                        word_ngrams=(1, 2), char_ngrams=(3, 5), min_df=2) -> Pipeline:
    """Word + char TF-IDF + lexicon counts -> classifier.

    clf: 'logreg' (LogisticRegression) or 'svm' (LinearSVC).
    class_weight: 'balanced' or None (pass None to show what imbalance costs).
    """
    if clf == "logreg":
        model = LogisticRegression(max_iter=2000, C=C, class_weight=class_weight,
                                   random_state=RANDOM_STATE)
    elif clf == "svm":
        model = LinearSVC(C=C, class_weight=class_weight, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown clf: {clf!r}")
    feats = build_feature_union(lexicon_terms, word_ngrams=word_ngrams,
                                char_ngrams=char_ngrams, min_df=min_df)
    return Pipeline([("features", feats), ("clf", model)])


def full_metrics(y_true, y_pred) -> dict:
    """Macro-F1, per-class precision/recall/F1, accuracy, confusion matrix."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "hate_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_not": p[0], "recall_not": r[0], "f1_not": f[0],
        "precision_hate": p[1], "recall_hate": r[1], "f1_hate": f[1],
        "confusion": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=["not", "hate"],
                                        zero_division=0, digits=3),
    }


def evaluate_pipeline(pipe, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, score on test, return full metrics."""
    pipe.fit(X_train, y_train)
    out = full_metrics(y_test, pipe.predict(X_test))
    out["n_train"], out["n_test"] = len(X_train), len(y_test)
    out["pipeline"] = pipe
    return out


def cross_validate_macro_f1(pipe, X, y, n_splits: int = 5) -> dict:
    """Stratified k-fold CV on macro-F1 -> mean and std across folds.

    Hunter's instruction: report the within-dataset number as mean +/- std over
    N=5 folds rather than a single split, so the headline figure carries a
    measure of variance.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")
    return {"scores": scores, "mean": float(scores.mean()), "std": float(scores.std()),
            "n_splits": n_splits}


def stratified_kfold_evaluate(pipe_factory, X, y, n_splits: int = 5,
                              random_state: int = RANDOM_STATE, verbose: bool = True):
    """Stratified k-fold cross-validation with an explicit, inspectable fold loop.

    Written as a visible loop rather than a cross_val_score call so every step
    can be explained: shuffle, split into N folds, rotate which fold is held
    out, train on (N-1)/N, test on 1/N, record metrics, repeat.

    Stratified (rather than plain) k-fold keeps the hate/not ratio the same in
    every fold, which matters because the data is imbalanced. shuffle=True with
    a fixed random_state does the randomisation that page 1 describes: it
    removes any ordering in the file (e.g. all hate rows together, or all rows
    from one source together) before slicing, so no fold gets a biased chunk.

    pipe_factory: a zero-argument callable returning a FRESH unfitted pipeline.
                  A factory (not a single pipeline) guarantees each fold trains
                  a model from scratch with no leakage between folds.

    Returns (per_fold_dataframe, summary_dict) where summary has mean and std
    for accuracy, precision, recall and macro-F1.
    """
    import numpy as _np
    import pandas as _pd

    X = list(X)
    y = list(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_te = [y[i] for i in test_idx]

        pipe = pipe_factory()          # fresh model for this fold
        pipe.fit(X_tr, y_tr)           # train on (N-1)/N of the data
        y_pred = pipe.predict(X_te)    # test on the held-out 1/N

        p, r, f, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0)
        rows.append({
            "fold": fold,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "test_hate_rate": round(float(_np.mean(y_te)), 3),
            "accuracy": accuracy_score(y_te, y_pred),
            "precision_macro": p,
            "recall_macro": r,
            "macro_f1": f1_score(y_te, y_pred, average="macro"),
            "hate_f1": f1_score(y_te, y_pred, pos_label=1, zero_division=0),
        })
        if verbose:
            print(f"  fold {fold}/{n_splits}: macro-F1 = {rows[-1]['macro_f1']:.3f} "
                  f"(train {len(X_tr)}, test {len(X_te)})")

    per_fold = _pd.DataFrame(rows)
    metrics = ["accuracy", "precision_macro", "recall_macro", "macro_f1", "hate_f1"]
    summary = {m: {"mean": float(per_fold[m].mean()), "std": float(per_fold[m].std(ddof=0))}
               for m in metrics}
    summary["n_splits"] = n_splits
    return per_fold, summary


def format_cv_summary(summary: dict) -> str:
    """One line per metric: 'macro_f1        0.629 +/- 0.021'."""
    lines = [f"Stratified {summary['n_splits']}-fold cross-validation"]
    for m in ["accuracy", "precision_macro", "recall_macro", "macro_f1", "hate_f1"]:
        s = summary[m]
        lines.append(f"  {m:16s} {s['mean']:.3f} +/- {s['std']:.3f}")
    return "\n".join(lines)
