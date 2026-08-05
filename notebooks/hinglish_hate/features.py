"""Feature construction for the statistical model (Day 10).

Combines three views of each post into one matrix:
  - word TF-IDF   (1-2 grams)      : lexical content
  - char TF-IDF   (3-5 char-grams) : robust to romanisation spelling variation
  - lexicon count : how many slur/abuse terms (from the curated lexicon) appear

The char n-grams and the lexicon are what make this more than a bag of words:
both help with the non-standard spelling that defines romanised Hinglish.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .preprocessing import clean_text

TOKEN = re.compile(r"[a-z]+")


def load_lexicon(csv_path: str | Path) -> list[str]:
    """Load kept terms + their variants from hinglish_slur_lexicon.csv.

    Returns a de-duplicated list of lowercased surface forms. Accepts either the
    curated CSV (with a `keep` column) or a plain one-term-per-line file.
    """
    p = Path(csv_path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "keep" in df.columns:
            df = df[df["keep"].astype(str).isin(["1", "1.0", "True", "true"])]
        forms = set()
        for _, r in df.iterrows():
            forms.add(str(r["term"]).strip().lower())
            if "variants" in df.columns and pd.notna(r.get("variants")) and str(r["variants"]).strip():
                forms.update(v.strip().lower() for v in str(r["variants"]).split("|") if v.strip())
    else:
        forms = {w.strip().lower() for w in open(p, encoding="utf-8") if w.strip()}
    forms.discard("")
    return sorted(forms)


class LexiconCounter(BaseEstimator, TransformerMixin):
    """Per-post lexicon features: raw match count and match ratio (per token)."""

    def __init__(self, terms):
        self.terms = set(terms)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = np.zeros((len(X), 2), dtype=float)
        for i, text in enumerate(X):
            toks = TOKEN.findall(clean_text(text))
            if not toks:
                continue
            hits = sum(1 for t in toks if t in self.terms)
            out[i, 0] = hits
            out[i, 1] = hits / len(toks)
        return out

    def get_feature_names_out(self, names=None):
        return np.array(["lex_count", "lex_ratio"])


def build_feature_union(lexicon_terms, word_ngrams=(1, 2), char_ngrams=(3, 5), min_df=2):
    """Word TF-IDF + char TF-IDF + scaled lexicon counts, as one FeatureUnion."""
    word = TfidfVectorizer(preprocessor=clean_text, lowercase=False,
                           ngram_range=word_ngrams, min_df=min_df, sublinear_tf=True)
    char = TfidfVectorizer(preprocessor=clean_text, lowercase=False, analyzer="char_wb",
                           ngram_range=char_ngrams, min_df=min_df, sublinear_tf=True)
    lex = Pipeline([("count", LexiconCounter(lexicon_terms)), ("scale", MaxAbsScaler())])
    return FeatureUnion([("word", word), ("char", char), ("lex", lex)])
