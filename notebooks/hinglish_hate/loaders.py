"""Dataset loaders that flatten every source into one schema:

    columns = [text, label, source, script]
      text   : str   raw utterance text
      label  : int   1 = hate/offensive, 0 = not
      source : str   'bohra2018' | 'hasoc2021' | 'hasoc2022'
      script : str   from preprocessing.script_profile

Each loader targets the exact on-disk schema of Mihir's Drive copy. Every
loader is defensive: it validates what it parsed and raises a clear error
rather than silently mislabelling.

Schemas (verified against the actual files):
  bohra2018   hate_speech.tsv  : headerless TSV, 2 cols [text, label],
                                 label in {yes, no}. All romanised.
  hasoc2021   topic/<thread_id>/{data.json, labels.json}
                                 data.json  = recursive tree of nodes
                                     {tweet_id, tweet, comments[], replies[]}
                                 labels.json = {tweet_id: "HOF"|"NONE"}
  hasoc2022   final.csv         : csv cols [_index, tweet_id, text, label],
                                 label in {HOF, NOT}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .preprocessing import script_profile

COLUMNS = ["text", "label", "source", "script"]

# label vocab -> binary. Everything maps through here so a surprise value
# fails loudly instead of being coerced to 0.
LABEL_MAP = {
    "yes": 1, "no": 0,          # bohra 2018
    "hof": 1, "not": 0,         # hasoc 2022
    "none": 0,                  # hasoc 2021 ("HOF"/"NONE")
    "hate": 1, "offensive": 1, "normal": 0,  # common variants, just in case
    "1": 1, "0": 0,
}


def _to_binary(raw) -> int:
    key = str(raw).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unrecognised label value: {raw!r}")
    return LABEL_MAP[key]


def _finish(rows, source: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["source"] = source
    df["script"] = df["text"].map(script_profile)
    return df[COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# Bohra 2018
# --------------------------------------------------------------------------
def load_bohra(path: str | Path) -> pd.DataFrame:
    """Load hate_speech.tsv (headerless, tab-separated: text <TAB> yes|no)."""
    rows = []
    tail = re.compile(r"^(?P<text>.*\S)\s+(?P<label>yes|no)\s*$", re.IGNORECASE)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            text, label = None, None
            if "\t" in line:
                *left, last = line.split("\t")
                cand = last.strip().lower()
                if cand in ("yes", "no"):
                    text, label = "\t".join(left).strip(), cand
            if label is None:  # fallback: last whitespace token is the label
                m = tail.match(line)
                if m:
                    text, label = m.group("text").strip(), m.group("label").lower()
            if label is None:
                # unparseable line - skip but keep count via the caller's validation
                continue
            rows.append((text, _to_binary(label)))
    if not rows:
        raise ValueError(f"No parseable rows in {path}. Check the file format.")
    return _finish(rows, "bohra2018")


# --------------------------------------------------------------------------
# HASOC 2021 (ICHCL conversational threads)
# --------------------------------------------------------------------------
def _walk_thread(node, out):
    """Recursively collect (tweet_id, tweet_text) from a data.json node."""
    if isinstance(node, dict):
        tid = node.get("tweet_id")
        txt = node.get("tweet")
        if tid is not None and txt is not None:
            out[str(tid)] = txt
        for child_key in ("comments", "replies"):
            for child in node.get(child_key, []) or []:
                _walk_thread(child, out)
    elif isinstance(node, list):
        for child in node:
            _walk_thread(child, out)


def load_hasoc2021(root: str | Path) -> pd.DataFrame:
    """Walk a HASOC-2021 tree and join thread text with per-tweet labels.

    `root` can be the split folder (e.g. .../data/train) or any ancestor -
    every folder that contains both data.json and labels.json is picked up.
    """
    root = Path(root)
    thread_dirs = {
        p.parent
        for p in root.rglob("labels.json")
        if (p.parent / "data.json").exists()
    }
    if not thread_dirs:
        raise ValueError(
            f"No data.json/labels.json thread folders found under {root}."
        )
    rows = []
    for d in sorted(thread_dirs):
        with open(d / "data.json", encoding="utf-8") as fh:
            data = json.load(fh)
        with open(d / "labels.json", encoding="utf-8") as fh:
            labels = json.load(fh)
        id2text = {}
        _walk_thread(data, id2text)
        for tid, lab in labels.items():
            text = id2text.get(str(tid))
            if text:
                rows.append((text, _to_binary(lab)))
    if not rows:
        raise ValueError(f"Found thread folders under {root} but joined 0 rows.")
    return _finish(rows, "hasoc2021")


# --------------------------------------------------------------------------
# HASOC 2022
# --------------------------------------------------------------------------
def load_hasoc2022(path: str | Path) -> pd.DataFrame:
    """Load final.csv (cols: _index, tweet_id, text, label; label HOF|NOT)."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "text" not in cols or "label" not in cols:
        raise ValueError(
            f"Expected 'text' and 'label' columns in {path}; got {list(df.columns)}"
        )
    rows = [
        (t, _to_binary(l))
        for t, l in zip(df[cols["text"]], df[cols["label"]])
        if isinstance(t, str) and pd.notna(l)
    ]
    return _finish(rows, "hasoc2022")


# --------------------------------------------------------------------------
# Corpus assembly
# --------------------------------------------------------------------------
def build_corpus(frames, drop_duplicates: bool = True) -> pd.DataFrame:
    """Concatenate loaded frames into one corpus.

    HASOC threads accumulate context, so exact-duplicate texts within a
    source are dropped by default.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        raise ValueError("No frames to build a corpus from.")
    corpus = pd.concat(frames, ignore_index=True)
    if drop_duplicates:
        corpus = corpus.drop_duplicates(subset=["source", "text"]).reset_index(drop=True)
    return corpus


def filter_romanised(df: pd.DataFrame, include_mixed: bool = False) -> pd.DataFrame:
    """Keep romanised (Latin-script) rows.

    Default keeps only 'mostly_latin'. Set include_mixed=True to also keep
    'mixed_script' rows (code-switched posts with some Devanagari).
    """
    keep = {"mostly_latin"}
    if include_mixed:
        keep.add("mixed_script")
    return df[df["script"].isin(keep)].reset_index(drop=True)
