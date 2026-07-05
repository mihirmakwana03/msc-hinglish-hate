# Hinglish Hate Speech - Data Loader + Statistical Baseline

The safety-net stage: flatten Bohra 2018 + HASOC 2021 + HASOC 2022 into one
frame, filter to romanised rows, and get a TF-IDF + Logistic Regression F1.

## Layout

```
hinglish_hate/
  preprocessing.py   script_profile (same as 01_data_exploration) + clean_text
  loaders.py         load_bohra / load_hasoc2021 / load_hasoc2022
                     build_corpus / filter_romanised
  baseline.py        build_pipeline / evaluate_within / evaluate_cross
02_baseline.ipynb    Colab-ready: mount Drive -> load -> filter -> F1
tests/
  make_fixtures.py   schema-matched synthetic files
  run_pipeline.py    runs the whole pipeline on fixtures (proves it works)
```

## Run it (Colab)

1. Put `hinglish_hate/` and `02_baseline.ipynb` in the same place (upload to
   Colab, or keep both in a Drive folder).
2. Add the shared dataset folder to *My Drive* so it mounts.
3. Open `02_baseline.ipynb`, set `DATA_ROOT` to the folder holding
   `hate_speech.tsv`, `final.csv`, and the HASOC-2021 thread folders.
4. Run all cells. Cell 6 prints the **within-Bohra macro-F1** (the safety-net
   number). Cell 7 prints the **cross-dataset** numbers.

## Run it (local sanity check, no real data)

```bash
python tests/run_pipeline.py
```

This builds fixtures shaped like the real files and runs every stage. It is
already green - it confirms the loaders, the romanised filter, and the baseline
all work. (The F1 is 1.0 on fixtures only because the synthetic vocab is
perfectly separable; it is a plumbing test, not a result.)

## Unified schema

Every loader returns `[text, label, source, script]`:

| column | meaning |
|--------|---------|
| text   | raw utterance |
| label  | 1 = hate/offensive, 0 = not |
| source | `bohra2018` / `hasoc2021` / `hasoc2022` |
| script | `mostly_latin` / `mixed_script` / `mostly_devanagari` / `other` / `empty` |

`filter_romanised(df, include_mixed=False)` keeps `mostly_latin` (add
`mixed_script` with `include_mixed=True`).

## Verified source schemas

Confirmed against the actual Drive files (not assumed):

**Bohra 2018 - `hate_speech.tsv`**
Headerless TSV, two columns: `text <TAB> label`, label in `{yes, no}`
(`yes` = hate). ~4,575 rows, all romanised. `yes -> 1`, `no -> 0`.

**HASOC 2022 - `final.csv`**
CSV columns: `_index, tweet_id, text, label`, label in `{HOF, NOT}`.
Mixed script (some rows Devanagari). `HOF -> 1`, `NOT -> 0`.

**HASOC 2021 - nested ICHCL threads**
`.../<topic>/<thread_id>/` each containing:
- `data.json` - recursive tree; each node `{tweet_id, tweet, comments[], replies[]}`
- `labels.json` - `{tweet_id: "HOF" | "NONE"}`
The loader walks the tree, joins text to labels by `tweet_id`. Mixed script.
`HOF -> 1`, `NONE -> 0`.

## Baseline choices

- `TfidfVectorizer`: word 1-2 grams, `min_df=2`, `sublinear_tf=True`,
  `clean_text` preprocessor (strips URLs/@mentions, unwraps hashtags, collapses
  elongation, lowercases). No stopword removal - no reliable romanised-Hinglish
  stopword list, and that is a deliberate later experiment.
- `LogisticRegression`: `class_weight='balanced'`, `max_iter=2000`, seed 42.
- Headline metric: **macro-F1** (both classes weighted equally), plus hate-class
  F1 and accuracy.

Keep the exact `bohra_r` / romanised test frames fixed so every later model
(XLM-R, IndicBERT, MuRIL, LoRA LLM) is scored on identical splits.
