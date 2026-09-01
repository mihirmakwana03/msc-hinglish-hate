# Hinglish Hate Speech Detection — MSc Dissertation

Benchmarking transformer and statistical models on hate speech classification in romanized Hindi-English (Hinglish) text.

## Overview

This repository contains the dissertation work evaluating pretrained language models on the task of classifying hate speech in transliterated (romanized) Indian-language text. The study compares:

- **Statistical baseline:** TF-IDF + Logistic Regression (with character, word, and lexicon features)
- **Multilingual encoders:** XLM-RoBERTa, MuRIL
- **Indian-language specialists:** IndicBERT

Evaluation uses both within-dataset (5-fold CV, single-split) and cross-dataset (HASOC 2022) protocols to measure generalization.

---

## Project Structure

```
data/
  hate_speech.tsv              # Bohra 2018 (primary within-dataset corpus)
  hinglish_slur_lexicon.csv    # Custom lexicon for feature engineering
  hate_lexicon.txt
  HASOC-2022-main/             # Annotated hate speech corpus
  data-20220220T075606Z-001/   # Additional training data
  test-20220220T075607Z-001/   # Test sets

notebooks/
  01_data_exploration.ipynb    # Schema, distribution, language profiles
  02_baseline.ipynb            # Statistical baseline (TF-IDF + LogReg)
  03_data_quality.ipynb        # Cleaning, annotation agreement, edge cases
  04_lexicon.ipynb             # Custom lexicon development
  05_features.ipynb            # Feature matrix construction
  06_baseline_model.ipynb      # Statistical model evaluation
  07_cross_validation.ipynb    # 5-fold CV for statistical model
  08_transformer_baseline_*.ipynb  # Model training for XLM-R, MuRIL, IndicBERT
  10_results_consolidation.ipynb   # Results synthesis

notebooks/hinglish_hate/       # Shared pipeline library
  loaders.py                   # Dataset loading (Bohra, HASOC 2021/22)
  preprocessing.py             # Text normalization, cleaning
  features.py                  # Feature extraction (lexicon, ngrams, etc.)
  baseline.py                  # Statistical model definition and evaluation
  variants.py                  # Model variants and ablations
  make_fixtures.py             # Synthetic test data for CI

writing/
  master_results_table.csv     # Consolidated results (see below)
  sample_posts_for_reading.csv # Annotated examples for discussion
  all_results.csv

MODEL_CITATIONS.md             # Pretrained model and dataset sources (Harvard style)
```

---

## Results Summary

The main results table ([writing/master_results_table.csv](writing/master_results_table.csv)) tracks:

| Metric | Description |
|--------|-------------|
| `family` | Model type (statistical, transformer) |
| `model` | Model identifier or checkpoint |
| `within_macro_f1` | Macro-averaged F1 on hold-out test set (Bohra 2018) |
| `within_hate_f1` | Class-specific F1 for hate speech (primary metric) |
| `accuracy` | Test accuracy |
| `cross_h22_macro_f1` | Macro F1 on HASOC 2022 cross-dataset test |
| `cross_h22_hate_f1` | Hate class F1 on HASOC 2022 |
| `degenerate` | Flags training failures (e.g., collapsed loss) |
| `note` | Configuration, best result markers, ablations |

**Best results to date:**
- Statistical: TF-IDF word+char+lexicon + LogReg (macro F1: 0.657, within-dataset; 0.567 hate F1)
- XLM-RoBERTa: 3 epochs, lr=2e-05 (macro F1: 0.665, within-dataset; 0.565 hate F1; 0.448 cross-dataset)
- MuRIL: 6 epochs, lr=2e-05 (macro F1: 0.665, within-dataset; 0.543 hate F1; 0.385 cross-dataset)
- **IndicBERT: pending** (see below)

---

## Running the Pipeline Locally

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Hugging Face `transformers` library
- Jupyter (for notebooks)

Install dependencies:
```bash
pip install -r requirements.txt  # if available, or:
pip install torch transformers pandas scikit-learn jupyter
```

### Quick Start (Statistical Baseline)

1. Copy `hinglish_hate/` and `notebooks/02_baseline.ipynb` to the same directory
2. Place datasets at a path (e.g., `./data/`) with:
   - `hate_speech.tsv`
   - `final.csv` (HASOC 2022)
   - HASOC 2021 thread folders
3. Update `DATA_ROOT` in the notebook
4. Run all cells
   - Cell 6: within-dataset macro-F1
   - Cell 7: cross-dataset results

### Transformer Models (Colab Recommended)

For training transformer models on GPU:
1. Upload `hinglish_hate/`, `08_transformer_baseline_*.ipynb`, and datasets to Google Drive
2. Open notebook in Colab, mount Drive
3. Set `DATA_ROOT` and `CHECKPOINT_DIR`
4. Run training cells

---

## Reproducibility Notes

- **Random seeds:** Set in each notebook (affects train/test split, model initialization)
- **Cross-dataset:** Uses same 915 test rows across all transformer runs for comparability
- **Degenerate runs:** Marked explicitly (e.g., loss collapsed to ln(2), predicting all hate)
- **Lexicon:** Custom slur list in `data/hinglish_slur_lexicon.csv` used for both statistical and transformer featurization
- **Tokenization:** Transformers use model-specific tokenizers (SentencePiece for XLM-R, native for others)

---

## Pending Work

- **IndicBERT results:** Training run in progress. When complete, add row to `master_results_table.csv` with:
  - Model: `ai4bharat/indic-bert`
  - Protocol: single split (to match XLM-R/MuRIL)
  - Expected metrics: within-dataset macro F1, hate F1; cross-dataset F1
  - Note: v1 (ALBERT-based), not v2

---

## References

Full citations (Harvard style) in [MODEL_CITATIONS.md](MODEL_CITATIONS.md).

**Key papers:**
- Bohra et al. (2018) — Primary hate speech corpus
- Khanuja et al. (2021) — MuRIL model
- Kakwani et al. (2020) — IndicBERT model
- Conneau et al. (2020) — XLM-RoBERTa model

---

## License & Attribution

Datasets sourced from published corpora (Bohra 2018, HASOC shared tasks). Models from Hugging Face Hub. See `MODEL_CITATIONS.md` for full attribution.
