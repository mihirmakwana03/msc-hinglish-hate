# Defence log

A record of what went wrong during this project and how it was resolved. Kept as the raw
material for the critical reflection section, which Dr Hunter asked should cover what was
experienced, what was learnt, and the challenges faced.

Every entry below is verifiable from the commit history, the notebook outputs, or
`writing/all_results.csv`.

---

## 1. A learning-rate increase collapsed two models

Raising the learning rate from 2e-5 to 3e-5 was intended to address apparent undertraining.
Instead both models degenerated.

- **XLM-RoBERTa, 6 epochs, 3e-5:** macro-F1 0.312. Predicted the hate class for 874 of 915
  test posts. Training loss sat flat at approximately 0.6931, which is ln(2), the
  chance-level loss for a balanced two-class problem. The model never left its initialisation.
- **MuRIL, 6 epochs, 3e-5:** the mirror image. Hate-class F1 of 0.000, because it predicted
  not-hate for every one of the 915 test posts.

Both runs are retained in `all_results.csv` and flagged `degenerate=yes`, because a collapse
is a result about fine-tuning stability rather than something to discard. The behaviour is
consistent with the instability reported by Dodge et al. (2020) and Mosbach et al. (2021).

**Learnt:** a flat training loss at ln(2) is a diagnostic, not a slow start. It is worth
checking the loss against chance level before interpreting any downstream metric.

## 2. Two variables were changed at once

The failed run above altered both the epoch count and the learning rate in the same step,
so when it collapsed the cause was ambiguous and a further run was needed to separate them.

**Learnt:** change one setting per run, however tempting it is to save GPU time.

## 3. Undertraining was the wrong diagnosis

The hypothesis behind the learning-rate change was that three epochs was too few. A
20-epoch run disproved it: training loss fell steadily throughout while test macro-F1 did
not improve. That is overfitting, and the original three-epoch setting was already at or
past the useful point.

**Learnt:** a training-loss curve on its own says nothing about generalisation. The
20-epoch run was the cheapest way to settle the question and should have come first.

## 4. Results came from three different machines

Experiments ran on Google Colab Pro+, a university RTX A5000, and a personal laptop at
different points. Re-running seed 42 on a different machine did not reproduce exactly:
XLM-RoBERTa matched to within 0.001 macro-F1, but MuRIL moved by 0.015.

Setting every seed does not make results bit-for-bit reproducible across machines, because
cuDNN selects different convolution and matrix-multiplication kernels depending on the GPU
and library versions, and floating-point addition is not associative. The multi-seed
protocol absorbs this, since a 0.015 shift sits inside the measured seed spread.

**Learnt:** report hardware alongside results, and prefer a mean over seeds to a single
headline number. This is recorded as a limitation.

## 5. A results file accumulated duplicate rows

`multiseed_results.csv` was appended to on each run without being cleared, so it reached 25
and then 31 rows where 9 were expected. A MuRIL summary was computed and reported from
n=10 rows before the duplication was noticed; the correct figure over three seeds was
0.667 ± 0.012 rather than the 0.666 ± 0.014 first reported.

Fixed by truncating the log at the start of the run (`LOG.unlink(missing_ok=True)`).

**Learnt:** an append-only results file silently corrupts every aggregate computed from it.
Check the row count against what was expected before reading any summary.

## 6. Two different models were both called IndicBERT

The epoch sweep used `ai4bharat/indic-bert`, the ALBERT-based v1 model of Kakwani et al.
(2020). The first multi-seed run used `ai4bharat/IndicBERTv2-MLM-only`, a BERT-based model
from Doddapaneni et al. (2023). These are different architectures trained on different
corpora, and comparing their scores as though they were the same system would have been
wrong.

**Learnt:** record the exact Hugging Face identifier, not the familiar name. This is why
`MODEL_CITATIONS.md` exists.

## 7. A model repository became gated mid-project

`ai4bharat/indic-bert` began returning HTTP 401 on all three seeds partway through the
multi-seed runs, having previously downloaded without authentication. Resolved by
requesting access on the Hugging Face Hub and reading the token from the `HF_TOKEN`
environment variable.

The token is read from the environment and never written into a notebook, so no credential
has entered the repository at any point. This was verified by searching the full commit
history for the Hugging Face token pattern.

**Learnt:** an external dependency can change its access terms during a project. Credentials
belong in the environment, not in source.

## 8. A tokenisation bug in the error analysis

The lexicon counter in `12_error_analysis.ipynb` split text on whitespace, so punctuation
stayed attached to words and suppressed matches against the lexicon. The production
featuriser in `features.py` uses a regular-expression tokeniser and was never affected, so
no model result changed. Only the reported diagnostic figures moved, from 1.65 to 1.72 mean
lexicon hits on caught posts and from 1.0% to 0.5% zero-hit rate.

**Learnt:** analysis code that reimplements a step of the pipeline will drift from it.
Importing the shared tokeniser removes the possibility.

## 9. The attack evaluation was placed after the model was deleted

The adversarial evaluation block was first inserted after `del model`, which raised
`NameError: cannot access free variable 'model'`. Moved to before the deletion.

## 10. The robustness prediction was wrong

The expectation set before running the attack was that the TF-IDF baseline would prove
more resistant than the transformers, on the reasoning that character n-grams degrade
gracefully under spelling changes while a fixed subword vocabulary does not.

The opposite held, by a wide margin. Under full character corruption the baseline lost
25.2% of its macro-F1 and XLM-RoBERTa lost 3.5%. The explanation is that subword
tokenisation re-segments a corrupted word into valid pieces and the surrounding context
recovers the meaning, whereas TF-IDF features carry no context and a corrupted token simply
becomes a different feature.

**Learnt:** the prediction was recorded before the experiment ran, which is what made the
result interpretable rather than merely surprising.

## 11. The word limit was taken from the wrong part of the guidelines

The planned length was based on the generic module descriptor figure of 12,000 to 18,000
words. The operative figure is in section 2 of the guidelines, which is specific to project
type, and this project is registered in the proposal as Type 1, experiment-based: a report
of 3,000 to 5,000 words accompanied by an assessed artefact.

Caught on 13 October, before the writing phase began.

**Learnt:** where a document gives both a general and a specific figure, the specific one
governs. Check requirements against the registration document rather than a summary.

---

## What this adds up to

Most of the failures in this project were not failures of code. They were failures of
inference: a flat loss read as a slow start, a training curve read as evidence about
generalisation, an append-only file trusted without a row count, two models treated as one
because they shared a name. The code errors, by contrast, surfaced immediately and cost
minutes.
