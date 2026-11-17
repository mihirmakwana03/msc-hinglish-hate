# Evidence summary: every finding on one page

Each line is one claim, the number that supports it, and the file it comes from. Use it in
the meeting so Hunter can see the whole project at once, and keep it afterwards as the
skeleton of the Results chapter.

---

## 1. In-domain, the models tie

| Model | macro-F1 | Source |
|---|---|---|
| MuRIL | 0.654 | `confidence_scores.csv` |
| XLM-RoBERTa | 0.653 | |
| IndicBERT v1 | 0.652 | |
| TF-IDF + LogReg | 0.642 | |

Multi-seed, three seeds each: all within one standard deviation of each other.
`multiseed_results.csv`

**Claim:** the transformer advantage reported at the formative viva was seed noise.

## 2. The errors are systematic, not uniform

| Model | Hate recall | Not-hate recall |
|---|---|---|
| TF-IDF + LogReg | 0.578 | 0.712 |
| IndicBERT v1 | 0.536 | 0.765 |
| XLM-RoBERTa | 0.533 | 0.768 |
| MuRIL | **0.476** | **0.822** |

`confusion_counts.csv`, `figures/confusion_grid_within.png`

**Claim:** the highest-accuracy model has the worst hate recall. Accuracy is bought by
abandoning the minority class.

## 3. Cross-dataset performance collapses, on two separate corpora

| Test set | Hate rate | macro-F1 | Hate recall |
|---|---|---|---|
| Bohra (within) | 0.363 | 0.642 | 0.578 |
| HASOC 2021 (cross) | 0.478 | 0.439 | 0.127 |
| HASOC 2022 (cross) | 0.513 | 0.435 | 0.155 |

`cross_dataset_three_corpora.csv`

**Claim:** replicates across two held-out corpora, four thousandths apart.

## 4. Transfer direction decides which way the model fails

| Trained on | Tested on | Hate recall | Not-hate recall |
|---|---|---|---|
| Bohra, 36.3% hate | HASOC, 51.3% hate | **0.167** | 0.850 |
| HASOC, 51.2% hate | Bohra, 36.3% hate | **0.928** | **0.069** |

`within_hasoc_replication.csv`

**Claim:** the model carries its training base rate across. Direction predicts the failure
mode. Magnitude does not scale with the gap, so state direction only.

## 5. It is not one bad dataset

Within HASOC 2022, five-fold: **0.662 ± 0.007**. Within Bohra, five-fold: **0.657 ± 0.005**.

`hasoc_cv_folds.csv`

**Claim:** the difficulty is a property of the task, not of the Bohra corpus.

## 6. It is not a coding error

| Condition | Train posts | Hate rate | macro-F1 |
|---|---|---|---|
| Hinglish (Bohra) | 3,659 | 36.3% | 0.621 |
| English (Davidson), matched | 3,151 | 36.3% | **0.787** |
| English, full corpus | 19,826 | 5.8% | 0.701 |

`english_comparison.csv`, `figures/english_control.png`

**Claim:** identical pipeline, same class balance, 500 fewer posts. The method works.

## 7. More data would not help

Hinglish: 0.597 at 700 training posts, 0.621 at 3,659. Movement over the last 30% of the
data: **zero**.

`learning_curves.csv`, `figures/learning_curves.png`

**Claim:** the curve has flattened. The limit is not sample size.

## 8. Reducing model capacity does not help either

Five-fold cross-validated, after a sweep of regularisation, vocabulary size and model family:

| Configuration | macro-F1 |
|---|---|
| current | **0.6473 ± 0.0136** |
| max_features=500 | 0.6303 |
| Decision tree, depth 5 (train-test gap 0.009) | 0.6141 |

`capacity_sweep_cv.csv`, `figures/capacity_sweep.png`

**Claim:** a model that barely overfits is the worst in the table. Parameter counts across all
models span a factor of 5,845 and 0.012 macro-F1 separates them. Overfitting is real but it
is not the ceiling.

## 9. Transformers are more attack-resistant, but the reason is not what it seemed

| Pipeline | Romanisation drop | Character drop |
|---|---|---|
| TF-IDF with lexicon features | 7.5% | **25.2%** |
| TF-IDF without lexicon features | 11.5% | **5.6%** |
| XLM-RoBERTa | 5.5% | 3.5% |

`attack_lexicon_independence.csv`, `multiseed_results.csv`

**Claim:** the lexicon features are the brittle component, not the n-grams. The feature that
most improves clean performance fails hardest under attack.

**Secondary claim:** the romanisation attack is not circular. Removing the lexicon makes it
stronger, not weaker, which is the opposite of what shared-rule inflation would produce.

## 10. Macro-F1 hides a difference between models

| Model | Confidence score | Errors made with >75% on the wrong class |
|---|---|---|
| XLM-RoBERTa | +0.152 | **7.6%** |
| TF-IDF | +0.149 | 12.0% |
| IndicBERT v1 | +0.172 | 19.6% |
| MuRIL | **+0.259** | **71.9%** |

`confidence_scores.csv`

**Claim:** three models that macro-F1 calls identical differ ninefold in confident error rate.
MuRIL scores highest on the metric and is the worst behaved underneath it.

**Second finding:** cross-dataset, `mean_D_correct` and `mean_D_wrong` are equal to three
decimals on every model. No confidence threshold could make these models safe on new data.

## 11. The ceiling is the label definition

Of 140 missed hate posts, hand-categorised by a Hinglish speaker:

| Category | Posts | % |
|---|---|---|
| Political criticism labelled as hate | 58 | 41.4 |
| Would not be called hate on reading | 26 | 18.6 |
| **Together** | **84** | **60.0** |

`true_hate_MISSED_categorised_v2.csv`

Supporting: missed hate posts are 7.9% likely to contain no lexicon term, caught hate 0.5%.
But 92% of missed posts do contain one, so vocabulary coverage explains only a slice.
`cell_lexicon_profile.csv`

**Claim:** 60% of what the model is marked wrong on is arguable. This is the ceiling, and it
is the finding that required reading the posts.

---

## How these fit together

Findings 6, 7 and 8 rule out the three obvious explanations: the method, the data volume, and
model capacity. Findings 3, 4 and 5 establish that the problem is real and reproduces.
Finding 11 supplies what is left.

That is the argument of the dissertation, and it is a complete one.

---

## What is not yet evidence

**The four-cell reading.** 480 verdict cells empty. Would upgrade finding 11 from a
categorisation to a measured ceiling via notebook 19.

**Annotation limitations.** Finding 11 is single-annotator, with no inter-annotator agreement,
and the annotator could see the model's prediction. State all three.

**The HASOC 2021 row count.** The loader reports 4,911 rows; you have quoted 2,819 elsewhere.
Settle it before Table 3.1 is final.

**Dropped scope.** LoRA with Llama-3.2-3B and SHAP were in the proposal. They belong in
suggestions for future work, named explicitly.
