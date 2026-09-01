# Models and datasets used — original sources

Dr Hunter, 20 August 2026: *"make a note of all those and make sure you cite the original work for each of them."*

Harvard style (Cite Them Right), matching the rest of the dissertation. Keep this file in the repo root and copy the entries into the reference list as they get used.

---

## Pretrained models

### XLM-RoBERTa
**Hugging Face:** `xlm-roberta-base` · 279M parameters · SentencePiece tokenizer, 250k vocabulary

> Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L. and Stoyanov, V. (2020) 'Unsupervised cross-lingual representation learning at scale', *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Online, 5–10 July. Stroudsburg, PA: Association for Computational Linguistics, pp. 8440–8451.

**Why it is in the study:** the standard multilingual encoder baseline, pretrained on CommonCrawl in 100 languages. It has no targeted exposure to romanised Indian-language text, which makes it the right control against MuRIL rather than the expected winner.

---

### MuRIL
**Hugging Face:** `google/muril-base-cased` · BERT architecture · Google Research India

> Khanuja, S., Bansal, D., Mehtani, S., Khosla, S., Dey, A., Gopalan, B., Margam, D.K., Aggarwal, P., Nagipogu, R.T., Dave, S., Gupta, S., Gali, S.C.B., Subramanian, V. and Talukdar, P. (2021) *MuRIL: multilingual representations for Indian languages*. arXiv:2103.10730. Available at: https://arxiv.org/abs/2103.10730 (Accessed: 20 August 2026).

**Why it is in the study, and this is the important one.** <cite index="10-1">MuRIL is a multilingual language model built specifically for Indian languages that significantly outperforms multilingual BERT across the cross-lingual XTREME benchmark, and the authors report results on transliterated test sets — native script rendered into Latin — demonstrating the model's effectiveness on transliteration data.</cite> <cite index="14-1">It differs from mBERT in being trained on translation and transliteration pairs, using the IndicNLP-Transliteration library on Wikipedia together with the Dakshina dataset.</cite>

That is a direct match to this project's input. If romanisation is the source of difficulty, MuRIL is the model with a principled reason to handle it, and XLM-R is the control that isolates the effect.

---

### IndicBERT
**Hugging Face:** `ai4bharat/indic-bert` · ALBERT architecture · AI4Bharat

> Kakwani, D., Kunchukuttan, A., Golla, S., Gokul, N.C., Bhattacharyya, A., Khapra, M.M. and Kumar, P. (2020) 'IndicNLPSuite: monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages', *Findings of the Association for Computational Linguistics: EMNLP 2020*. Online, 16–20 November. Stroudsburg, PA: Association for Computational Linguistics, pp. 4948–4961.

**Note for the write-up:** this is IndicBERT v1, ALBERT-based, trained on IndicCorp for 11 Indian languages plus Indian English. A v2 exists in a separate repository. State explicitly which one you used — v1 via `ai4bharat/indic-bert` — since the two are not interchangeable and reviewers do check.

**Why it is in the study:** the lighter parameter-shared alternative. If it matches the larger encoders, that is a compute-efficiency finding worth reporting.

---

### BERT (architectural ancestor — cite when explaining the encoder family)

> Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: pre-training of deep bidirectional transformers for language understanding', *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. Minneapolis, MN, 2–7 June. Stroudsburg, PA: Association for Computational Linguistics, pp. 4171–4186.

---

## Datasets

### Bohra 2018 (primary within-dataset corpus)

> Bohra, A., Vijay, D., Singh, V., Akhtar, S.S. and Shrivastava, M. (2018) 'A dataset of Hindi-English code-mixed social media text for hate speech detection', *Proceedings of the Second Workshop on Computational Modeling of People's Opinions, Personality, and Emotions in Social Media*. New Orleans, LA, 6 June. Stroudsburg, PA: Association for Computational Linguistics, pp. 36–41.

**Outstanding action:** written usage permission from Deepanshu Vijay for the ethics documentation. Still not obtained. Chase this — it is the kind of thing that becomes a problem at submission rather than now.

### HASOC ICHCL 2021 and 2022

Cite the specific shared-task overview papers for the years used. Verify the exact volume and page numbers against the FIRE proceedings before submission rather than trusting a secondary source.

### Hinglish slur lexicon (seed terms)

> Mathur, P., Sawhney, R., Ayyar, M. and Shah, R. (2018) 'Did you offend me? Classification of offensive tweets in Hinglish language', *Proceedings of the 2nd Workshop on Abusive Language Online (ALW2)*. Brussels, 31 October. Stroudsburg, PA: Association for Computational Linguistics, pp. 138–148.

---

## Fine-tuning instability — cite these when reporting the variance

These two papers are what turn "my three runs disagreed" into a documented phenomenon with a literature behind it.

> Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H. and Smith, N. (2020) *Fine-tuning pretrained language models: weight initializations, data orders, and early stopping*. arXiv:2002.06305. Available at: https://arxiv.org/abs/2002.06305 (Accessed: 20 August 2026).

> Mosbach, M., Andriushchenko, M. and Klakow, D. (2021) 'On the stability of fine-tuning BERT: misconceptions, explanations, and strong baselines', *Proceedings of the 9th International Conference on Learning Representations (ICLR 2021)*. Online, 3–7 May.

**Read the Mosbach abstract and introduction at minimum.** You have an instance of exactly what it describes, and being able to say "this matches the failure mode Mosbach et al. characterise" is far stronger than "my runs were inconsistent."

---

## Software

> Wolf, T. et al. (2020) 'Transformers: state-of-the-art natural language processing', *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Online, 16–20 November. Stroudsburg, PA: Association for Computational Linguistics, pp. 38–45.

> Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

> Paszke, A. et al. (2019) 'PyTorch: an imperative style, high-performance deep learning library', *Advances in Neural Information Processing Systems 32*, pp. 8024–8035.

---

## Before you cite anything

Do not paste these into the dissertation unchecked. For each one, open the ACL Anthology or arXiv page, confirm the page numbers and year, and check the author list matches. The two arXiv preprints above (MuRIL, Dodge et al.) may have since appeared in a venue — if so, cite the published version instead.
