# Multilingual Regional News Classifier

A single multilingual model that classifies news headlines into topics — **politics/national, international, sports, entertainment, technology** — across **English, Hindi, and Bengali**, using one shared model instead of training separate classifiers per language.

**[Live demo](https://multilingual-regional-news-classifier-zgmxwwvqb96qsdn23mzeya.streamlit.app)** — paste a headline in any of the three languages and get a predicted category, detected language, and the nearest topically-related headlines in the other two languages.

> Note: the app runs on Streamlit Community Cloud's free tier, which sleeps after 12 hours of inactivity. If you see a "wake up" screen, just click through — it takes under a minute.

---

## What this demonstrates

- **Cross-lingual embeddings over translate-then-classify.** Text in all three languages is embedded with [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) (Language-Agnostic BERT Sentence Embeddings) into one shared 768-dimensional space — semantically similar headlines land close together regardless of language, with no translation step involved anywhere in the pipeline.
- **Zero-shot cross-lingual transfer.** A classifier trained only on English + Hindi is evaluated on Bengali, a language it never saw a single labeled example of during training — and still reaches 77% accuracy, because the shared embedding space generalizes.
- **Diagnosing and fixing a real labeling-schema problem**, not just a modeling one — see [Results](#results) below.
- **Data imbalance across languages/categories**, handled explicitly rather than ignored.

## Dataset

- **Hindi + Bengali:** [L3Cube-IndicNews](https://github.com/l3cube-pune/indic-nlp) (Short Headlines Classification split) — chosen because it provides a harmonized category schema across languages out of the box.
- **English:** the same L3Cube-IndicNews corpus's English SHC split, for consistency with the other two languages.
- After mapping each language's native categories onto a canonical schema and filtering to the 6 categories with solid coverage in all three languages, then deduplicating, the final dataset is ~65,000 headlines across English, Hindi, and Bengali.
- **Known limitation:** categories present in only one or two languages in the source data (business, crime, health, education, lifestyle, auto) were excluded from the core model rather than force-fit — a production system would need supplementary data collection to cover them for every language.

## Approach

1. **Preprocessing** — Unicode NFC normalization and whitespace cleanup only. Deliberately *no* aggressive lowercasing or stemming, since English NLP habits don't transfer safely to Devanagari/Bengali script.
2. **Embeddings** — every headline is embedded with LaBSE into a shared space, L2-normalized for cosine similarity.
3. **Classification** — a single `LogisticRegression` trained on top of the frozen LaBSE embeddings (no fine-tuning).
4. **Evaluation** — per-language accuracy/F1, a zero-shot English+Hindi → Bengali transfer test, confusion-matrix diagnosis, and a monolingual-vs-multilingual comparison.

## Results

| Setup | English | Hindi | Bengali |
|---|---|---|---|
| Monolingual (own language only) | 89.2% | 87.9% | 85.3% |
| Multilingual (single shared model) | 87.0% | 86.0% | 83.0% |
| Zero-shot (Bengali unseen at train time) | — | — | **77.0%** |

**The diagnosis that got zero-shot from 63% → 77%:** the first zero-shot run showed a large accuracy gap, concentrated almost entirely in two categories — `politics` and `national`. A confusion matrix confirmed the two were being confused in both directions, in roughly half of all instances. Rather than an embedding or model weakness, this turned out to be **an inconsistent labeling convention across languages/outlets** — the same election-announcement headline might be labeled `politics` in one source and `national` in another. Merging the two categories into one (`politics_national`) resolved most of the gap and confirmed the diagnosis.

**Monolingual vs. multilingual is a genuine tradeoff, not a clean win either way:** monolingual models edge out the shared multilingual model by ~2 points per language (a known "capacity-sharing" cost of one model serving multiple languages) — but the multilingual model can classify a language with **zero labeled training data**, which no monolingual model can do at all.

Full analysis, including the embedding-space sanity checks and per-language confusion matrices, is in [`multilingual_regional_news_classifier.ipynb`](./multilingual_regional_news_classifier.ipynb).

## Repo structure

```
├── app.py                          # Streamlit app
├── requirements.txt
├── multilingual_clf.joblib         # trained on English + Hindi + Bengali
├── zeroshot_clf.joblib             # trained on English + Hindi only
├── reference_data.csv              # sampled headlines for nearest-neighbor lookup
├── reference_embeddings.npy        # matching LaBSE embeddings for reference_data.csv
└── multilingual_regional_news_classifier.ipynb   # full pipeline: data → preprocessing → embeddings → evaluation
```

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-regional-news-classifier.git
cd multilingual-regional-news-classifier
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the LaBSE model (~1.8GB) from Hugging Face Hub — subsequent runs are fast, since Streamlit caches it in memory for the session.

## Tech stack

Python · sentence-transformers (LaBSE) · scikit-learn · pandas · Streamlit · langdetect

## Limitations & possible extensions

- No agriculture category in the source dataset — would need separate data collection.
- Cross-lingual `sports` recall is weaker than in-language (see notebook Section 8) — headlines with regional context (e.g. local league names) transfer less cleanly than universal-topic categories like technology.
- Currently deployed with a sampled reference set for the nearest-neighbor feature; a full-dataset version with embeddings hosted on Hugging Face Hub would give richer neighbor results.
- Classifier is a simple LogisticRegression on frozen embeddings — a small MLP or fine-tuning LaBSE itself could likely close some of the remaining monolingual/multilingual gap.
