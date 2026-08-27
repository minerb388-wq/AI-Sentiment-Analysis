# Amazon Product Review Sentiment Analysis

A machine-learning project that classifies Amazon product reviews as **negative**,
**neutral**, or **positive**. The Streamlit dashboard supports two product categories:

- Video Games
- All Beauty

The application lets users analyse a review, explore the processed dataset, and
inspect model performance for each category.

## Project objective

The objective is to build and evaluate a reproducible sentiment-analysis workflow
for Amazon customer reviews. The workflow samples category data, derives sentiment
labels from star ratings, cleans review text, trains a TF-IDF-based classifier, and
serves predictions through a web dashboard.

## Dataset

The project uses category review data from the
[Amazon Reviews'23 dataset](https://amazon-reviews-2023.github.io/). Reviews are
mapped to three sentiment classes using their star ratings:

| Star rating | Sentiment label |
| --- | --- |
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

For each category, the preparation pipeline starts with a reproducible balanced
sample of 50,000 reviews. Cleaning removes missing text, duplicate reviews, and
reviews that become empty after preprocessing.

| Category | Initial sample | Cleaned reviews |
| --- | ---: | ---: |
| Video Games | 50,000 | 48,371 |
| All Beauty | 50,000 | 48,443 |

The raw downloaded data is deliberately excluded from version control; only the
processed datasets and required model artifacts are included for deployment.

## Methodology

### Text preprocessing

For every review, the pipeline:

1. Removes missing and duplicate review text.
2. Converts text to lowercase.
3. Removes HTML, URLs, punctuation, numbers, and excess whitespace.
4. Removes English stop words while preserving important negation words such as
   `not`, `no`, and `never`.
5. Lemmatizes tokens with NLTK WordNet lemmatization.
6. Removes reviews with no remaining cleaned text.

### Model training and evaluation

The data is split using an 80/20 stratified train/test split with `random_state=42`.
TF-IDF is fitted on the training split only, which prevents test-set information from
leaking into the feature vocabulary.

The deployed classifier is Logistic Regression with:

- Maximum 30,000 TF-IDF features
- Unigrams and bigrams (`ngram_range=(1, 2)`)
- `min_df=2`, `max_df=0.95`, and sublinear term frequency
- `max_iter=1000` and `random_state=42`

The dashboard reports accuracy, weighted precision, weighted recall, weighted F1,
a class-level report, and a confusion matrix using the same fixed test split.

## Current model results

The current Logistic Regression models achieved the following held-out test
accuracy during local validation:

| Category | Accuracy |
| --- | ---: |
| Video Games | 68.34% |
| All Beauty | 69.33% |

Run `python evaluate_models.py` to reproduce a comparison among a majority-class
baseline, Multinomial Naive Bayes, and Logistic Regression. It writes reusable CSV
and JSON summaries under `results/`.

## Run the application locally

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies and NLTK resources

```powershell
pip install -r requirements.txt
python setup_nltk.py
```

### 3. Start Streamlit

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Prepare and train a category

The reusable category pipeline accepts an Amazon Reviews JSONL or JSONL.GZ source.
For example, to rebuild the All Beauty artifacts:

```powershell
python prepare_and_train_category.py `
  --raw data/raw/All_Beauty.jsonl `
  --name all_beauty `
  --size 50000
```

This creates:

- `data/processed/all_beauty_sentiment_50k.csv`
- `models/all_beauty_tfidf_vectorizer.joblib`
- `models/all_beauty_logistic_regression_model.joblib`

## Reproduce model comparison

```powershell
python evaluate_models.py
```

To evaluate only one category:

```powershell
python evaluate_models.py --category "All Beauty"
```

## Run automated tests

Install development dependencies, then run the focused pipeline tests:

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

The tests verify the rating-to-sentiment mapping, text-cleaning behavior, and
balanced sampling on a small temporary dataset.

## Project structure

```text
AI-Sentiment-Analysis/
├── app.py                           # Streamlit dashboard
├── prepare_and_train_category.py    # Sampling, cleaning, and category training
├── preprocess_data.py               # Video Games cleaning workflow
├── evaluate_models.py               # Reproducible model comparison
├── data/
│   └── processed/                   # Cleaned category datasets
├── models/                          # Saved TF-IDF vectorizers and classifiers
├── results/                         # Generated evaluation outputs
├── tests/                           # Automated tests
└── requirements.txt
```

## Limitations

- Sentiment labels are inferred from star ratings rather than manually annotated
  text, so mixed or sarcastic reviews may receive imperfect labels.
- The model is trained only on English text and may not generalize well to other
  languages, categories, or writing styles.
- A bag-of-words TF-IDF model has limited contextual understanding, especially for
  negation, sarcasm, and product-specific terminology.
- Results are based on a single train/test split; cross-validation and error
  analysis would provide a more robust estimate of generalization performance.

## Deployment

The dashboard is deployed with Streamlit Community Cloud. Changes pushed to the
repository's `main` branch trigger a new deployment automatically.

## Academic notes

For an assignment report or presentation, cite the Amazon Reviews'23 dataset,
describe the rating-to-sentiment conversion, report the held-out metrics, and
discuss the limitations above. Keep a record of the exact project commit used for
your submitted results so they remain reproducible.
