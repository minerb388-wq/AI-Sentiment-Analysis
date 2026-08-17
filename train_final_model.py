import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/processed/video_games_sentiment_clean.csv"

VECTORIZER_FILE = "models/tfidf_vectorizer.joblib"
MODEL_FILE = "models/logistic_regression_model.joblib"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

X = df["clean_text"]
y = df["sentiment"]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ============================================================
# TF-IDF
# ============================================================

print("Training TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=30_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(X_train_tfidf, y_train)

# ============================================================
# SAVE MODEL + VECTORIZER
# ============================================================

joblib.dump(vectorizer, VECTORIZER_FILE)
joblib.dump(model, MODEL_FILE)

print("\n" + "=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print(f"Vectorizer: {VECTORIZER_FILE}")
print(f"Model:      {MODEL_FILE}")

print("\nTraining complete!")
