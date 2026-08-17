import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. LOAD DATA
# ============================================================

INPUT_FILE = "data/processed/video_games_sentiment_clean.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("AMAZON VIDEO GAME SENTIMENT ANALYSIS")
print("=" * 60)

print(f"\nTotal reviews: {len(df):,}")

# ============================================================
# 2. INPUT AND TARGET
# ============================================================

X = df["clean_text"]
y = df["sentiment"]

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nDataset split:")
print(f"Training reviews: {len(X_train):,}")
print(f"Testing reviews:  {len(X_test):,}")

# ============================================================
# 4. TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF complete!")
print(f"Training matrix: {X_train_tfidf.shape}")
print(f"Testing matrix:  {X_test_tfidf.shape}")

# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=42),
}

# ============================================================
# 6. TRAIN AND EVALUATE
# ============================================================

results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    )

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# ============================================================
# 7. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 8. BEST MODEL
# ============================================================

best_model = results_df.loc[results_df["F1 Score"].idxmax()]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model:     {best_model['Model']}")
print(f"Accuracy:  {best_model['Accuracy']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall:    {best_model['Recall']:.4f}")
print(f"F1 Score:  {best_model['F1 Score']:.4f}")
