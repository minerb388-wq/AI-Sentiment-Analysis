import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

INPUT_FILE = "data/processed/video_games_sentiment_clean.csv"

df = pd.read_csv(INPUT_FILE)

X = df["clean_text"]
y = df["sentiment"]

# --------------------------------------------------
# Same train/test split as baseline
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# Experiment 1: 20,000 features
# --------------------------------------------------

print("=" * 60)
print("EXPERIMENT 1: 20,000 FEATURES")
print("=" * 60)

vectorizer_20k = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_20k = vectorizer_20k.fit_transform(X_train)
X_test_20k = vectorizer_20k.transform(X_test)

model_20k = LogisticRegression(max_iter=1000, random_state=42)
model_20k.fit(X_train_20k, y_train)
pred_20k = model_20k.predict(X_test_20k)

acc_20k = accuracy_score(y_test, pred_20k)
precision_20k, recall_20k, f1_20k, _ = precision_recall_fscore_support(
    y_test,
    pred_20k,
    average="weighted",
    zero_division=0,
)

print(f"Accuracy:  {acc_20k:.4f}")
print(f"Precision: {precision_20k:.4f}")
print(f"Recall:    {recall_20k:.4f}")
print(f"F1 Score:  {f1_20k:.4f}")

# --------------------------------------------------
# Experiment 2: 30,000 features
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXPERIMENT 2: 30,000 FEATURES")
print("=" * 60)

vectorizer_30k = TfidfVectorizer(
    max_features=30_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_30k = vectorizer_30k.fit_transform(X_train)
X_test_30k = vectorizer_30k.transform(X_test)

model_30k = LogisticRegression(max_iter=1000, random_state=42)
model_30k.fit(X_train_30k, y_train)
pred_30k = model_30k.predict(X_test_30k)

acc_30k = accuracy_score(y_test, pred_30k)
precision_30k, recall_30k, f1_30k, _ = precision_recall_fscore_support(
    y_test,
    pred_30k,
    average="weighted",
    zero_division=0,
)

print(f"Accuracy:  {acc_30k:.4f}")
print(f"Precision: {precision_30k:.4f}")
print(f"Recall:    {recall_30k:.4f}")
print(f"F1 Score:  {f1_30k:.4f}")

# --------------------------------------------------
# Balanced Logistic Regression on 30K features
# --------------------------------------------------

print("\n" + "=" * 60)
print("30K TF-IDF + BALANCED LOGISTIC REGRESSION")
print("=" * 60)

model_30k_balanced = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
model_30k_balanced.fit(X_train_30k, y_train)
pred_30k_balanced = model_30k_balanced.predict(X_test_30k)

acc_balanced = accuracy_score(y_test, pred_30k_balanced)
precision_balanced, recall_balanced, f1_balanced, _ = precision_recall_fscore_support(
    y_test,
    pred_30k_balanced,
    average="weighted",
    zero_division=0,
)

print(f"Accuracy:  {acc_balanced:.4f}")
print(f"Precision: {precision_balanced:.4f}")
print(f"Recall:    {recall_balanced:.4f}")
print(f"F1 Score:  {f1_balanced:.4f}")

# --------------------------------------------------
# Comparison
# --------------------------------------------------

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\n20K TF-IDF F1: {f1_20k:.4f}")
print(f"30K TF-IDF F1: {f1_30k:.4f}")
print(f"30K + Balanced F1: {f1_balanced:.4f}")

if f1_balanced > f1_30k:
    print("\nBalanced weighting performed better than plain 30K logistic regression.")
elif f1_balanced < f1_30k:
    print("\nPlain 30K logistic regression performed better than balanced weighting.")
else:
    print("\nBalanced and unweighted 30K logistic regression performed equally.")
