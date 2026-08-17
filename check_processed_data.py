from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
FILE_PATH = REPO_ROOT / "data" / "processed" / "video_games_sentiment_50k.csv"

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("PROCESSED DATASET CHECK")
print("=" * 60)

# Basic information
print(f"\nNumber of reviews: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print("-", column)

# Sentiment distribution
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSentiment percentages:")
print(
    (df["sentiment"].value_counts(normalize=True) * 100)
    .round(2)
)

# Rating distribution
print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print(f"\nDuplicate rows: {df.duplicated().sum():,}")

# Review length
df["text_length"] = df["text"].astype(str).str.len()

print("\nReview text length:")
print(df["text_length"].describe())

# Show examples
print("\n" + "=" * 60)
print("SAMPLE REVIEWS")
print("=" * 60)

print(
    df[["rating", "sentiment", "title", "text"]]
    .sample(5, random_state=42)
    .to_string(index=False)
)
