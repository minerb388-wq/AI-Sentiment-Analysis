import pandas as pd
import re
import sys

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


INPUT_FILE = "data/processed/video_games_sentiment_50k.csv"
OUTPUT_FILE = "data/processed/video_games_sentiment_clean.csv"


# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset:")
print(f"Rows: {len(df):,}")


# --------------------------------------------------
# 1. Remove missing review text
# --------------------------------------------------

df = df.dropna(subset=["text"])

print(f"Rows after dropping missing text: {len(df):,}")


# --------------------------------------------------
# 2. Remove duplicate reviews
# --------------------------------------------------

df = df.drop_duplicates(subset=["text"])

print(f"Rows after dropping duplicates: {len(df):,}")


# --------------------------------------------------
# 3. NLP resources
# --------------------------------------------------

stop_words = set(stopwords.words("english"))

# Keep important sentiment words
important_words = {
    "not",
    "no",
    "never",
    "nor",
    "neither",
    "hardly",
    "barely",
    "nothing"
}

stop_words = stop_words - important_words

lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# 4. Text cleaning function
# --------------------------------------------------

def clean_text(text):

    text = str(text)

    # Replace ALL HTML tags with a space
    # This prevents words from being joined together
    text = re.sub(r"<[^>]*>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Replace URLs with a space
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Replace apostrophes with a space
    text = re.sub(r"['’]", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords but keep sentiment-important words
    words = [
        word
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# --------------------------------------------------
# 5. Clean reviews
# --------------------------------------------------

print("\nCleaning review text...")

df["clean_text"] = df["text"].apply(clean_text)


# --------------------------------------------------
# 6. Remove empty cleaned reviews
# --------------------------------------------------

before_empty = len(df)

df = df[df["clean_text"].str.strip() != ""]

removed_empty = before_empty - len(df)

print(
    f"Rows after removing empty cleaned reviews: "
    f"{len(df):,} ({removed_empty:,} removed)"
)


# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\nPreprocessing complete!")
print(f"Final rows: {len(df):,}")
print(f"Saved to: {OUTPUT_FILE}")


# --------------------------------------------------
# 8. Display examples
# --------------------------------------------------

print("\nExamples:")

print(
    df[["text", "clean_text", "sentiment"]]
    .head(5)
    .to_string(index=False)
)
