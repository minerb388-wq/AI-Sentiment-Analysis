import argparse
import csv
import gzip
import json
import random
from pathlib import Path

import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


IMPORTANT_WORDS = {
    "not",
    "no",
    "never",
    "nor",
    "neither",
    "hardly",
    "barely",
    "nothing",
}


def clean_text(text, stop_words, lemmatizer):
    import re

    text = re.sub(r"<[^>]*>", " ", str(text)).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"['’]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [
        word
        for word in re.sub(r"\s+", " ", text).strip().split()
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(lemmatizer.lemmatize(word) for word in words)


def sentiment_for_rating(rating):
    if rating in (1.0, 2.0):
        return "negative"
    if rating == 3.0:
        return "neutral"
    if rating in (4.0, 5.0):
        return "positive"
    return None


def prepare_dataset(raw_file, output_file, target_size):
    random.seed(42)
    target_per_class = target_size // 3
    remainder = target_size - target_per_class * 3
    targets = {
        label: target_per_class + (1 if index < remainder else 0)
        for index, label in enumerate(("negative", "neutral", "positive"))
    }
    samples = {label: [] for label in targets}
    seen = {label: 0 for label in targets}

    opener = gzip.open if raw_file.suffix == ".gz" else open
    with opener(raw_file, "rt", encoding="utf-8") as file:
        for line in file:
            review = json.loads(line)
            text = str(review.get("text", "")).strip()
            sentiment = sentiment_for_rating(review.get("rating"))
            if not text or sentiment is None:
                continue
            seen[sentiment] += 1
            row = {
                "rating": review.get("rating"),
                "title": review.get("title", ""),
                "text": text,
                "asin": review.get("asin"),
                "timestamp": review.get("timestamp"),
                "helpful_vote": review.get("helpful_vote", 0),
                "verified_purchase": review.get("verified_purchase", False),
                "sentiment": sentiment,
            }

            # Reservoir sampling gives every eligible review an equal chance of
            # inclusion without keeping the entire source dataset in memory.
            if len(samples[sentiment]) < targets[sentiment]:
                samples[sentiment].append(row)
            else:
                index = random.randrange(seen[sentiment])
                if index < targets[sentiment]:
                    samples[sentiment][index] = row

    if any(seen[label] < targets[label] for label in targets):
        raise ValueError(
            f"Not enough reviews for a balanced {target_size:,}-row sample: {seen}"
        )

    rows = []
    for label in targets:
        rows.extend(samples[label])
    random.shuffle(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_model(dataset_file, vectorizer_file, model_file):
    df = pd.read_csv(dataset_file)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    stop_words = set(stopwords.words("english")) - IMPORTANT_WORDS
    lemmatizer = WordNetLemmatizer()
    df["clean_text"] = df["text"].map(lambda text: clean_text(text, stop_words, lemmatizer))
    df = df[df["clean_text"].str.strip() != ""]
    df.to_csv(dataset_file, index=False, encoding="utf-8-sig")

    X_train, _, y_train, _ = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["sentiment"],
    )

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(features, y_train)

    vectorizer_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(model, model_file)


def main():
    parser = argparse.ArgumentParser(description="Prepare and train an Amazon Reviews '23 category.")
    parser.add_argument("--raw", type=Path, required=True, help="Path to category JSONL or JSONL.GZ file")
    parser.add_argument("--name", required=True, help="File-safe category name, e.g. all_beauty")
    parser.add_argument("--size", type=int, default=50_000, help="Balanced sample size")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset = root / "data" / "processed" / f"{args.name}_sentiment_50k.csv"
    vectorizer = root / "models" / f"{args.name}_tfidf_vectorizer.joblib"
    model = root / "models" / f"{args.name}_logistic_regression_model.joblib"

    prepare_dataset(args.raw, dataset, args.size)
    train_model(dataset, vectorizer, model)
    print(f"Prepared dataset: {dataset}")
    print(f"Saved vectorizer: {vectorizer}")
    print(f"Saved model: {model}")


if __name__ == "__main__":
    main()
