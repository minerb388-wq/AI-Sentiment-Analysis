"""Evaluate sentiment classifiers on the cleaned category datasets.

The script uses one fixed, stratified 80/20 split per category. TF-IDF is fit
only on the training partition, so the reported test metrics are free from
feature-extraction leakage.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


ROOT = Path(__file__).resolve().parent
CATEGORY_DATASETS = {
    "Video Games": ROOT / "data" / "processed" / "video_games_sentiment_clean.csv",
    "All Beauty": ROOT / "data" / "processed" / "all_beauty_sentiment_50k.csv",
}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def metric_row(category, model_name, y_true, predictions, train_size, test_size):
    """Return comparable weighted classification metrics for one model."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "Category": category,
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Train Samples": train_size,
        "Test Samples": test_size,
    }


def evaluate_category(category, dataset_path):
    """Evaluate baseline, Naive Bayes, and Logistic Regression for a category."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    required_columns = {"clean_text", "sentiment"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_path.name} is missing required columns: {sorted(missing_columns)}"
        )

    dataset = dataset.dropna(subset=["clean_text", "sentiment"])
    dataset = dataset[dataset["clean_text"].str.strip() != ""]
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["clean_text"],
        dataset["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset["sentiment"],
    )

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)
        rows.append(
            metric_row(
                category,
                model_name,
                y_test,
                predictions,
                train_size=len(X_train),
                test_size=len(X_test),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare sentiment classifiers on the cleaned project datasets."
    )
    parser.add_argument(
        "--category",
        choices=tuple(CATEGORY_DATASETS),
        action="append",
        help="Category to evaluate. Omit to evaluate every category.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for generated CSV and JSON summaries.",
    )
    args = parser.parse_args()

    categories = args.category or list(CATEGORY_DATASETS)
    rows = []
    for category in categories:
        print(f"Evaluating {category}...")
        rows.extend(evaluate_category(category, CATEGORY_DATASETS[category]))

    results = pd.DataFrame(rows)
    results["Model"] = pd.Categorical(
        results["Model"],
        categories=["Majority baseline", "Multinomial Naive Bayes", "Logistic Regression"],
        ordered=True,
    )
    results = results.sort_values(["Category", "Model"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "model_comparison.csv"
    json_path = args.output_dir / "model_comparison.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(results.to_json(orient="records", indent=2), encoding="utf-8")

    print("\nModel comparison")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
