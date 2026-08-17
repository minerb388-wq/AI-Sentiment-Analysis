import csv
import gzip
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = REPO_ROOT / "data" / "raw" / "Video_Games.jsonl.gz"
OUTPUT_FILE = REPO_ROOT / "data" / "processed" / "video_games_sentiment_50k.csv"
TARGET_PER_CLASS = 16666


def main() -> None:
    random.seed(42)
    negative = []
    neutral = []
    positive = []

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(INPUT_FILE, "rt", encoding="utf-8") as file:
        for line in file:
            review = json.loads(line)
            rating = review.get("rating")
            text = review.get("text", "").strip()
            title = review.get("title", "").strip()

            if not text:
                continue

            if rating in [1.0, 2.0]:
                sentiment = "negative"
                collection = negative
            elif rating == 3.0:
                sentiment = "neutral"
                collection = neutral
            elif rating in [4.0, 5.0]:
                sentiment = "positive"
                collection = positive
            else:
                continue

            collection.append(
                {
                    "rating": rating,
                    "title": title,
                    "text": text,
                    "asin": review.get("asin"),
                    "timestamp": review.get("timestamp"),
                    "helpful_vote": review.get("helpful_vote", 0),
                    "verified_purchase": review.get("verified_purchase", False),
                    "sentiment": sentiment,
                }
            )

    print("Original class sizes:")
    print(f"Negative: {len(negative):,}")
    print(f"Neutral:  {len(neutral):,}")
    print(f"Positive: {len(positive):,}")

    negative_sample = random.sample(negative, TARGET_PER_CLASS)
    neutral_sample = random.sample(neutral, TARGET_PER_CLASS)
    positive_sample = random.sample(positive, 50000 - (TARGET_PER_CLASS * 2))

    dataset = negative_sample + neutral_sample + positive_sample
    random.shuffle(dataset)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "rating",
            "title",
            "text",
            "asin",
            "timestamp",
            "helpful_vote",
            "verified_purchase",
            "sentiment",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print("\nDataset created successfully!")
    print(f"Total reviews: {len(dataset):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
