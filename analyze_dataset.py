import gzip
import json
from collections import Counter

file_path = "data/raw/Video_Games.jsonl.gz"

total_reviews = 0
rating_counts = Counter()
missing_text = 0
missing_title = 0
verified_counts = Counter()

with gzip.open(file_path, "rt", encoding="utf-8") as file:

    for line in file:
        review = json.loads(line)

        total_reviews += 1

        # Count ratings
        rating_counts[review.get("rating")] += 1

        # Check missing text
        if not review.get("text"):
            missing_text += 1

        # Check missing title
        if not review.get("title"):
            missing_title += 1

        # Counts the verified purchases
        verified_counts[review.get("verified_purchase")] += 1

print("=" * 50)
print("AMAZON VIDEO GAMES DATASET ANALYSIS")
print("=" * 50)

print(f"\nTotal reviews: {total_reviews:,}")

print("\nRating distribution:")
for rating in sorted(rating_counts):
    count = rating_counts[rating]
    percentage = (count / total_reviews) * 100
    print(f"{rating} stars: {count:,} ({percentage:.2f}%)")

print(f"\nMissing review text: {missing_text:,}")
print(f"Missing review titles: {missing_title:,}")

print("\nVerified purchases:")
for status, count in verified_counts.items():
    percentage = (count / total_reviews) * 100
    print(f"{status}: {count:,} ({percentage:.2f}%)")