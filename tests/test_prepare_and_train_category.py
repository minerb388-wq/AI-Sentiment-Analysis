import csv
import json

from prepare_and_train_category import clean_text, prepare_dataset, sentiment_for_rating


class IdentityLemmatizer:
    def lemmatize(self, word):
        return word


def test_sentiment_for_rating_maps_star_ratings_to_three_classes():
    assert sentiment_for_rating(1.0) == "negative"
    assert sentiment_for_rating(2.0) == "negative"
    assert sentiment_for_rating(3.0) == "neutral"
    assert sentiment_for_rating(4.0) == "positive"
    assert sentiment_for_rating(5.0) == "positive"
    assert sentiment_for_rating(0.0) is None
    assert sentiment_for_rating(None) is None


def test_clean_text_removes_noise_but_preserves_negation():
    cleaned = clean_text(
        "<b>This</b> is NOT great! Visit https://example.com now.",
        stop_words={"this", "is", "great", "visit", "now"},
        lemmatizer=IdentityLemmatizer(),
    )

    assert cleaned == "not"


def test_prepare_dataset_creates_balanced_sample(tmp_path):
    raw_path = tmp_path / "reviews.jsonl"
    output_path = tmp_path / "processed" / "sample.csv"
    reviews = [
        {"rating": rating, "text": f"review {rating}-{index}", "title": "title"}
        for rating in (1.0, 2.0, 3.0, 4.0, 5.0)
        for index in range(3)
    ]
    raw_path.write_text(
        "\n".join(json.dumps(review) for review in reviews),
        encoding="utf-8",
    )

    prepare_dataset(raw_path, output_path, target_size=6)

    with output_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 6
    labels = [row["sentiment"] for row in rows]
    assert labels.count("negative") == 2
    assert labels.count("neutral") == 2
    assert labels.count("positive") == 2
