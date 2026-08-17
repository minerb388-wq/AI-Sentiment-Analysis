import re
from pathlib import Path

import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORIZER_PATH = REPO_ROOT / "models" / "tfidf_vectorizer.joblib"
MODEL_PATH = REPO_ROOT / "models" / "logistic_regression_model.joblib"

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    import nltk

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    stop_words = set(stopwords.words("english"))

important_words = {"not", "no", "never", "nor", "neither", "hardly", "barely", "nothing"}
stop_words = stop_words - important_words
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"['’]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def load_model_assets():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


vectorizer, model = load_model_assets()


def predict_sentiment(review):
    cleaned = clean_text(review)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    probability_dict = dict(zip(model.classes_, probabilities))
    return prediction, probability_dict


print("=" * 60)
print("AMAZON VIDEO GAME SENTIMENT ANALYZER")
print("=" * 60)
print("\nType a review to analyze it.")
print("Type 'quit' to exit.\n")

while True:
    review = input("Enter review: ")

    if review.lower() == "quit":
        print("\nExiting...")
        break

    if not review.strip():
        print("Please enter a review.\n")
        continue

    prediction, probabilities = predict_sentiment(review)

    print("\nPrediction:", prediction.upper())
    print("\nConfidence:")

    for sentiment, probability in sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{sentiment.capitalize():10}: {probability * 100:.2f}%")

    print("\n" + "-" * 60 + "\n")
