import os
import re

import joblib
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split




st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4 {
            color: #f8fafc;
        }
        .page-header {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.3);
        }
        .page-header h1 {
            margin-bottom: 0.2rem;
        }
        .page-header p {
            margin: 0;
            color: #cbd5e1;
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.9);
        }
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, #22c55e 0%, #14b8a6 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: 600;
        }
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        .stProgress > div > div {
            background: linear-gradient(90deg, #14b8a6, #22c55e);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

CATEGORY_CONFIG = {
    "Video Games": {
        "dataset": "data/processed/video_games_sentiment_clean.csv",
        "vectorizer": "models/tfidf_vectorizer.joblib",
        "model": "models/logistic_regression_model.joblib",
    },
    "All Beauty": {
        "dataset": "data/processed/all_beauty_sentiment_50k.csv",
        "vectorizer": "models/all_beauty_tfidf_vectorizer.joblib",
        "model": "models/all_beauty_logistic_regression_model.joblib",
    },
}


def category_model_available(category):
    base_dir = os.path.dirname(__file__)
    config = CATEGORY_CONFIG[category]
    return all(
        os.path.exists(os.path.join(base_dir, config[key]))
        for key in ("vectorizer", "model")
    )


@st.cache_resource
def load_model(category):
    base_dir = os.path.dirname(__file__)
    config = CATEGORY_CONFIG[category]
    vectorizer = joblib.load(os.path.join(base_dir, config["vectorizer"]))
    model = joblib.load(os.path.join(base_dir, config["model"]))
    return vectorizer, model


@st.cache_data
def load_dataset(category):
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, CATEGORY_CONFIG[category]["dataset"])

    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)

    st.warning(f"{category} dataset is not available yet.")
    return None


@st.cache_data
def load_model_performance(category):
    df = load_dataset(category)
    if df is None:
        return None
    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer, model = load_model(category)
    X_test_tfidf = vectorizer.transform(X_test)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )
    class_report_df = pd.DataFrame(report).T
    class_report_df = class_report_df.loc[
        ["negative", "neutral", "positive"],
        ["precision", "recall", "f1-score", "support"],
    ]

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"],
    )
    confusion_df = pd.DataFrame(
        confusion,
        index=["Actual Negative", "Actual Neutral", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"],
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_report": class_report_df,
        "confusion_matrix": confusion_df,
    }


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


def predict_sentiment(review):
    vectorizer, model = load_model(selected_category)
    cleaned = clean_text(review)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    probability_dict = dict(zip(model.classes_, probabilities))
    return prediction, probability_dict


st.markdown(
    """
    <div class="page-header">
        <h1>Amazon Review Sentiment Analyzer</h1>
        <p>Machine learning dashboard for analyzing Amazon product-review sentiment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["Review Analyzer", "Dataset Dashboard", "Model Performance"],
)

selected_category = st.sidebar.selectbox(
    "Review category",
    list(CATEGORY_CONFIG),
)

if page == "Review Analyzer":
    st.subheader("Review Analyzer")
    st.caption(f"Selected category: {selected_category}")
    review = st.text_area(
        "Enter a review",
        value="I absolutely love this game. The graphics are amazing and the gameplay is fantastic.",
        height=160,
    )

    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review before analyzing.")
        elif not category_model_available(selected_category):
            st.error(
                f"The {selected_category} model is not available yet. "
                "Prepare and train this category before analyzing reviews."
            )
        else:
            prediction, probabilities = predict_sentiment(review)
            probability_df = pd.DataFrame(
                {
                    "Sentiment": ["Negative", "Neutral", "Positive"],
                    "Probability": [
                        probabilities.get("negative", 0.0),
                        probabilities.get("neutral", 0.0),
                        probabilities.get("positive", 0.0),
                    ],
                }
            )

            prediction_label = prediction.upper()
            color = {
                "POSITIVE": "#2ecc71",
                "NEUTRAL": "#f39c12",
                "NEGATIVE": "#e74c3c",
            }.get(prediction_label, "#3498db")

            st.subheader("Prediction")
            st.markdown(
                f"<div style='background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 1rem;'><h2 style='color:{color}; margin:0;'> {prediction_label} </h2></div>",
                unsafe_allow_html=True,
            )

            st.write("Confidence by class:")
            st.bar_chart(
                probability_df.set_index("Sentiment")["Probability"] * 100,
                use_container_width=True,
            )

            st.write("Probability breakdown:")
            for sentiment in ["Positive", "Neutral", "Negative"]:
                key = sentiment.lower()
                prob = probabilities.get(key, 0.0) * 100
                st.progress(min(100, max(0, prob / 100)))
                st.write(f"{sentiment}: {prob:.2f}%")

elif page == "Dataset Dashboard":
    st.subheader("Dataset Dashboard")
    st.caption(f"Summary of the {selected_category} review dataset.")

    df = load_dataset(selected_category)

    if df is None:
        st.error("Dataset could not be loaded. Please check the logs.")
    else:
        total_reviews = len(df)
        sentiment_counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        sentiment_percentages = (sentiment_counts / total_reviews * 100).round(1)
        rating_counts = (
            df["rating"]
            .value_counts()
            .sort_index()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
        )
        rating_labels = [f"{int(rating)}★" for rating in rating_counts.index]
        rating_chart = pd.Series(rating_counts.values, index=rating_labels)

        verified_counts = df["verified_purchase"].value_counts().reindex([True, False], fill_value=0)
        verified_labels = verified_counts.index.map(lambda value: "Verified" if value else "Not Verified")
        verified_chart = pd.Series(verified_counts.values, index=verified_labels)
        sentiment_rating = pd.crosstab(df["rating"], df["sentiment"])

        if "text_length" in df.columns:
            review_lengths = df["text_length"]
        else:
            review_lengths = df["text"].fillna("").astype(str).str.len()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", f"{total_reviews:,}")
        col2.metric("Positive", f"{sentiment_counts.get('positive', 0):,} ({sentiment_percentages.get('positive', 0):.1f}%)")
        col3.metric("Negative", f"{sentiment_counts.get('negative', 0):,} ({sentiment_percentages.get('negative', 0):.1f}%)")

        st.write("")
        col1, col2, col3 = st.columns(3)
        col1.metric("Neutral", f"{sentiment_counts.get('neutral', 0):,} ({sentiment_percentages.get('neutral', 0):.1f}%)")
        col2.metric("Avg Review Length", f"{review_lengths.mean():.1f} chars")
        col3.metric("Median Rating", f"{df['rating'].median():.1f}/5")

        st.write("### Sentiment Distribution")
        sentiment_chart = sentiment_counts.rename(index={"positive": "Positive", "neutral": "Neutral", "negative": "Negative"})
        st.bar_chart(sentiment_chart)

        st.write("### Rating Distribution")
        st.bar_chart(rating_chart)

        st.write("### Verified Purchase Distribution")
        st.bar_chart(verified_chart)

        st.write("### Sentiment by Star Rating")
        st.bar_chart(sentiment_rating)

        st.write("### Review Length Summary")
        st.write(review_lengths.describe())

elif page == "Model Performance":
    st.subheader("Model Performance")
    st.caption(f"Evaluation of the {selected_category} 30K TF-IDF + Logistic Regression model.")

    model_metrics = load_model_performance(selected_category)

    if model_metrics is None:
        st.error("Model metrics could not be loaded. Please check the logs.")
    else:
        metrics = model_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        col4.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%")

        st.write("### Classification Report")
        st.dataframe(metrics["class_report"], use_container_width=True)

        st.write("### Confusion Matrix")
        st.dataframe(metrics["confusion_matrix"], use_container_width=True)
        st.caption(
            "The confusion matrix shows how many reviews were correctly or incorrectly classified for each sentiment class."
        )

        st.write("### Classifier Comparison")
        comparison = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Naive Bayes", "Linear SVM"],
                "Accuracy": [0.6834, 0.6660, 0.6578],
                "Precision": [0.6844, 0.6723, 0.6552],
                "Recall": [0.6834, 0.6660, 0.6578],
                "F1 Score": [0.6837, 0.6678, 0.6562],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]])

st.sidebar.divider()
st.sidebar.subheader("Model")
st.sidebar.write(
    """**Algorithm:** Logistic Regression
**Features:** 30,000 TF-IDF
**N-grams:** 1–2
**Accuracy:** 68.34%
**F1 Score:** 68.37%"""
)
