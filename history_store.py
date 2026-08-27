"""SQLite persistence helpers for dashboard analysis history and feedback."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pandas as pd


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "app_history.db"
VALID_SENTIMENTS = {"negative", "neutral", "positive"}


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    review_text TEXT NOT NULL,
    predicted_sentiment TEXT NOT NULL
        CHECK (predicted_sentiment IN ('negative', 'neutral', 'positive')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL UNIQUE,
    prediction_correct INTEGER NOT NULL CHECK (prediction_correct IN (0, 1)),
    corrected_sentiment TEXT
        CHECK (corrected_sentiment IS NULL OR corrected_sentiment IN ('negative', 'neutral', 'positive')),
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
"""


@contextmanager
def _connection(database_path=None):
    """Open a short-lived SQLite connection with foreign keys enabled."""
    path = Path(database_path) if database_path is not None else DEFAULT_DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path=None):
    """Create the analysis and feedback tables when they do not yet exist."""
    with _connection(database_path) as connection:
        connection.executescript(SCHEMA)


def _required_text(value, field_name, maximum_length):
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    if len(text) > maximum_length:
        raise ValueError(f"{field_name} must be {maximum_length} characters or fewer.")
    return text


def _sentiment(value, field_name):
    sentiment = _required_text(value, field_name, maximum_length=20).lower()
    if sentiment not in VALID_SENTIMENTS:
        allowed = ", ".join(sorted(VALID_SENTIMENTS))
        raise ValueError(f"{field_name} must be one of: {allowed}.")
    return sentiment


def record_analysis(category, review_text, predicted_sentiment, confidence, database_path=None):
    """Store one prediction and return its database identifier."""
    category = _required_text(category, "Category", maximum_length=100)
    review_text = _required_text(review_text, "Review text", maximum_length=3000)
    predicted_sentiment = _sentiment(predicted_sentiment, "Predicted sentiment")

    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError("Confidence must be a number between 0 and 1.") from error
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be a number between 0 and 1.")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                created_at, category, review_text, predicted_sentiment, confidence
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (created_at, category, review_text, predicted_sentiment, confidence),
        )
        return int(cursor.lastrowid)


def save_feedback(
    analysis_id,
    prediction_correct,
    corrected_sentiment=None,
    comment=None,
    database_path=None,
):
    """Create or update one feedback record for a saved prediction."""
    if not isinstance(analysis_id, int) or analysis_id <= 0:
        raise ValueError("Analysis ID must be a positive integer.")
    if not isinstance(prediction_correct, bool):
        raise ValueError("Prediction correctness must be True or False.")

    if prediction_correct:
        corrected_sentiment = None
    else:
        corrected_sentiment = _sentiment(
            corrected_sentiment,
            "Corrected sentiment",
        )

    comment = "" if comment is None else str(comment).strip()
    if len(comment) > 1000:
        raise ValueError("Feedback comment must be 1000 characters or fewer.")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO feedback (
                analysis_id, prediction_correct, corrected_sentiment, comment, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(analysis_id) DO UPDATE SET
                prediction_correct = excluded.prediction_correct,
                corrected_sentiment = excluded.corrected_sentiment,
                comment = excluded.comment,
                created_at = excluded.created_at
            """,
            (
                analysis_id,
                int(prediction_correct),
                corrected_sentiment,
                comment or None,
                created_at,
            ),
        )


def get_analysis_history(limit=100, database_path=None):
    """Return recent analyses and any associated user feedback."""
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("Limit must be an integer between 1 and 1000.")

    query = """
        SELECT
            analyses.id AS analysis_id,
            analyses.created_at,
            analyses.category,
            analyses.review_text,
            analyses.predicted_sentiment,
            analyses.confidence,
            feedback.prediction_correct,
            feedback.corrected_sentiment,
            feedback.comment AS feedback_comment,
            feedback.created_at AS feedback_created_at
        FROM analyses
        LEFT JOIN feedback ON feedback.analysis_id = analyses.id
        ORDER BY analyses.id DESC
        LIMIT ?
    """
    with _connection(database_path) as connection:
        return pd.read_sql_query(query, connection, params=(limit,))


def clear_analysis_history(database_path=None):
    """Delete all saved analyses and cascaded feedback records."""
    with _connection(database_path) as connection:
        cursor = connection.execute("DELETE FROM analyses")
        return cursor.rowcount
