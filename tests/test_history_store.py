import pytest

from history_store import (
    clear_analysis_history,
    get_analysis_history,
    initialize_database,
    record_analysis,
    save_feedback,
)


def test_history_store_records_and_updates_feedback(tmp_path):
    database_path = tmp_path / "history.db"
    initialize_database(database_path)

    analysis_id = record_analysis(
        "All Beauty",
        "This product is gentle on my skin.",
        "positive",
        0.87,
        database_path,
    )

    history = get_analysis_history(database_path=database_path)
    assert len(history) == 1
    assert history.iloc[0]["analysis_id"] == analysis_id
    assert history.iloc[0]["predicted_sentiment"] == "positive"
    assert history.iloc[0]["prediction_correct"] is None

    save_feedback(
        analysis_id,
        prediction_correct=False,
        corrected_sentiment="neutral",
        comment="The review is mixed rather than fully positive.",
        database_path=database_path,
    )
    save_feedback(
        analysis_id,
        prediction_correct=True,
        comment="Updated after a second review.",
        database_path=database_path,
    )

    history = get_analysis_history(database_path=database_path)
    assert len(history) == 1
    assert history.iloc[0]["prediction_correct"] == 1
    assert history.iloc[0]["corrected_sentiment"] is None
    assert history.iloc[0]["feedback_comment"] == "Updated after a second review."

    assert clear_analysis_history(database_path) == 1
    assert get_analysis_history(database_path=database_path).empty


def test_history_store_rejects_invalid_feedback(tmp_path):
    database_path = tmp_path / "history.db"
    initialize_database(database_path)
    analysis_id = record_analysis(
        "Video Games",
        "The controls are difficult to use.",
        "negative",
        0.62,
        database_path,
    )

    with pytest.raises(ValueError, match="Corrected sentiment"):
        save_feedback(
            analysis_id,
            prediction_correct=False,
            corrected_sentiment=None,
            database_path=database_path,
        )

    with pytest.raises(ValueError, match="Confidence"):
        record_analysis(
            "Video Games",
            "A valid review.",
            "positive",
            1.2,
            database_path,
        )
