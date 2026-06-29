import pytest

from backend.services.financial.financial_model import calculate_fundamental_score


@pytest.mark.parametrize(
    ("probabilities", "expected_raw", "expected_score"),
    [
        (
            {"negative": 1.0, "neutral": 0.0, "positive": 0.0},
            -1.0,
            1.0,
        ),
        (
            {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            0.0,
            5.0,
        ),
        (
            {"negative": 0.0, "neutral": 0.0, "positive": 1.0},
            1.0,
            10.0,
        ),
        (
            {"negative": 0.20, "neutral": 0.30, "positive": 0.50},
            0.30,
            6.50,
        ),
        (
            {"negative": 0.60, "neutral": 0.30, "positive": 0.10},
            -0.50,
            3.0,
        ),
    ],
)
def test_calculate_fundamental_score(
    probabilities,
    expected_raw,
    expected_score,
):
    raw_outlook, fundamental_score = calculate_fundamental_score(probabilities)

    assert raw_outlook == expected_raw
    assert fundamental_score == expected_score
