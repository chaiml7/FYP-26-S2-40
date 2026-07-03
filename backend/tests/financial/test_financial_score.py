import pytest

from backend.services.financial.financial_model import (
    _binary_deployment_assessment,
    _binary_selection_score,
    _rank_binary_metrics,
    _rank_binary_tuning_result,
    calculate_fundamental_score,
    classify_binary_display_label,
)


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


def test_binary_metric_ranking_prioritizes_macro_f1():
    higher_f1 = {
        "macro_f1": 0.56,
        "balanced_accuracy": 0.55,
        "accuracy": 0.55,
        "log_loss": 0.69,
    }
    higher_accuracy = {
        "macro_f1": 0.52,
        "balanced_accuracy": 0.53,
        "accuracy": 0.65,
        "log_loss": 0.66,
    }

    assert _rank_binary_metrics(higher_f1) > _rank_binary_metrics(higher_accuracy)


def test_binary_tuning_ranking_does_not_use_final_holdout():
    validation = {
        "average_metrics": {
            "macro_f1": 0.58,
            "balanced_accuracy": 0.59,
            "accuracy": 0.62,
            "log_loss": 0.66,
        }
    }
    strong_holdout = {
        "rolling_validation": validation,
        "latest_holdout_metrics": {"accuracy": 0.9},
    }
    weak_holdout = {
        "rolling_validation": validation,
        "latest_holdout_metrics": {"accuracy": 0.3},
    }

    assert _rank_binary_tuning_result(strong_holdout) == (
        _rank_binary_tuning_result(weak_holdout)
    )


def test_selection_score_rewards_probability_calibration():
    marginally_higher_f1 = {
        "macro_f1": 0.6319,
        "balanced_accuracy": 0.6340,
        "accuracy": 0.7374,
        "log_loss": 0.6802,
    }
    better_calibrated = {
        "macro_f1": 0.6234,
        "balanced_accuracy": 0.6275,
        "accuracy": 0.7273,
        "log_loss": 0.5977,
    }

    assert _binary_selection_score(better_calibrated) > (
        _binary_selection_score(marginally_higher_f1)
    )


def test_deployment_requires_all_active_baselines_to_be_beaten():
    eligible = _binary_deployment_assessment({
        "balanced_accuracy": 0.52,
        "macro_f1": 0.53,
        "log_loss": 0.68,
    })
    rejected = _binary_deployment_assessment({
        "balanced_accuracy": 0.52,
        "macro_f1": 0.53,
        "log_loss": 0.72,
    })

    assert eligible["eligible"] is True
    assert rejected["eligible"] is False
    assert rejected["checks"]["log_loss"] is False


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.44, "negative"), (0.5, "neutral"), (0.56, "positive")],
)
def test_binary_display_labels(probability, expected):
    assert classify_binary_display_label(probability) == expected
