from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.financial.financial_repository import save_financial_prediction


@patch("backend.services.financial.financial_repository.supabase")
def test_save_financial_prediction_requires_persisted_score_fields(mock_supabase):
    table = MagicMock()
    mock_supabase.table.return_value = table
    table.upsert.return_value.execute.return_value = SimpleNamespace(
        data=[{
            "ticker": "AAPL",
            "probabilities": None,
            "raw_outlook": None,
            "fundamental_score": None,
        }]
    )

    with pytest.raises(RuntimeError, match="without saved score fields"):
        save_financial_prediction({
            "ticker": "AAPL",
            "probabilities": {
                "negative": 0.2,
                "neutral": 0.5,
                "positive": 0.3,
            },
            "raw_outlook": 0.1,
            "fundamental_score": 5.5,
        })


@patch("backend.services.financial.financial_repository.supabase")
def test_save_financial_prediction_returns_confirmed_stored_row(mock_supabase):
    table = MagicMock()
    mock_supabase.table.return_value = table
    stored = {
        "ticker": "AAPL",
        "probabilities": {
            "negative": 0.2,
            "neutral": 0.5,
            "positive": 0.3,
        },
        "raw_outlook": 0.1,
        "fundamental_score": 5.5,
    }
    table.upsert.return_value.execute.return_value = SimpleNamespace(data=[stored])

    assert save_financial_prediction(stored) == stored
