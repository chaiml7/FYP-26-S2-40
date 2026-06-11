from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.technical.technical_repository import (
    save_technical_prediction,
)


@patch("backend.services.technical.technical_repository.supabase")
def test_prediction_persistence_requires_score_fields(mock_supabase):
    table = MagicMock()
    mock_supabase.table.return_value = table
    table.upsert.return_value.execute.return_value = SimpleNamespace(
        data=[{
            "symbol": "AAPL",
            "prediction": None,
            "probabilities": None,
            "raw_outlook": None,
            "technical_score": None,
            "model_version": None,
        }]
    )

    with pytest.raises(RuntimeError, match="without fields"):
        save_technical_prediction({"symbol": "AAPL"})


@patch("backend.services.technical.technical_repository.supabase")
def test_prediction_upsert_uses_model_version_key(mock_supabase):
    table = MagicMock()
    mock_supabase.table.return_value = table
    stored = {
        "prediction": "bullish",
        "probabilities": {
            "bearish": 0.1,
            "neutral": 0.2,
            "bullish": 0.7,
        },
        "raw_outlook": 0.6,
        "technical_score": 8.0,
        "model_version": "lightgbm_technical_20260611T000000000000Z",
    }
    table.upsert.return_value.execute.return_value = SimpleNamespace(
        data=[stored]
    )

    assert save_technical_prediction(stored) == stored
    table.upsert.assert_called_once()
    assert (
        table.upsert.call_args.kwargs["on_conflict"]
        == "stock_id,latest_date,model_version"
    )
