from unittest.mock import patch

from backend.services.analysis_pipeline import run_scheduled_analysis_pipeline


MODULE = "backend.services.analysis_pipeline"


def test_scheduled_pipeline_refreshes_active_data_before_predictions(monkeypatch):
    calls = []
    monkeypatch.setenv("TECHNICAL_IMPORT_PERIOD", "1y")

    def import_prices(**kwargs):
        calls.append(("prices", kwargs))
        return {"results": []}

    def sentiment(**kwargs):
        calls.append(("sentiment", kwargs))
        return {"results": []}

    def predictions():
        calls.append(("predictions", {}))
        return {"stocks_failed": 0}

    with (
        patch(f"{MODULE}.import_all_technical_prices", side_effect=import_prices),
        patch(f"{MODULE}.run_pipeline", side_effect=sentiment),
        patch(f"{MODULE}.generate_all_technical_predictions", side_effect=predictions),
    ):
        result = run_scheduled_analysis_pipeline()

    assert calls == [
        ("prices", {"period": "1y", "stock_scope": "active"}),
        ("sentiment", {"refresh_existing": True}),
        ("predictions", {}),
    ]
    assert result["status"] == "ok"


def test_scheduled_pipeline_isolates_stage_failures():
    with (
        patch(f"{MODULE}.import_all_technical_prices", side_effect=RuntimeError("prices down")),
        patch(f"{MODULE}.run_pipeline", return_value={"results": []}),
        patch(
            f"{MODULE}.generate_all_technical_predictions",
            return_value={"stocks_failed": 0},
        ),
    ):
        result = run_scheduled_analysis_pipeline()

    assert result["status"] == "partial"
    assert result["stages"]["technical_price_import"]["status"] == "error"
    assert result["stages"]["sentiment_analysis"]["status"] == "ok"
    assert result["stages"]["technical_predictions"]["status"] == "ok"
