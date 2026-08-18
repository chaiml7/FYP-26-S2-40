from datetime import date
from unittest.mock import patch

from backend.services.admin_report_service import (
    _build_summary_cards,
    _build_prediction_summary,
    _build_watchlist_summary,
    _component_from_freshness,
    _filter_rows_by_date,
    _freshness_row,
    _model_row,
    _version_row,
    build_model_accuracy_log,
    render_report_csv,
    render_report_pdf,
)

MODULE = "backend.services.admin_report_service"


def test_version_row_extracts_metrics_and_active_flag():
    row = _version_row(
        {
            "model_version": "tech_v3",
            "trained_at": "2026-08-01T00:00:00Z",
            "is_active": True,
            "test_metrics": {
                "accuracy": 0.72,
                "balanced_accuracy": 0.70,
                "macro_f1": 0.69,
                "log_loss": 0.5,
            },
        },
        ["test_metrics", "metrics"],
    )

    assert row["version"] == "tech_v3"
    assert row["trained_at"] == "2026-08-01"
    assert row["is_active"] is True
    assert row["accuracy"] == "72.0%"
    assert row["balanced_accuracy"] == "70.0%"
    assert row["macro_f1"] == "69.0%"
    assert row["log_loss"] == "0.5000"


def test_version_row_handles_missing_metrics():
    row = _version_row(
        {"model_version": "fin_v1", "created_at": "2026-07-01T00:00:00Z", "is_active": False},
        ["metrics", "test_metrics"],
    )

    assert row["version"] == "fin_v1"
    assert row["is_active"] is False
    assert row["accuracy"] == "N/A"


def test_watchlist_summary_ranks_current_favourites():
    watchlists = [
        {"user_id": "u1", "stock_id": 1},
        {"user_id": "u2", "stock_id": 1},
        {"user_id": "u1", "stock_id": 2},
    ]
    stocks = {
        "1": {"symbol": "AAPL", "company_name": "Apple Inc."},
        "2": {"symbol": "MSFT", "company_name": "Microsoft"},
    }

    result = _build_watchlist_summary(watchlists, stocks)

    assert result["total_entries"] == 3
    assert result["users_with_watchlists"] == 2
    assert result["average_per_user"] == "1.50"
    assert result["top_stocks"][0] == {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "count": 2,
        "percent": "66.7%",
    }


def test_prediction_summary_normalizes_prediction_labels():
    result = _build_prediction_summary(
        technical_predictions=[
            {"prediction": "bullish", "technical_score": 7, "latest_date": "2026-08-01"},
            {"prediction": "bearish", "technical_score": 3, "latest_date": "2026-08-02"},
        ],
        financial_predictions=[
            {"prediction": "neutral", "fundamental_score": 5, "period": "2026-07-01"},
        ],
        combined_predictions=[
            {"action": "BUY", "confidence_score": 0.8, "created_at": "2026-08-03T00:00:00Z"},
            {"action": "HOLD", "confidence_score": 0.6, "created_at": "2026-08-04T00:00:00Z"},
        ],
    )

    technical = result["rows"][0]
    combined = result["rows"][2]

    assert technical["bullish"] == 1
    assert technical["bearish"] == 1
    assert technical["average_score"] == "5.00"
    assert technical["latest_date"] == "2026-08-02"
    assert combined["bullish"] == 1
    assert combined["neutral"] == 1


def test_freshness_row_reports_missing_and_stale_symbols():
    result = _freshness_row(
        "Market prices",
        ["AAPL", "MSFT", "NVDA"],
        {
            "AAPL": date(2026, 8, 1),
            "MSFT": date(2026, 8, 9),
        },
        today=date(2026, 8, 10),
    )

    assert result["latest_date"] == "2026-08-09"
    assert result["covered_count"] == 2
    assert result["missing_count"] == 1
    assert result["stale_count"] == 1
    assert result["missing_examples"] == "NVDA"
    assert result["stale_examples"] == "AAPL"


def test_component_from_freshness_is_healthy_when_no_gaps():
    rows = [
        {"name": "Market prices", "latest_date": "2026-08-10", "stale_count": 0, "missing_count": 0},
        {"name": "Technical indicators", "latest_date": "2026-08-09", "stale_count": 0, "missing_count": 0},
    ]

    result = _component_from_freshness(
        "Data Ingestion Pipeline", rows, ["Market prices", "Technical indicators"]
    )

    assert result["status"] == "Healthy"
    assert result["last_successful_run"] == "2026-08-10"


def test_component_from_freshness_is_degraded_when_stale():
    rows = [
        {"name": "Market prices", "latest_date": "2026-08-10", "stale_count": 0, "missing_count": 0},
        {"name": "Technical indicators", "latest_date": "2026-08-01", "stale_count": 3, "missing_count": 0},
    ]

    result = _component_from_freshness(
        "Data Ingestion Pipeline", rows, ["Market prices", "Technical indicators"]
    )

    assert result["status"] == "Degraded"
    assert result["last_successful_run"] == "2026-08-10"


@patch(f"{MODULE}._fetch_rows")
def test_build_model_accuracy_log_filters_by_version_and_date(mock_fetch_rows):
    def fake_fetch_rows(table, errors, limit=20, order_by=None):
        if table == "technical_model_versions":
            return [
                {"model_version": "tech_v1", "trained_at": "2026-07-01T00:00:00Z", "is_active": False},
                {"model_version": "tech_v2", "trained_at": "2026-08-01T00:00:00Z", "is_active": True},
            ]
        return []

    mock_fetch_rows.side_effect = fake_fetch_rows

    result = build_model_accuracy_log(
        model_version="tech_v2", date_from="2026-07-15", date_to="2026-08-15"
    )

    assert len(result["technical"]) == 1
    assert result["technical"][0]["version"] == "tech_v2"


def test_filter_rows_by_date_scopes_to_range():
    rows = [
        {"created_at": "2026-08-05T00:00:00Z"},
        {"created_at": "2026-07-01T00:00:00Z"},
        {"created_at": "2026-08-20T00:00:00Z"},
    ]

    result = _filter_rows_by_date(rows, "created_at", "2026-08-01", "2026-08-10")

    assert result == [{"created_at": "2026-08-05T00:00:00Z"}]


def test_filter_rows_by_date_returns_all_rows_when_no_range_given():
    rows = [{"created_at": "2026-08-05T00:00:00Z"}]

    assert _filter_rows_by_date(rows, "created_at", None, None) == rows


def _sample_report():
    return {
        "generated_at": "2026-08-18 10:00 SGT",
        "generated_by": "admin@example.com",
        "date_from": "2026-08-01",
        "date_to": "2026-08-15",
        "row_limit": 1000,
        "summary_cards": [{"label": "Total Users", "value": "10", "detail": "2 premium"}],
        "prediction_summary": {"rows": [{
            "name": "Technical", "total": 5, "bullish": 3, "neutral": 1,
            "bearish": 1, "unknown": 0, "average_score": "0.62", "latest_date": "2026-08-14",
        }]},
        "freshness_summary": {"rows": [{
            "name": "Market prices", "latest_date": "2026-08-15",
            "covered_count": 10, "missing_count": 0, "stale_count": 0,
        }]},
        "weightage_summary": {"technical": "40%", "sentiment": "30%", "financial": "30%"},
    }


def test_render_report_csv_includes_key_sections():
    csv_text = render_report_csv(_sample_report())

    assert "Total Users" in csv_text
    assert "Technical" in csv_text
    assert "Market prices" in csv_text


def test_render_report_pdf_returns_valid_pdf_bytes():
    pdf_bytes = render_report_pdf(_sample_report())

    assert pdf_bytes[:4] == b"%PDF"


def test_model_row_formats_stored_metrics():
    result = _model_row(
        "Technical",
        {
            "model_version": "technical_v2",
            "trained_at": "2026-08-01T00:00:00Z",
            "evaluation_mode": "holdout",
            "test_metrics": {
                "accuracy": 0.723,
                "balanced_accuracy": 0.701,
                "macro_f1": 0.694,
                "log_loss": 0.51234,
            },
        },
        ["test_metrics"],
    )

    details = {detail["label"]: detail["value"] for detail in result["details"]}

    assert details["Version"] == "technical_v2"
    assert details["Trained"] == "2026-08-01"
    assert details["Accuracy"] == "72.3%"
    assert details["Balanced Accuracy"] == "70.1%"
    assert details["Macro F1"] == "69.4%"
    assert details["Log Loss"] == "0.5123"


def test_model_row_skips_unavailable_metrics():
    result = _model_row(
        "Financial",
        {
            "model_version": "financial_v1",
            "metrics": {"accuracy": 0.6},
        },
        ["metrics"],
    )

    details = {detail["label"]: detail["value"] for detail in result["details"]}

    assert details == {
        "Version": "financial_v1",
        "Accuracy": "60.0%",
    }


def test_summary_cards_do_not_include_model_versions():
    result = _build_summary_cards(
        user_summary={"total_users": 7, "premium_users": 1},
        stock_summary={"active_stocks": 14, "inactive_stocks": 154},
        watchlist_summary={"total_entries": 1, "users_with_watchlists": 1},
        prediction_summary={"rows": [{"total": 49}]},
        sentiment_summary={"article_rows": 1000, "latest_date": "2026-08-04"},
        freshness_summary={"rows": [{"stale_count": 1}, {"stale_count": 0}]},
    )

    assert [card["label"] for card in result] == [
        "Total Users",
        "Active Stocks",
        "Watchlist Entries",
        "Predictions Stored",
        "Sentiment Articles",
        "Data Health Flags",
    ]
