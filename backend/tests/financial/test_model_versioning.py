from datetime import datetime, timezone

import pytest

from backend.services.financial import financial_model
from backend.services.financial import financial_service


def test_version_id_is_stable_and_path_safe():
    trained_at = datetime(2026, 6, 11, 7, 8, 9, 123456, tzinfo=timezone.utc)

    version = financial_model._new_version_id(trained_at)

    assert version == "xgboost_financial_binary_20260611T070809123456Z"
    financial_model._validate_version(version)


def test_invalid_version_is_rejected():
    with pytest.raises(ValueError, match="Invalid financial model version"):
        financial_model._version_paths("../../other-model")


def test_activate_local_model_writes_latest_manifest(tmp_path, monkeypatch):
    version = "xgboost_financial_20260611T070809123456Z"
    models_dir = tmp_path / "models" / "financial"
    version_dir = models_dir / version
    version_dir.mkdir(parents=True)
    (version_dir / "model.ubj").write_bytes(b"model")
    (version_dir / "metadata.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(financial_model, "MODELS_DIR", models_dir)
    monkeypatch.setattr(
        financial_model,
        "LATEST_MANIFEST_PATH",
        models_dir / "latest.json",
    )
    monkeypatch.setattr(
        financial_model,
        "_relative_backend_path",
        lambda path: path.relative_to(tmp_path).as_posix(),
    )

    manifest = financial_model.activate_local_model(version)

    assert manifest["model_version"] == version
    assert financial_model.LATEST_MANIFEST_PATH.exists()


def test_binary_model_can_be_activated_as_product_model(monkeypatch):
    version = "xgboost_financial_binary_20260626T070809123456Z"

    monkeypatch.setattr(
        financial_service,
        "get_model_version",
        lambda model_version: {"model_version": model_version},
    )
    monkeypatch.setattr(
        financial_service,
        "activate_local_model",
        lambda model_version: {"model_version": model_version},
    )
    monkeypatch.setattr(
        financial_service,
        "activate_model_version",
        lambda model_version: {"model_version": model_version, "is_active": True},
    )

    activated = financial_service.set_active_financial_model(version)

    assert activated["model_version"] == version
    assert activated["is_active"] is True


def test_legacy_three_class_model_cannot_be_activated(monkeypatch):
    version = "xgboost_financial_20260611T070809123456Z"
    monkeypatch.setattr(
        financial_service,
        "get_model_version",
        lambda model_version: {"model_version": model_version},
    )

    with pytest.raises(ValueError, match="not a binary financial model"):
        financial_service.set_active_financial_model(version)
