from datetime import datetime, timezone

import pytest

from services.financial import financial_model


def test_version_id_is_stable_and_path_safe():
    trained_at = datetime(2026, 6, 11, 7, 8, 9, 123456, tzinfo=timezone.utc)

    version = financial_model._new_version_id(trained_at)

    assert version == "xgboost_financial_20260611T070809123456Z"
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
