import pytest
from unittest.mock import patch, MagicMock, call
import torch
import services.sentiment.finbert_service as finbert_module
from services.sentiment.finbert_service import score_headlines
from tests.sentiment.conftest import make_mock_model, make_mock_tokenizer, make_mock_model_outputs, make_mock_tokenizer_output

MODEL_PATH = "services.sentiment.finbert_service"


@pytest.fixture(autouse=True)
def reset(reset_finbert):
    pass


def _patch_transformers(label_idx=0):
    mock_model = make_mock_model(label_idx)
    mock_tok = make_mock_tokenizer()
    model_patch = patch(f"{MODEL_PATH}.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model)
    tok_patch = patch(f"{MODEL_PATH}.AutoTokenizer.from_pretrained", return_value=mock_tok)
    return tok_patch, model_patch


def test_score_empty_list():
    result = score_headlines([])
    assert result == []


def test_lazy_load_on_first_call():
    tok_p, model_p = _patch_transformers()
    with tok_p, model_p:
        assert finbert_module._model is None
        score_headlines(["Apple profits surge"])
        assert finbert_module._model is not None


def test_model_not_reloaded_on_second_call():
    tok_p, model_p = _patch_transformers()
    with tok_p as mock_tok, model_p as mock_model:
        score_headlines(["Apple profits surge"])
        score_headlines(["Tesla recall"])
        assert mock_model.call_count == 1
        assert mock_tok.call_count == 1


def test_score_returns_label_and_score():
    tok_p, model_p = _patch_transformers(label_idx=0)
    with tok_p, model_p:
        results = score_headlines(["Apple profits surge"])
    assert len(results) == 1
    assert results[0]["label"] == "positive"
    assert 0.0 <= results[0]["score"] <= 1.0


def test_score_negative_label():
    tok_p, model_p = _patch_transformers(label_idx=1)
    with tok_p, model_p:
        results = score_headlines(["Tesla massive recall"])
    assert results[0]["label"] == "negative"


def test_score_neutral_label():
    tok_p, model_p = _patch_transformers(label_idx=2)
    with tok_p, model_p:
        results = score_headlines(["Market remains flat"])
    assert results[0]["label"] == "neutral"


def test_score_single_headline():
    tok_p, model_p = _patch_transformers()
    with tok_p, model_p:
        results = score_headlines(["One headline"])
    assert len(results) == 1


def test_score_batch_larger_than_16():
    tok_p, model_p = _patch_transformers()
    headlines = [f"Headline number {i}" for i in range(20)]
    with tok_p, model_p:
        results = score_headlines(headlines)
    assert len(results) == 20


def test_headline_over_512_tokens():
    tok_p, model_p = _patch_transformers()
    long_headline = "word " * 600
    with tok_p as mock_tok_cls, model_p:
        results = score_headlines([long_headline])
    tok_instance = mock_tok_cls.return_value
    call_kwargs = tok_instance.call_args
    assert call_kwargs.kwargs.get("truncation") is True
    assert call_kwargs.kwargs.get("max_length") == 512
    assert len(results) == 1


def test_load_failure_raises():
    with patch(f"{MODEL_PATH}.AutoModelForSequenceClassification.from_pretrained", side_effect=OSError("no disk space")):
        with patch(f"{MODEL_PATH}.AutoTokenizer.from_pretrained", return_value=make_mock_tokenizer()):
            with pytest.raises(OSError):
                score_headlines(["Apple earnings"])


def test_result_count_matches_input():
    tok_p, model_p = _patch_transformers()
    headlines = ["Headline A", "Headline B", "Headline C"]
    with tok_p, model_p:
        results = score_headlines(headlines)
    assert len(results) == len(headlines)
