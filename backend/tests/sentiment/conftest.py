import pytest
from unittest.mock import MagicMock
import torch
import services.sentiment.finbert_service as finbert_module

SAMPLE_HEADLINES = [
    {"headline": "Apple reports record profits", "source": "finnhub", "published_at": "2026-05-24T09:00:00+00:00"},
    {"headline": "Tesla recalls 500,000 vehicles due to safety issue", "source": "finnhub", "published_at": "2026-05-24T10:00:00+00:00"},
    {"headline": "NVIDIA announces new GPU architecture", "source": "newsapi", "published_at": "2026-05-24T11:00:00+00:00"},
]

SAMPLE_SCORED_HEADLINES = [
    {**SAMPLE_HEADLINES[0], "label": "positive", "score": 0.92},
    {**SAMPLE_HEADLINES[1], "label": "negative", "score": 0.88},
    {**SAMPLE_HEADLINES[2], "label": "neutral", "score": 0.65},
]


@pytest.fixture(autouse=False)
def reset_finbert():
    finbert_module._model = None
    finbert_module._tokenizer = None
    yield
    finbert_module._model = None
    finbert_module._tokenizer = None


def make_mock_model_outputs(label_idx: int, batch_size: int = 1):
    """Logits with high value at label_idx so softmax → high probability for that label."""
    logits = torch.zeros(batch_size, 3)
    logits[:, label_idx] = 10.0
    mock_out = MagicMock()
    mock_out.logits = logits
    return mock_out


def make_mock_tokenizer_output(batch_size: int = 1, seq_len: int = 10):
    return {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


def make_mock_model(label_idx: int = 0):
    mock = MagicMock()
    mock.side_effect = lambda **kwargs: make_mock_model_outputs(
        label_idx, batch_size=kwargs["input_ids"].shape[0]
    )
    mock.eval.return_value = None
    return mock


def make_mock_tokenizer():
    mock = MagicMock()
    mock.side_effect = lambda texts, **kwargs: make_mock_tokenizer_output(
        batch_size=len(texts) if isinstance(texts, list) else 1
    )
    return mock
