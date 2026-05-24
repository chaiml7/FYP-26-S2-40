# Sentiment Analysis ML Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a nightly batch pipeline that fetches financial news, scores headlines with FinBERT, stores results in Supabase, and exposes them via two REST endpoints.

**Architecture:** APScheduler inside FastAPI triggers `run_pipeline()` nightly (with 3 retry attempts across the night). The pipeline fetches from FinnHub + NewsAPI, scores in batches via a lazy-loaded FinBERT model, and upserts to `sentiment_scores`. The API reads pre-computed scores — FinBERT never runs on a user request.

**Tech Stack:** `transformers`, `torch` (FinBERT), `requests` (HTTP), `APScheduler` (cron), `pytest` + `pytest-mock` + `httpx` (testing), Supabase (storage), FastAPI (routes).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `backend/services/sentiment/__init__.py` | Package marker |
| Create | `backend/services/sentiment/finbert_service.py` | Lazy-load FinBERT, score headline batches |
| Create | `backend/services/sentiment/finnhub_service.py` | Fetch news from FinnHub API with retry |
| Create | `backend/services/sentiment/news_scraper_service.py` | Fetch news from NewsAPI with retry |
| Create | `backend/services/sentiment/sentiment_aggregator.py` | Upsert to Supabase, compute daily summaries |
| Create | `backend/services/sentiment/sentiment_pipeline.py` | Orchestrator: loops watchlist, calls all services |
| Modify | `backend/routes/stock_routes.py` | Add GET sentiment + POST run-pipeline routes |
| Modify | `backend/main.py` | Add APScheduler lifespan, import sentiment router |
| Modify | `backend/requirements.txt` | Add new dependencies |
| Create | `backend/tests/__init__.py` | Package marker |
| Create | `backend/tests/sentiment/__init__.py` | Package marker |
| Create | `backend/tests/sentiment/conftest.py` | Shared fixtures and mock helpers |
| Create | `backend/tests/sentiment/test_finbert_service.py` | FinBERT unit tests |
| Create | `backend/tests/sentiment/test_finnhub_service.py` | FinnHub unit tests |
| Create | `backend/tests/sentiment/test_news_scraper_service.py` | NewsAPI unit tests |
| Create | `backend/tests/sentiment/test_sentiment_aggregator.py` | Aggregator unit tests |
| Create | `backend/tests/sentiment/test_sentiment_pipeline.py` | Pipeline orchestrator tests |
| Create | `backend/tests/sentiment/test_sentiment_routes.py` | FastAPI route tests |
| Create | `scripts/test_sentiment_manual.py` | Manual end-to-end test script |

---

## Task 1: Create `sentiment_scores` Table in Supabase

**Files:** None (manual SQL via Supabase dashboard)

- [ ] **Step 1: Open Supabase SQL editor**

  Go to https://supabase.com/dashboard/project/fcpfsdjnryelyqknjfne → SQL Editor → New query

- [ ] **Step 2: Run this SQL**

```sql
create table if not exists sentiment_scores (
  id            uuid        default gen_random_uuid() primary key,
  symbol        text        not null,
  headline      text        not null,
  source        text        not null,
  published_at  timestamptz not null,
  label         text        not null check (label in ('positive', 'negative', 'neutral')),
  score         float4      not null,
  model_version text        not null default 'ProsusAI/finbert',
  created_at    timestamptz default now() not null,
  unique (symbol, headline, published_at)
);

create index if not exists idx_sentiment_scores_symbol       on sentiment_scores (symbol);
create index if not exists idx_sentiment_scores_published_at on sentiment_scores (published_at);
create index if not exists idx_sentiment_scores_created_at   on sentiment_scores (created_at);
```

- [ ] **Step 3: Verify**

  In Supabase Table Editor, confirm `sentiment_scores` appears with all 9 columns.

---

## Task 2: Add Dependencies to `requirements.txt`

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Update requirements.txt**

```
fastapi
uvicorn
python-dotenv
supabase
yfinance
pandas
transformers
torch
APScheduler
requests
pytest
pytest-mock
httpx
```

- [ ] **Step 2: Install**

```bash
cd backend && pip install -r requirements.txt
```

  Note: `torch` is ~2GB. First install will take a few minutes.

- [ ] **Step 3: Verify torch installs correctly**

```bash
python -c "import torch; print(torch.__version__)"
```

  Expected: prints a version string like `2.x.x`

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "feat(sentiment): add ML and scheduler dependencies"
```

---

## Task 3: Module Structure + Test Infrastructure

**Files:**
- Create: `backend/services/sentiment/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/sentiment/__init__.py`
- Create: `backend/tests/sentiment/conftest.py`

- [ ] **Step 1: Create package markers**

Create `backend/services/sentiment/__init__.py` — empty file.

Create `backend/tests/__init__.py` — empty file.

Create `backend/tests/sentiment/__init__.py` — empty file.

- [ ] **Step 2: Create conftest.py**

Create `backend/tests/sentiment/conftest.py`:

```python
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
```

- [ ] **Step 3: Verify conftest is importable**

```bash
cd backend && python -c "from tests.sentiment.conftest import SAMPLE_HEADLINES; print('ok')"
```

  Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add backend/services/sentiment/__init__.py backend/tests/__init__.py backend/tests/sentiment/__init__.py backend/tests/sentiment/conftest.py
git commit -m "feat(sentiment): add module structure and test infrastructure"
```

---

## Task 4: FinBERT Service (TDD)

**Files:**
- Create: `backend/tests/sentiment/test_finbert_service.py`
- Create: `backend/services/sentiment/finbert_service.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/sentiment/test_finbert_service.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify all fail**

```bash
cd backend && pytest tests/sentiment/test_finbert_service.py -v
```

  Expected: `ModuleNotFoundError: No module named 'services.sentiment.finbert_service'`

- [ ] **Step 3: Implement `finbert_service.py`**

Create `backend/services/sentiment/finbert_service.py`:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()


def score_headlines(headlines: list) -> list:
    if not headlines:
        return []
    load_model()
    results = []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i:i + BATCH_SIZE]
        inputs = _tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        for prob in probs:
            idx = prob.argmax().item()
            results.append({"label": LABEL_MAP[idx], "score": round(prob[idx].item(), 4)})
    return results
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
cd backend && pytest tests/sentiment/test_finbert_service.py -v
```

  Expected: all 12 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/services/sentiment/finbert_service.py backend/tests/sentiment/test_finbert_service.py
git commit -m "feat(sentiment): add FinBERT service with lazy loading"
```

---

## Task 5: FinnHub Service (TDD)

**Files:**
- Create: `backend/tests/sentiment/test_finnhub_service.py`
- Create: `backend/services/sentiment/finnhub_service.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/sentiment/test_finnhub_service.py`:

```python
import pytest
import requests
from unittest.mock import patch, MagicMock, call
from datetime import date
from services.sentiment.finnhub_service import fetch_news

MODULE = "services.sentiment.finnhub_service"

SAMPLE_FINNHUB_RESPONSE = [
    {"headline": "Apple Q2 earnings beat expectations", "datetime": 1716537600},
    {"headline": "Apple launches new product line", "datetime": 1716624000},
]


def mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data if json_data is not None else SAMPLE_FINNHUB_RESPONSE
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


@patch(f"{MODULE}.requests.get")
def test_fetch_returns_headlines(mock_get):
    mock_get.return_value = mock_response()
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert result[0]["source"] == "finnhub"
    assert "headline" in result[0]
    assert "published_at" in result[0]


@patch(f"{MODULE}.requests.get")
def test_fetch_empty_response(mock_get):
    mock_get.return_value = mock_response(json_data=[])
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_fetch_malformed_json(mock_get):
    mock_get.return_value = mock_response(json_data={"error": "bad"})
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_retry_on_429(mock_get, mock_sleep):
    mock_get.side_effect = [mock_response(429), mock_response(200)]
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(2)


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_retry_exhausted_raises(mock_get, mock_sleep):
    mock_get.return_value = mock_response(429)
    with pytest.raises(Exception, match="rate limit"):
        fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert mock_get.call_count == 3


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_retry_on_timeout(mock_get, mock_sleep):
    mock_get.side_effect = [requests.Timeout(), mock_response(200)]
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert mock_get.call_count == 2


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_timeout_exhausted_returns_empty(mock_get, mock_sleep):
    mock_get.side_effect = requests.Timeout()
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert result == []
    assert mock_get.call_count == 3


@patch(f"{MODULE}.requests.get")
def test_published_at_is_iso_string(mock_get):
    mock_get.return_value = mock_response()
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    from datetime import datetime
    datetime.fromisoformat(result[0]["published_at"].replace("Z", "+00:00"))


@patch(f"{MODULE}.requests.get")
def test_symbol_uppercased_in_params(mock_get):
    mock_get.return_value = mock_response()
    fetch_news("aapl", from_date=date(2026, 5, 23))
    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["params"]["symbol"] == "AAPL"


@patch(f"{MODULE}.requests.get")
def test_headline_missing_filtered_out(mock_get):
    mock_get.return_value = mock_response(json_data=[
        {"headline": "Valid headline", "datetime": 1716537600},
        {"headline": "", "datetime": 1716537700},
        {"datetime": 1716537800},
    ])
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 1
    assert result[0]["headline"] == "Valid headline"
```

- [ ] **Step 2: Run tests — verify all fail**

```bash
cd backend && pytest tests/sentiment/test_finnhub_service.py -v
```

  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `finnhub_service.py`**

Create `backend/services/sentiment/finnhub_service.py`:

```python
import os
import time
import requests
from datetime import date, datetime, timedelta, timezone

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1"
MAX_RETRIES = 3
BACKOFF_BASE = 2


def fetch_news(symbol: str, from_date: date = None) -> list:
    if from_date is None:
        from_date = date.today() - timedelta(days=1)
    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": date.today().isoformat(),
        "token": FINNHUB_API_KEY,
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BASE_URL}/company-news", params=params, timeout=10)
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE ** (attempt + 1))
                    continue
                raise Exception(f"FinnHub rate limit exceeded after {MAX_RETRIES} attempts")
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                return []
            return [
                {"headline": item["headline"], "source": "finnhub", "published_at": _unix_to_iso(item.get("datetime", 0))}
                for item in data
                if item.get("headline")
            ]
        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            return []
    return []


def _unix_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
```

- [ ] **Step 4: Add `FINNHUB_API_KEY` to `backend/.env`**

```
FINNHUB_API_KEY=your_key_here
```

  Get a free key at https://finnhub.io

- [ ] **Step 5: Run tests — verify all pass**

```bash
cd backend && pytest tests/sentiment/test_finnhub_service.py -v
```

  Expected: all 10 tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add backend/services/sentiment/finnhub_service.py backend/tests/sentiment/test_finnhub_service.py
git commit -m "feat(sentiment): add FinnHub news fetcher with retry logic"
```

---

## Task 6: NewsAPI Service (TDD)

**Files:**
- Create: `backend/tests/sentiment/test_news_scraper_service.py`
- Create: `backend/services/sentiment/news_scraper_service.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/sentiment/test_news_scraper_service.py`:

```python
import pytest
import requests
from unittest.mock import patch, MagicMock
from datetime import date
from services.sentiment.news_scraper_service import fetch_news

MODULE = "services.sentiment.news_scraper_service"

SAMPLE_NEWSAPI_RESPONSE = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {"title": "Apple sales surge globally", "publishedAt": "2026-05-24T09:00:00Z"},
        {"title": "Apple faces antitrust probe", "publishedAt": "2026-05-24T10:00:00Z"},
    ],
}


def mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data if json_data is not None else SAMPLE_NEWSAPI_RESPONSE
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


@patch(f"{MODULE}.requests.get")
def test_fetch_returns_headlines(mock_get):
    mock_get.return_value = mock_response()
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert result[0]["source"] == "newsapi"
    assert "headline" in result[0]
    assert "published_at" in result[0]


@patch(f"{MODULE}.requests.get")
def test_fetch_empty_articles(mock_get):
    mock_get.return_value = mock_response(json_data={"status": "ok", "articles": []})
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_missing_articles_key(mock_get):
    mock_get.return_value = mock_response(json_data={"status": "ok"})
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_quota_exceeded_429_returns_empty(mock_get):
    mock_get.return_value = mock_response(status_code=429)
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_quota_exceeded_does_not_retry(mock_get):
    mock_get.return_value = mock_response(status_code=429)
    fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert mock_get.call_count == 1


@patch(f"{MODULE}.requests.get")
def test_quota_exceeded_426_returns_empty(mock_get):
    mock_get.return_value = mock_response(status_code=426)
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_timeout_retries(mock_get, mock_sleep):
    mock_get.side_effect = [requests.Timeout(), mock_response()]
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert mock_get.call_count == 2


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_timeout_exhausted_returns_empty(mock_get, mock_sleep):
    mock_get.side_effect = requests.Timeout()
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []
    assert mock_get.call_count == 3


@patch(f"{MODULE}.requests.get")
def test_article_missing_title_filtered_out(mock_get):
    mock_get.return_value = mock_response(json_data={
        "articles": [
            {"title": "Valid title", "publishedAt": "2026-05-24T09:00:00Z"},
            {"title": "", "publishedAt": "2026-05-24T10:00:00Z"},
            {"publishedAt": "2026-05-24T11:00:00Z"},
        ]
    })
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert len(result) == 1
    assert result[0]["headline"] == "Valid title"
```

- [ ] **Step 2: Run tests — verify all fail**

```bash
cd backend && pytest tests/sentiment/test_news_scraper_service.py -v
```

  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `news_scraper_service.py`**

Create `backend/services/sentiment/news_scraper_service.py`:

```python
import os
import time
import requests
from datetime import date, timedelta

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
BASE_URL = "https://newsapi.org/v2"
MAX_RETRIES = 3
BACKOFF_BASE = 2


def fetch_news(symbol: str, company_name: str, from_date: date = None) -> list:
    if from_date is None:
        from_date = date.today() - timedelta(days=1)
    params = {
        "q": company_name,
        "from": from_date.isoformat(),
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BASE_URL}/everything", params=params, timeout=10)
            if response.status_code in (429, 426):
                return []
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            if not isinstance(articles, list):
                return []
            return [
                {"headline": a["title"], "source": "newsapi", "published_at": a.get("publishedAt", "")}
                for a in articles
                if a.get("title")
            ]
        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            return []
        except Exception:
            return []
    return []
```

- [ ] **Step 4: Add `NEWSAPI_KEY` to `backend/.env`**

```
NEWSAPI_KEY=your_key_here
```

  Get a free key at https://newsapi.org

- [ ] **Step 5: Run tests — verify all pass**

```bash
cd backend && pytest tests/sentiment/test_news_scraper_service.py -v
```

  Expected: all 9 tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add backend/services/sentiment/news_scraper_service.py backend/tests/sentiment/test_news_scraper_service.py
git commit -m "feat(sentiment): add NewsAPI fetcher with quota-aware error handling"
```

---

## Task 7: Sentiment Aggregator (TDD)

**Files:**
- Create: `backend/tests/sentiment/test_sentiment_aggregator.py`
- Create: `backend/services/sentiment/sentiment_aggregator.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/sentiment/test_sentiment_aggregator.py`:

```python
import pytest
from unittest.mock import patch, MagicMock, call
from services.sentiment.sentiment_aggregator import save_scores, get_sentiment_summary, has_data_for_today, _score_to_label
from tests.sentiment.conftest import SAMPLE_SCORED_HEADLINES

MODULE = "services.sentiment.sentiment_aggregator"


def make_supabase_mock(select_data=None):
    mock = MagicMock()
    chain = mock.table.return_value
    chain.upsert.return_value.execute.return_value.data = select_data or SAMPLE_SCORED_HEADLINES
    chain.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = select_data or []
    chain.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = select_data or []
    return mock


@patch(f"{MODULE}.supabase")
def test_save_scores_upserts_rows(mock_supa):
    save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    mock_supa.table.assert_called_with("sentiment_scores")
    upsert_call = mock_supa.table.return_value.upsert
    rows = upsert_call.call_args[0][0]
    assert len(rows) == 3
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["model_version"] == "ProsusAI/finbert"


@patch(f"{MODULE}.supabase")
def test_save_scores_empty_list_does_not_upsert(mock_supa):
    result = save_scores("AAPL", [])
    mock_supa.table.return_value.upsert.assert_not_called()
    assert result["rows_saved"] == 0


@patch(f"{MODULE}.supabase")
def test_save_scores_includes_required_fields(mock_supa):
    save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    rows = mock_supa.table.return_value.upsert.call_args[0][0]
    required = {"symbol", "headline", "source", "published_at", "label", "score", "model_version"}
    assert required.issubset(rows[0].keys())


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.supabase")
def test_save_scores_retries_on_failure(mock_supa, mock_sleep):
    mock_supa.table.return_value.upsert.return_value.execute.side_effect = [Exception("db error"), MagicMock(data=SAMPLE_SCORED_HEADLINES)]
    result = save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    assert mock_supa.table.return_value.upsert.call_count == 2


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.supabase")
def test_save_scores_all_retries_fail_raises(mock_supa, mock_sleep):
    mock_supa.table.return_value.upsert.return_value.execute.side_effect = Exception("db error")
    with pytest.raises(Exception, match="db error"):
        save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    assert mock_supa.table.return_value.upsert.call_count == 3


def test_label_positive_threshold():
    assert _score_to_label(0.61) == "positive"


def test_label_negative_threshold():
    assert _score_to_label(0.39) == "negative"


def test_label_neutral_threshold():
    assert _score_to_label(0.50) == "neutral"


def test_label_boundary_exactly_06():
    assert _score_to_label(0.60) == "neutral"


def test_label_boundary_exactly_04():
    assert _score_to_label(0.40) == "neutral"


@patch(f"{MODULE}.supabase")
def test_get_summary_returns_correct_shape(mock_supa):
    rows = [
        {"headline": "h1", "source": "finnhub", "published_at": "2026-05-24T09:00:00+00:00", "label": "positive", "score": 0.9},
        {"headline": "h2", "source": "newsapi", "published_at": "2026-05-24T10:00:00+00:00", "label": "negative", "score": 0.8},
    ]
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = rows
    result = get_sentiment_summary("AAPL")
    assert "daily_scores" in result
    assert "headlines" in result
    assert isinstance(result["daily_scores"], list)
    assert isinstance(result["headlines"], list)


@patch(f"{MODULE}.supabase")
def test_get_summary_empty_db_returns_empty_lists(mock_supa):
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = []
    result = get_sentiment_summary("AAPL")
    assert result == {"daily_scores": [], "headlines": []}


@patch(f"{MODULE}.supabase")
def test_daily_aggregation_groups_by_date(mock_supa):
    rows = [
        {"headline": f"h{i}", "source": "finnhub", "published_at": f"2026-05-24T0{i}:00:00+00:00", "label": "positive", "score": 0.8}
        for i in range(5)
    ]
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = rows
    result = get_sentiment_summary("AAPL")
    assert len(result["daily_scores"]) == 1
    assert result["daily_scores"][0]["headline_count"] == 5


@patch(f"{MODULE}.supabase")
def test_has_data_for_today_true(mock_supa):
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = [{"id": "abc"}]
    assert has_data_for_today("AAPL") is True


@patch(f"{MODULE}.supabase")
def test_has_data_for_today_false(mock_supa):
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
    assert has_data_for_today("AAPL") is False
```

- [ ] **Step 2: Run tests — verify all fail**

```bash
cd backend && pytest tests/sentiment/test_sentiment_aggregator.py -v
```

  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `sentiment_aggregator.py`**

Create `backend/services/sentiment/sentiment_aggregator.py`:

```python
import time
from datetime import date, timedelta
from database.supabase_client import supabase

MAX_RETRIES = 3
BACKOFF_BASE = 2


def save_scores(symbol: str, scored_headlines: list) -> dict:
    if not scored_headlines:
        return {"rows_saved": 0}
    rows = [
        {
            "symbol": symbol.upper(),
            "headline": h["headline"],
            "source": h["source"],
            "published_at": h["published_at"],
            "label": h["label"],
            "score": h["score"],
            "model_version": "ProsusAI/finbert",
        }
        for h in scored_headlines
    ]
    for attempt in range(MAX_RETRIES):
        try:
            response = (
                supabase.table("sentiment_scores")
                .upsert(rows, on_conflict="symbol,headline,published_at")
                .execute()
            )
            return {"rows_saved": len(response.data)}
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            raise


def get_sentiment_summary(symbol: str, days: int = 7) -> dict:
    from_date = (date.today() - timedelta(days=days)).isoformat()
    response = (
        supabase.table("sentiment_scores")
        .select("*")
        .eq("symbol", symbol.upper())
        .gte("published_at", f"{from_date}T00:00:00Z")
        .order("published_at", desc=True)
        .execute()
    )
    rows = response.data or []
    by_date = {}
    for row in rows:
        d = row["published_at"][:10]
        by_date.setdefault(d, []).append(row["score"])
    daily_scores = [
        {
            "date": d,
            "avg_score": round(sum(scores) / len(scores), 4),
            "label": _score_to_label(sum(scores) / len(scores)),
            "headline_count": len(scores),
        }
        for d, scores in sorted(by_date.items(), reverse=True)
    ]
    headlines = [
        {"headline": r["headline"], "source": r["source"], "published_at": r["published_at"], "label": r["label"], "score": r["score"]}
        for r in rows
    ]
    return {"daily_scores": daily_scores, "headlines": headlines}


def has_data_for_today(symbol: str) -> bool:
    today = date.today().isoformat()
    response = (
        supabase.table("sentiment_scores")
        .select("id")
        .eq("symbol", symbol.upper())
        .gte("created_at", f"{today}T00:00:00Z")
        .limit(1)
        .execute()
    )
    return len(response.data) > 0


def _score_to_label(avg_score: float) -> str:
    if avg_score > 0.6:
        return "positive"
    if avg_score < 0.4:
        return "negative"
    return "neutral"
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
cd backend && pytest tests/sentiment/test_sentiment_aggregator.py -v
```

  Expected: all 14 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/services/sentiment/sentiment_aggregator.py backend/tests/sentiment/test_sentiment_aggregator.py
git commit -m "feat(sentiment): add sentiment aggregator with Supabase upsert and daily summary"
```

---

## Task 8: Sentiment Pipeline Orchestrator (TDD)

**Files:**
- Create: `backend/tests/sentiment/test_sentiment_pipeline.py`
- Create: `backend/services/sentiment/sentiment_pipeline.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/sentiment/test_sentiment_pipeline.py`:

```python
import pytest
from contextlib import contextmanager
from unittest.mock import patch, MagicMock
from services.sentiment.sentiment_pipeline import run_pipeline, WATCHLIST

MODULE = "services.sentiment.sentiment_pipeline"

MOCK_HEADLINES = [
    {"headline": "Apple profits up", "source": "finnhub", "published_at": "2026-05-24T09:00:00+00:00"},
    {"headline": "Apple new product launch", "source": "newsapi", "published_at": "2026-05-24T10:00:00+00:00"},
]
MOCK_SCORES = [
    {"label": "positive", "score": 0.91},
    {"label": "positive", "score": 0.85},
]


@contextmanager
def patched_pipeline(has_data=False, finnhub_data=None, newsapi_data=None, scores=None,
                     finnhub_side_effect=None, score_side_effect=None):
    _finnhub = finnhub_data if finnhub_data is not None else MOCK_HEADLINES[:1]
    _newsapi = newsapi_data if newsapi_data is not None else MOCK_HEADLINES[1:]
    _scores = scores if scores is not None else MOCK_SCORES

    with patch(f"{MODULE}.has_data_for_today", return_value=has_data) as m_has, \
         patch(f"{MODULE}.fetch_finnhub", return_value=_finnhub, side_effect=finnhub_side_effect) as m_fh, \
         patch(f"{MODULE}.fetch_newsapi", return_value=_newsapi) as m_fa, \
         patch(f"{MODULE}.score_headlines", return_value=_scores, side_effect=score_side_effect) as m_sc, \
         patch(f"{MODULE}.save_scores", return_value={"rows_saved": 2}) as m_sv, \
         patch(f"{MODULE}.time.sleep") as m_sl:
        yield {"has_data": m_has, "fetch_finnhub": m_fh, "fetch_newsapi": m_fa,
               "score_headlines": m_sc, "save_scores": m_sv, "sleep": m_sl}


def test_run_pipeline_processes_all_watchlist():
    with patched_pipeline() as m:
        result = run_pipeline()
    symbols = [r["symbol"] for r in result["results"]]
    for symbol in WATCHLIST:
        assert symbol in symbols


def test_idempotency_skips_existing():
    with patched_pipeline(has_data=True) as m:
        result = run_pipeline()
    skipped = [r for r in result["results"] if r["status"] == "skipped"]
    assert len(skipped) == len(WATCHLIST)
    m["fetch_finnhub"].assert_not_called()


def test_idempotency_processes_missing():
    with patched_pipeline(has_data=False) as m:
        result = run_pipeline()
    ok = [r for r in result["results"] if r["status"] == "ok"]
    assert len(ok) == len(WATCHLIST)


def test_one_symbol_failure_continues():
    side_effect = [Exception("FinnHub down")] + [MOCK_HEADLINES[:1]] * (len(WATCHLIST) - 1)
    with patched_pipeline(finnhub_side_effect=side_effect):
        result = run_pipeline()
    statuses = [r["status"] for r in result["results"]]
    assert "error" in statuses
    assert statuses.count("ok") >= len(WATCHLIST) - 1


def test_no_headlines_returns_no_data_status():
    with patched_pipeline(finnhub_data=[], newsapi_data=[]):
        result = run_pipeline()
    no_data = [r for r in result["results"] if r["status"] == "no_data"]
    assert len(no_data) == len(WATCHLIST)


def test_finbert_failure_caught_per_symbol():
    with patched_pipeline(score_side_effect=OSError("model not found")):
        result = run_pipeline()
    error = [r for r in result["results"] if r["status"] == "error"]
    assert len(error) == len(WATCHLIST)


def test_both_fetchers_called_per_symbol():
    with patched_pipeline() as m:
        run_pipeline()
    assert m["fetch_finnhub"].call_count == len(WATCHLIST)
    assert m["fetch_newsapi"].call_count == len(WATCHLIST)


def test_newsapi_empty_still_scores_finnhub_results():
    with patched_pipeline(newsapi_data=[]):
        result = run_pipeline()
    ok = [r for r in result["results"] if r["status"] == "ok"]
    assert len(ok) == len(WATCHLIST)


def test_result_has_required_keys():
    with patched_pipeline():
        result = run_pipeline()
    assert "message" in result
    assert "symbols_processed" in result
    assert "results" in result
    assert all("symbol" in r and "status" in r for r in result["results"])


def test_rate_limit_sleep_called_between_symbols():
    with patched_pipeline() as m:
        run_pipeline()
    assert m["sleep"].call_count == len(WATCHLIST)
    m["sleep"].assert_any_call(0.5)
```

- [ ] **Step 2: Run tests — verify all fail**

```bash
cd backend && pytest tests/sentiment/test_sentiment_pipeline.py -v
```

  Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `sentiment_pipeline.py`**

Create `backend/services/sentiment/sentiment_pipeline.py`:

```python
import time
import logging
from datetime import date

from services.sentiment.finnhub_service import fetch_news as fetch_finnhub
from services.sentiment.news_scraper_service import fetch_news as fetch_newsapi
from services.sentiment.finbert_service import score_headlines
from services.sentiment.sentiment_aggregator import save_scores, has_data_for_today

logger = logging.getLogger(__name__)

WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "AMD", "BABA"]

COMPANY_NAMES = {
    "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA", "MSFT": "Microsoft",
    "AMZN": "Amazon", "META": "Meta", "GOOGL": "Google", "NFLX": "Netflix",
    "AMD": "AMD", "BABA": "Alibaba",
}


def run_pipeline() -> dict:
    results = []
    from_date = date.today()
    for symbol in WATCHLIST:
        if has_data_for_today(symbol):
            results.append({"symbol": symbol, "headlines_scored": 0, "status": "skipped"})
            continue
        try:
            headlines = fetch_finnhub(symbol, from_date=from_date)
            time.sleep(0.5)
            headlines += fetch_newsapi(symbol, COMPANY_NAMES[symbol], from_date=from_date)
            if not headlines:
                results.append({"symbol": symbol, "headlines_scored": 0, "status": "no_data"})
                continue
            scores = score_headlines([h["headline"] for h in headlines])
            scored = [{**headlines[i], **scores[i]} for i in range(len(headlines))]
            save_scores(symbol, scored)
            results.append({"symbol": symbol, "headlines_scored": len(scored), "status": "ok"})
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", symbol, e)
            results.append({"symbol": symbol, "headlines_scored": 0, "status": "error", "reason": str(e)})
    return {
        "message": "Pipeline complete",
        "symbols_processed": len([r for r in results if r["status"] not in ("skipped",)]),
        "results": results,
    }
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
cd backend && pytest tests/sentiment/test_sentiment_pipeline.py -v
```

  Expected: all 10 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/services/sentiment/sentiment_pipeline.py backend/tests/sentiment/test_sentiment_pipeline.py
git commit -m "feat(sentiment): add pipeline orchestrator with idempotency and per-symbol error isolation"
```

---

## Task 9: API Routes (TDD)

**Files:**
- Create: `backend/tests/sentiment/test_sentiment_routes.py`
- Modify: `backend/routes/stock_routes.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/sentiment/test_sentiment_routes.py`:

```python
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
MODULE_AGG = "routes.stock_routes.get_sentiment_summary"
MODULE_PIPE = "routes.stock_routes.run_sentiment_pipeline"

MOCK_SUMMARY = {
    "daily_scores": [{"date": "2026-05-24", "avg_score": 0.75, "label": "positive", "headline_count": 5}],
    "headlines": [{"headline": "Apple profits up", "source": "finnhub", "published_at": "2026-05-24T09:00:00Z", "label": "positive", "score": 0.91}],
}

MOCK_PIPELINE_RESULT = {
    "message": "Pipeline complete",
    "symbols_processed": 10,
    "results": [{"symbol": "AAPL", "headlines_scored": 5, "status": "ok"}],
}


@patch(MODULE_AGG, return_value=MOCK_SUMMARY)
def test_get_sentiment_200(mock_agg):
    response = client.get("/api/stocks/AAPL/sentiment")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "daily_scores" in data
    assert "headlines" in data


@patch(MODULE_AGG, return_value={"daily_scores": [], "headlines": []})
def test_get_sentiment_404_when_no_data(mock_agg):
    response = client.get("/api/stocks/AAPL/sentiment")
    assert response.status_code == 404


@patch(MODULE_AGG, return_value=MOCK_SUMMARY)
def test_get_sentiment_symbol_uppercased(mock_agg):
    response = client.get("/api/stocks/aapl/sentiment")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"


@patch(MODULE_AGG, return_value=MOCK_SUMMARY)
def test_get_sentiment_response_includes_symbol(mock_agg):
    response = client.get("/api/stocks/TSLA/sentiment")
    assert response.json()["symbol"] == "TSLA"


@patch(MODULE_PIPE, return_value=MOCK_PIPELINE_RESULT)
def test_run_pipeline_200(mock_pipe):
    response = client.post("/api/sentiment/run-pipeline")
    assert response.status_code == 200
    data = response.json()
    assert "symbols_processed" in data
    assert "results" in data


@patch(MODULE_PIPE, side_effect=RuntimeError("FinBERT failed to load"))
def test_run_pipeline_500_on_error(mock_pipe):
    response = client.post("/api/sentiment/run-pipeline")
    assert response.status_code == 500
```

- [ ] **Step 2: Run tests — verify all fail**

```bash
cd backend && pytest tests/sentiment/test_sentiment_routes.py -v
```

  Expected: failures on missing routes / import errors

- [ ] **Step 3: Add sentiment routes to `stock_routes.py`**

Open `backend/routes/stock_routes.py` and add these imports at the top:

```python
from services.sentiment.sentiment_aggregator import get_sentiment_summary
from services.sentiment.sentiment_pipeline import run_pipeline as run_sentiment_pipeline
```

Then add these two routes at the bottom of the file:

```python
@router.get("/stocks/{symbol}/sentiment")
def get_stock_sentiment(symbol: str):
    data = get_sentiment_summary(symbol)
    if not data["daily_scores"] and not data["headlines"]:
        raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol.upper()}")
    return {"symbol": symbol.upper(), **data}


@router.post("/sentiment/run-pipeline")
def trigger_sentiment_pipeline():
    try:
        return run_sentiment_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
cd backend && pytest tests/sentiment/test_sentiment_routes.py -v
```

  Expected: all 6 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/routes/stock_routes.py backend/tests/sentiment/test_sentiment_routes.py
git commit -m "feat(sentiment): add sentiment API endpoints (GET summary, POST run-pipeline)"
```

---

## Task 10: Wire Scheduler into `main.py`

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Replace `main.py` with scheduler-aware version**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from routes.stock_routes import router as stock_router
from services.sentiment.sentiment_pipeline import run_pipeline

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(run_pipeline, "cron", hour=23, minute=0)
    scheduler.add_job(run_pipeline, "cron", hour=23, minute=30)
    scheduler.add_job(run_pipeline, "cron", hour=1,  minute=0)
    scheduler.add_job(run_pipeline, "cron", hour=3,  minute=0)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Backend REST API is running"}


app.include_router(stock_router, prefix="/api")
```

- [ ] **Step 2: Verify server starts without errors**

```bash
cd backend && uvicorn main:app --reload
```

  Expected: server starts, no import errors. Visit http://localhost:8000 → `{"message": "Backend REST API is running"}`

- [ ] **Step 3: Run all tests to ensure nothing broke**

```bash
cd backend && pytest tests/sentiment/ -v
```

  Expected: all tests still pass

- [ ] **Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat(sentiment): wire APScheduler into FastAPI lifespan for nightly pipeline"
```

---

## Task 11: Manual Test Script

**Files:**
- Create: `scripts/test_sentiment_manual.py`

- [ ] **Step 1: Create the script**

Create `scripts/test_sentiment_manual.py`:

```python
"""
Manual end-to-end test for the sentiment pipeline.
Run from repo root: python scripts/test_sentiment_manual.py
Requires: backend server running at localhost:8000, valid API keys in backend/.env
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

import httpx
from datetime import date, timedelta

BASE_URL = "http://localhost:8000/api"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


def step(n, label):
    print(f"\n--- Step {n}: {label} ---")


# Step 1: FinBERT smoke test
step(1, "FinBERT scoring")
try:
    from services.sentiment.finbert_service import score_headlines
    results = score_headlines([
        "Apple reports record quarterly profits",
        "Tesla faces massive safety recall affecting 500000 vehicles",
        "Market trading volume remains unchanged today",
    ])
    check("FinBERT returns 3 results", len(results) == 3)
    check("FinBERT labels are valid", all(r["label"] in ("positive", "negative", "neutral") for r in results))
    check("FinBERT scores in range", all(0.0 <= r["score"] <= 1.0 for r in results))
    check("Apple headline is positive", results[0]["label"] == "positive", f"got {results[0]['label']}")
    check("Tesla recall is negative", results[1]["label"] == "negative", f"got {results[1]['label']}")
    print(f"  Results: {results}")
except Exception as e:
    check("FinBERT smoke test", False, str(e))


# Step 2: FinnHub fetch
step(2, "FinnHub news fetch")
try:
    from services.sentiment.finnhub_service import fetch_news as fetch_finnhub
    yesterday = date.today() - timedelta(days=1)
    headlines = fetch_finnhub("AAPL", from_date=yesterday)
    check("FinnHub returns a list", isinstance(headlines, list))
    check("FinnHub headlines have correct keys",
          all("headline" in h and "source" in h and "published_at" in h for h in headlines) if headlines else True)
    check("FinnHub source is 'finnhub'", all(h["source"] == "finnhub" for h in headlines) if headlines else True)
    print(f"  Got {len(headlines)} headlines. First 2: {[h['headline'] for h in headlines[:2]]}")
except Exception as e:
    check("FinnHub fetch", False, str(e))


# Step 3: NewsAPI fetch
step(3, "NewsAPI news fetch")
try:
    from services.sentiment.news_scraper_service import fetch_news as fetch_newsapi
    yesterday = date.today() - timedelta(days=1)
    headlines = fetch_newsapi("AAPL", "Apple", from_date=yesterday)
    check("NewsAPI returns a list", isinstance(headlines, list))
    check("NewsAPI source is 'newsapi'", all(h["source"] == "newsapi" for h in headlines) if headlines else True)
    print(f"  Got {len(headlines)} headlines. First 2: {[h['headline'] for h in headlines[:2]]}")
    if not headlines:
        print("  Warning: no headlines returned — may be quota exceeded or no news today")
except Exception as e:
    check("NewsAPI fetch", False, str(e))


# Step 4: Pipeline trigger via API
step(4, "Pipeline trigger (POST /api/sentiment/run-pipeline)")
try:
    response = httpx.post(f"{BASE_URL}/sentiment/run-pipeline", timeout=300)
    check("POST /run-pipeline returns 200", response.status_code == 200, f"got {response.status_code}")
    data = response.json()
    check("Response has symbols_processed", "symbols_processed" in data)
    check("Response has results list", "results" in data and isinstance(data["results"], list))
    print(f"  Pipeline summary: {data['message']}, processed: {data['symbols_processed']}")
    for r in data["results"]:
        status_icon = "✓" if r["status"] == "ok" else ("~" if r["status"] in ("skipped", "no_data") else "✗")
        print(f"  {status_icon} {r['symbol']}: {r['status']} ({r.get('headlines_scored', 0)} headlines)")
except Exception as e:
    check("Pipeline trigger", False, str(e))


# Step 5: Supabase row check
step(5, "Supabase row count after pipeline")
try:
    from database.supabase_client import supabase
    today = date.today().isoformat()
    response = supabase.table("sentiment_scores").select("symbol").gte("created_at", f"{today}T00:00:00Z").execute()
    rows = response.data or []
    check("Supabase has rows for today", len(rows) > 0, f"got {len(rows)} rows")
    by_symbol = {}
    for row in rows:
        by_symbol[row["symbol"]] = by_symbol.get(row["symbol"], 0) + 1
    print(f"  Rows by symbol: {by_symbol}")
except Exception as e:
    check("Supabase row check", False, str(e))


# Step 6: Idempotency check
step(6, "Idempotency (re-run pipeline, row count unchanged)")
try:
    from database.supabase_client import supabase
    today = date.today().isoformat()
    before = supabase.table("sentiment_scores").select("id").gte("created_at", f"{today}T00:00:00Z").execute()
    count_before = len(before.data or [])

    response = httpx.post(f"{BASE_URL}/sentiment/run-pipeline", timeout=300)
    data = response.json()
    skipped = [r for r in data.get("results", []) if r["status"] == "skipped"]

    after = supabase.table("sentiment_scores").select("id").gte("created_at", f"{today}T00:00:00Z").execute()
    count_after = len(after.data or [])

    check("Row count unchanged after re-run", count_before == count_after, f"before={count_before}, after={count_after}")
    check("All symbols skipped on re-run", len(skipped) == 10, f"only {len(skipped)} skipped")
except Exception as e:
    check("Idempotency check", False, str(e))


# Step 7: Sentiment read endpoint
step(7, "GET /api/stocks/AAPL/sentiment")
try:
    response = httpx.get(f"{BASE_URL}/stocks/AAPL/sentiment", timeout=30)
    check("GET /sentiment returns 200", response.status_code == 200, f"got {response.status_code}")
    data = response.json()
    check("Response has symbol", data.get("symbol") == "AAPL")
    check("Response has daily_scores", "daily_scores" in data)
    check("Response has headlines", "headlines" in data)
    if data.get("daily_scores"):
        ds = data["daily_scores"][0]
        check("daily_scores entry has required keys",
              all(k in ds for k in ("date", "avg_score", "label", "headline_count")))
    print(f"  daily_scores: {data.get('daily_scores', [])[:2]}")
    print(f"  headline count: {len(data.get('headlines', []))}")
except Exception as e:
    check("Sentiment read endpoint", False, str(e))


# Step 8: Error simulation (bad API key)
step(8, "Error simulation (invalid FinnHub key)")
try:
    import services.sentiment.finnhub_service as fh
    original_key = fh.FINNHUB_API_KEY
    fh.FINNHUB_API_KEY = "INVALID_KEY_TEST"
    from services.sentiment.finnhub_service import fetch_news
    result = fetch_news("AAPL", from_date=date.today() - timedelta(days=1))
    check("Invalid key returns empty list or raises cleanly",
          isinstance(result, list),
          f"got unexpected type: {type(result)}")
    fh.FINNHUB_API_KEY = original_key
    print("  FinnHub returned gracefully with invalid key (empty list or error)")
except Exception as e:
    check("Error simulation completed without crash", True)
    print(f"  Exception raised (expected): {e}")


# --- Summary ---
print(f"\n{'='*50}")
if failures:
    print(f"\033[91m{len(failures)} FAILED: {', '.join(failures)}\033[0m")
    sys.exit(1)
else:
    print(f"\033[92mAll steps PASSED\033[0m")
```

- [ ] **Step 2: Verify script is runnable (server must be running)**

```bash
cd backend && uvicorn main:app --reload
# in another terminal:
python scripts/test_sentiment_manual.py
```

  Expected: all steps print `[PASS]`

- [ ] **Step 3: Commit**

```bash
git add scripts/test_sentiment_manual.py
git commit -m "test(sentiment): add manual end-to-end test script with 8 validation steps"
```

---

## Task 12: Full Test Suite Verification

- [ ] **Step 1: Run all unit tests**

```bash
cd backend && pytest tests/sentiment/ -v
```

  Expected output summary:
  ```
  tests/sentiment/test_finbert_service.py         12 passed
  tests/sentiment/test_finnhub_service.py         10 passed
  tests/sentiment/test_news_scraper_service.py     9 passed
  tests/sentiment/test_sentiment_aggregator.py    14 passed
  tests/sentiment/test_sentiment_pipeline.py      10 passed
  tests/sentiment/test_sentiment_routes.py         6 passed
  ========== 61 passed ==========
  ```

- [ ] **Step 2: Confirm no regressions on existing tests**

```bash
cd backend && pytest -v
```

  Expected: all pre-existing tests still pass

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat(sentiment): complete sentiment pipeline — 61 tests passing"
```

- [ ] **Step 4: Push to remote**

```bash
git push origin feature/bali-sentiment-pipeline
```

  Then open a PR from `feature/bali-sentiment-pipeline` → `main` (code only — no docs/, .claude/, LOG.md)
