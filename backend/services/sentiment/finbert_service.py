# Fine-tuned model hosted on HuggingFace Hub
# Trained on Twitter Financial News Sentiment: 87.2% accuracy (vs 53% baseline)
# Auto-downloads on first use - zero setup required for team members
FINETUNED_MODEL = "balibpt/finbert-stocklens"
BASE_MODEL = "ProsusAI/finbert"

# Set to True to use fine-tuned model (production/demos)
# Set to False to use base model (faster loading for dev)
USE_FINETUNED = True

BATCH_SIZE = 16

_tokenizer = None
_model = None
_label_map = None
_torch = None
_softmax = None


def load_model():
    """
    Load FinBERT model from HuggingFace Hub.

    If USE_FINETUNED=True, loads our fine-tuned model (87% accuracy).
    Otherwise uses base ProsusAI/finbert (53% accuracy).

    Model auto-downloads and caches on first use - no manual setup needed.
    """
    global _tokenizer, _model, _label_map, _torch, _softmax
    if _model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise RuntimeError(
                "FinBERT sentiment scoring requires ML dependencies. "
                "Install them with: pip install -r requirements-ml.txt"
            ) from e

        model_name = FINETUNED_MODEL if USE_FINETUNED else BASE_MODEL

        print(f"[OK] Loading {model_name} from HuggingFace Hub...")

        try:
            # Tokenizer always from base model (fine-tuning doesn't change vocabulary)
            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            # Model from fine-tuned if available
            mdl = AutoModelForSequenceClassification.from_pretrained(model_name)

            if USE_FINETUNED:
                print(f"     Fine-tuned model: 87.2% accuracy, F1=0.83")
            else:
                print(f"     Base model: 53% accuracy (fallback mode)")

            # Use model's config label mapping (normalize to int keys)
            # HuggingFace configs use string keys, but we need int for indexing
            _label_map = {int(k): v for k, v in mdl.config.id2label.items()}
            print(f"     Label mapping: {_label_map}")

            mdl.eval()
            _tokenizer = tok
            _model = mdl
            _torch = torch
            _softmax = F.softmax

            print(f"     Model loaded successfully!")

        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            print(f"        Falling back to base model...")

            # Fallback to base model on any error
            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
            _label_map = mdl.config.id2label
            mdl.eval()
            _tokenizer = tok
            _model = mdl
            _torch = torch
            _softmax = F.softmax


def score_headlines(headlines: list) -> list:
    """
    Score sentiment of financial news headlines using fine-tuned FinBERT.

    Args:
        headlines: List of text strings to analyze

    Returns:
        List of dicts with keys: {"label": str, "score": float}
        Labels: "positive", "negative", "neutral"
        Score: confidence probability [0.0, 1.0]
    """
    if not headlines:
        return []
    load_model()
    results = []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i:i + BATCH_SIZE]
        inputs = _tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with _torch.no_grad():
            outputs = _model(**inputs)
        probs = _softmax(outputs.logits, dim=-1)
        for prob in probs:
            idx = prob.argmax().item()
            # Use label map from model config (int keys)
            results.append({
                "label": _label_map[idx],
                "score": round(prob[idx].item(), 4)
            })
    return results
