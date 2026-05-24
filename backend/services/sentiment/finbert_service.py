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
