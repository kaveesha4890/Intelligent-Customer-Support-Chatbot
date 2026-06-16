import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_model.eval()

CRISIS_KEYWORDS = ["suicide", "kill myself", "end my life", "harm myself"]

ESCALATION_PHRASES = [
    "i want to speak to a human",
    "speak to an agent",
    "talk to a person",
    "this is unacceptable",
    "i am furious",
    "i am outraged",
    "you are useless",
    "terrible service",
    "worst service",
    "i will sue",
    "legal action",
    "i demand",
    "extremely angry",
    "i hate this",
]

def analyse_sentiment(text: str) -> dict:
    text_lower = text.lower()

    # Crisis check first
    if any(k in text_lower for k in CRISIS_KEYWORDS):
        return {"label": "Crisis", "score": 1.0, "escalate": True}

    # Explicit escalation phrase check
    if any(phrase in text_lower for phrase in ESCALATION_PHRASES):
        return {"label": "Negative", "score": 1.0, "escalate": True}

    # Sentiment scoring - only for display, NOT for escalation decisions
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    score = probs[0][pred_id].item()

    label = _model.config.id2label[pred_id]
    if label == "NEGATIVE":
        mapped = "Negative"
    elif label == "POSITIVE":
        mapped = "Positive"
    else:
        mapped = "Neutral"

    # Never escalate based on ML score alone - only keywords/phrases above
    return {"label": mapped, "score": score, "escalate": False}