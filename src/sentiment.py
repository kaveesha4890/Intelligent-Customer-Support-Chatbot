from transformers import pipeline

sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

CRISIS_KEYWORDS = ["suicide", "kill myself", "end my life", "harm myself"]

def analyse_sentiment(text: str) -> dict:
    text_lower = text.lower()
    if any(k in text_lower for k in CRISIS_KEYWORDS):
        return {"label": "Crisis", "score": 1.0, "escalate": True}

    result = sentiment_pipe(text)[0]
    label, score = result["label"], result["score"]

    if label == "NEGATIVE":
        mapped = "Negative"
        escalate = score > 0.75
    elif label == "POSITIVE":
        mapped = "Positive"
        escalate = False
    else:
        mapped = "Neutral"
        escalate = False

    return {"label": mapped, "score": score, "escalate": escalate}