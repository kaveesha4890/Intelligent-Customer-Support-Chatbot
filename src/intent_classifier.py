from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/intent_classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Banking77 label names (order matters - get from dataset.features["label"].names)
from datasets import load_dataset
LABELS = load_dataset("banking77")["train"].features["label"].names

def classify_intent(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    return {"intent": LABELS[pred_id], "confidence": probs[0][pred_id].item()}