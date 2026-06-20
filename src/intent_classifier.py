from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os

MODEL_PATH = "models/intent_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

with open(os.path.join(MODEL_PATH, "labels.json"), "r") as f:
    LABELS = json.load(f)

def classify_intent(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    return {"intent": LABELS[pred_id], "confidence": probs[0][pred_id].item()}