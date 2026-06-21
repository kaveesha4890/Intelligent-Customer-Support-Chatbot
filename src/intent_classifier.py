import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "intent_classifier")

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"\n\n[Intent Classifier] Model not found at:\n  {MODEL_PATH}\n\n"
        "You need to train it first. Run this command from the project root:\n"
        "    python train_intent_classifier.py\n\n"
        "Training takes ~10-30 minutes. It only needs to be done once.\n"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

labels_path = os.path.join(MODEL_PATH, "labels.json")
if os.path.isfile(labels_path):
    with open(labels_path, "r") as f:
        LABELS = json.load(f)
else:
    from datasets import load_dataset
    _ds = load_dataset("mteb/banking77")["train"]
    _pairs = sorted({(r["label"], r["label_text"]) for r in _ds}, key=lambda x: x[0])
    LABELS = [name for _, name in _pairs]

def classify_intent(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    return {"intent": LABELS[pred_id], "confidence": probs[0][pred_id].item()}