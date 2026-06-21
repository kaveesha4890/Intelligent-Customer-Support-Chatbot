"""
Train the BERT intent classifier on the Banking77 dataset.
Uses a plain PyTorch training loop (no Trainer API) for Windows compatibility.

Run from the project root:
    python train_intent_classifier.py

Output: models/intent_classifier/   (tokenizer + model + labels.json)

Time estimate:
  - MX330 GPU (2GB VRAM, fp16): ~15-25 minutes
  - CPU only:                   ~60-120 minutes
"""

import os
import json

# Import order matters on Windows with CUDA:
# datasets must come before torch to avoid a DLL conflict
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "intent_classifier")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16    = torch.cuda.is_available()
BASE_MODEL  = "distilbert-base-uncased"   # 40% smaller than BERT, 2x faster, ~91% accuracy

print("=" * 60)
print("Intent Classifier Training")
print("=" * 60)
print(f"Device : {DEVICE}" + (f" — {torch.cuda.get_device_name(0)}" if USE_FP16 else ""))
print(f"fp16   : {USE_FP16}")
print(f"Saving : {OUTPUT_DIR}\n")

# ── 1. Load dataset ───────────────────────────────────────────
print("Loading Banking77 dataset...")
raw = load_dataset("mteb/banking77")

pairs       = sorted({(r["label"], r["label_text"]) for r in raw["train"]}, key=lambda x: x[0])
label_names = [name for _, name in pairs]
num_labels  = len(label_names)

print(f"  Classes : {num_labels}")
print(f"  Train   : {len(raw['train'])} examples")
print(f"  Test    : {len(raw['test'])} examples\n")

# ── 2. Tokenise ───────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

class IntentDataset(Dataset):
    def __init__(self, hf_split):
        enc = tokenizer(
            list(hf_split["text"]),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(list(hf_split["label"]), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

print("Tokenising dataset...")
train_ds = IntentDataset(raw["train"])
test_ds  = IntentDataset(raw["test"])

BATCH_SIZE = 8   # small for 2GB GPU
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── 3. Model ──────────────────────────────────────────────────
print(f"Loading {BASE_MODEL} model...")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=num_labels
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scaler    = GradScaler("cuda") if USE_FP16 else None

# ── 4. Training loop ──────────────────────────────────────────
EPOCHS              = 3
ACCUM_STEPS         = 2   # effective batch = 8 × 2 = 16
best_f1             = 0.0
best_state          = None

print(f"\nStarting training ({EPOCHS} epochs)...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader, 1):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        if USE_FP16:
            with autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss    = outputs.loss / ACCUM_STEPS
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / ACCUM_STEPS
            loss.backward()

        total_loss += loss.item() * ACCUM_STEPS

        if step % ACCUM_STEPS == 0:
            if USE_FP16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"  Epoch {epoch} | step {step}/{len(train_loader)} | loss {total_loss/step:.4f}", flush=True)

    # ── Evaluate ──────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch}/{EPOCHS} — loss: {avg_loss:.4f} | acc: {acc:.4f} | f1: {f1:.4f}")

    if f1 > best_f1:
        best_f1    = f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ✓ New best F1: {best_f1:.4f} — checkpoint saved\n")

# ── 5. Save best model ────────────────────────────────────────
print("\nSaving best model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.load_state_dict(best_state)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
    json.dump(label_names, f, indent=2)

print(f"\nDone! Best F1: {best_f1:.4f}")
print(f"Files saved to: {OUTPUT_DIR}")
print("\nYou can now run:  python ui/app.py")
