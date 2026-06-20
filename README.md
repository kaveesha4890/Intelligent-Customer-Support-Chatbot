# 🤖 Intelligent Customer Support Chatbot

An end-to-end AI-powered customer support chatbot built for the banking and telecom domain. This project combines state-of-the-art NLP techniques — fine-tuned BERT intent classification, sentiment-aware escalation routing, Retrieval-Augmented Generation (RAG), and a local LLM — into a fully working web chat interface.

Built as a final-year BSc Computer Engineering group project (EC7203 — Advanced Artificial Intelligence, 2026).

---

## 🌟 What Makes This Special?

Most simple chatbots rely on **keyword matching** or **fixed decision trees** — they break the moment a user phrases something differently. This project is different:

| Feature | This Project | Typical Chatbot |
|---|---|---|
| Intent understanding | Fine-tuned BERT (77 intents) | Keyword matching |
| Response generation | RAG + Local LLM (grounded answers) | Pre-written templates |
| Sentiment detection | DistilBERT neural model | None |
| Human escalation | Automatic (crisis + anger detection) | Manual button only |
| Knowledge retrieval | FAISS semantic vector search | Database lookup |
| Privacy | 100% local (no cloud API) | Cloud-dependent |

The system understands **what the customer means**, retrieves **relevant knowledge**, and generates **grounded, accurate answers** — not hallucinated ones. All processing runs locally on your machine with no external API calls.

---

## 🏗️ System Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  chatbot_pipeline.py                │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ BERT Intent  │    │ DistilBERT Sentiment     │  │
│  │ Classifier   │    │ + Crisis Keyword Check   │  │
│  │ (77 classes) │    │                          │  │
│  └──────┬───────┘    └────────────┬─────────────┘  │
│         │                         │                 │
│         │              ┌──────────▼──────────┐      │
│         │              │  Escalate to Human? │      │
│         │              └──────────┬──────────┘      │
│         │                    No   │                 │
│         │            ┌────────────▼──────────────┐  │
│         │            │  FAISS Semantic Retrieval │  │
│         │            │  (knowledge_base/ docs)   │  │
│         │            └────────────┬──────────────┘  │
│         │                         │                 │
│         └──────────┐   ┌──────────▼──────────────┐  │
│                    │   │   Prompt Construction   │  │
│                    │   │   (context + query)     │  │
│                    │   └──────────┬──────────────┘  │
│                    │              │                 │
│                    │   ┌──────────▼──────────────┐  │
│                    │   │  Ollama LLM (llama3.2)  │  │
│                    │   │  Response Generation    │  │
│                    │   └──────────┬──────────────┘  │
└────────────────────┼──────────────┼─────────────────┘
                     │              │
                     ▼              ▼
              Intent Label     Final Response
              Sentiment Label  → Flask Web UI
```

---

## 📁 Folder Structure

```
AI-Chatbot-Project-2026/
│
├── src/                          # Core AI modules
│   ├── __init__.py               # Makes src a Python package
│   ├── intent_classifier.py      # BERT intent classification
│   ├── sentiment.py              # DistilBERT sentiment + escalation
│   ├── document_processor.py     # Loads & chunks knowledge base docs
│   ├── embedder.py               # Embeds docs & builds FAISS index
│   ├── retriever.py              # Retrieves relevant docs for a query
│   ├── prompt_templates.py       # LLM prompt templates
│   ├── llm_generator.py          # Calls Ollama LLM
│   └── chatbot_pipeline.py       # Orchestrates the full pipeline
│
├── ui/
│   └── app.py                    # Flask web UI (chat interface)
│
├── knowledge_base/               # FAQ documents for RAG
│   ├── billing/                  # 15 billing-related FAQ .txt files
│   ├── account/                  # 15 account-related FAQ .txt files
│   ├── technical/                # 10 technical support FAQ .txt files
│   └── fraud/                    # 10 fraud/security FAQ .txt files
│
├── models/                       # Generated model files (not in git)
│   ├── intent_classifier/        # Fine-tuned BERT checkpoint (from Colab)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   │   └── labels.json           # Banking77 label names (77 classes)
│   ├── faiss_index.bin           # FAISS vector index (built locally)
│   └── chunks.pkl                # Knowledge base text chunks
│
├── notebooks/
│   ├── 01_data_eda.ipynb         # Banking77 dataset exploration
│   ├── 02_intent_model.ipynb     # BERT fine-tuning (run on Colab GPU)
│   ├── 03_rag_pipeline.ipynb     # RAG pipeline testing
│   ├── 04_prompt_engineering.ipynb # Prompt strategy comparison
│   └── 05_evaluation.ipynb       # Full system evaluation + charts
│
├── evaluation/
│   ├── test_queries.csv          # Test queries + reference answers
│   ├── results.csv               # Generated evaluation results
│   └── run_eval.py               # Evaluation script (BLEU + ROUGE-L)
│
├── build_index.py                # One-time script to build FAISS index
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes models/, venv/, __pycache__/
└── README.md                     # This file
```

---

## 🧠 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Intent Classification | BERT (`bert-base-uncased`) fine-tuned on Banking77 | Classify user message into 77 banking intents |
| Sentiment Analysis | DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) | Detect customer emotion |
| Escalation Detection | Keyword matching + sentiment scoring | Route distressed customers to humans |
| Document Chunking | LangChain `RecursiveCharacterTextSplitter` | Split FAQ docs into 512-token chunks |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) | Convert text to 384-dim vectors |
| Vector Search | FAISS `IndexFlatIP` | Find most relevant knowledge base chunks |
| LLM | `llama3.2:1b` via Ollama | Generate final grounded response |
| Web UI | Flask + vanilla JavaScript | Chat interface with real-time responses |
| Training Platform | Google Colab (GPU) | Fine-tune BERT intent classifier |

---

## ⚙️ What Each Source File Does

### `src/intent_classifier.py`
Loads the fine-tuned BERT model from `models/intent_classifier/` and classifies any user message into one of 77 Banking77 intent categories (e.g. `card_payment_fee_charged`, `passcode_forgotten`, `lost_or_stolen_card`). Uses `labels.json` to map model output IDs to human-readable intent names.

### `src/sentiment.py`
Uses DistilBERT to score the emotional tone of a message. Also checks for crisis keywords (`"suicide"`, `"kill myself"` etc.) which trigger immediate escalation regardless of sentiment score. Sets the `escalate` flag which the pipeline checks before calling the LLM.

### `src/document_processor.py`
Walks through all `.txt` files in `knowledge_base/` and loads them as LangChain `Document` objects. Splits each document into overlapping 512-token chunks (50-token overlap) to ensure context is not cut off at boundaries.

### `src/embedder.py`
Takes document chunks and encodes them into 384-dimensional vectors using `all-MiniLM-L6-v2`. Saves the FAISS index (`.bin`) and the raw text chunks (`.pkl`) to the `models/` folder for later retrieval.

### `src/retriever.py`
Loads the FAISS index and chunks at startup (lazy-loaded on first call). For any query, encodes it into a vector and finds the top-K most similar chunks using cosine similarity. Returns only chunks above a similarity threshold of 0.1.

### `src/prompt_templates.py`
Contains the prompt template used to instruct the LLM. The template injects the retrieved context and the user's question, and explicitly tells the model to answer only from the provided context — preventing hallucination.

### `src/llm_generator.py`
Sends the constructed prompt to the locally running Ollama instance (`llama3.2:1b` model) and returns the generated text response.

### `src/chatbot_pipeline.py`
The main orchestrator. For each user message:
1. Classifies intent (BERT)
2. Detects sentiment (DistilBERT)
3. Checks escalation flag
4. If escalating → returns handoff message
5. If not → retrieves top-3 knowledge base chunks → builds prompt → generates LLM response → cleans output
6. Returns: `(response, intent, sentiment, escalated)`

### `ui/app.py`
Flask web application. Serves an HTML/JS chat interface at `http://127.0.0.1:5000`. The frontend sends each message to the `/chat` endpoint via `fetch()`, receives the response JSON, and updates the chat window, intent label, sentiment label, and escalation banner.

### `build_index.py`
One-time setup script. Loads documents from `knowledge_base/`, chunks them, embeds them, and saves `models/faiss_index.bin` and `models/chunks.pkl`. Must be rerun whenever `knowledge_base/` content changes.

---

## 🚀 Setup Guide (From Scratch)

### Prerequisites
- Python 3.10, 3.11, or 3.12 (recommended — **avoid 3.13** due to torch wheel limitations)
- Git
- [Ollama](https://ollama.com/download) installed
- Google Colab account (for BERT training only)
- At least 8GB RAM, 10GB free disk space

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/AI-Chatbot-Project-2026.git
cd AI-Chatbot-Project-2026
```

### Step 2 — Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install torch==2.6.0 transformers==4.49.0 tokenizers==0.21.0 safetensors==0.5.2
pip install faiss-cpu sentence-transformers ollama flask
pip install langchain-text-splitters langchain-community langchain-core
pip install pandas matplotlib rouge-score nltk scikit-learn
```

### Step 4 — Set Up Ollama

1. Download and install Ollama from https://ollama.com/download
2. Pull the small LLM model:
```bash
ollama pull llama3.2:1b
```
3. Verify it works:
```bash
ollama run llama3.2:1b
```
Type a message, confirm it responds, then type `/bye` to exit.

> **Low RAM tip:** `llama3.2:1b` requires ~2GB RAM. If you have only 8GB total, close other applications before running the chatbot. Do NOT use `llama3` (8B) on 8GB RAM machines.

> **Disk space tip:** If your C: drive is limited, set the model storage location before pulling:
> Add environment variable `OLLAMA_MODELS` = `D:\OllamaModels` (or any drive with space)

### Step 5 — Generate the Intent Classifier (Google Colab)

The BERT intent classifier must be trained on a GPU. Your laptop CPU is not sufficient for training.

1. Open `notebooks/02_intent_model.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set Runtime → Change runtime type → **T4 GPU**
3. Add this as the **first cell** (fixes a Colab environment issue):
```python
from datasets import config
config.TORCHVISION_AVAILABLE = False
```
4. Run all cells — training takes approximately 20-30 minutes
5. After training, the model is saved to `models/intent_classifier/`
6. Also run this cell in Colab to save the label names:
```python
import json
ds = load_dataset("mteb/banking77")
import pandas as pd
df = ds["train"].to_pandas()
label_map = df[["label", "label_text"]].drop_duplicates().sort_values("label")
labels = label_map["label_text"].tolist()
with open("models/intent_classifier/labels.json", "w") as f:
    json.dump(labels, f)
print("Saved", len(labels), "labels")
```
7. Download the entire `models/intent_classifier/` folder from Colab to your local machine
8. Place it at `models/intent_classifier/` in your project root

Your `models/intent_classifier/` folder must contain:
```
models/intent_classifier/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── labels.json          ← important, must be present
```

### Step 6 — Build the Knowledge Base Index

This step embeds all FAQ documents and builds the FAISS vector index:

```bash
python build_index.py
```

Expected output:
```
Loaded 50 chunks
FAISS index built and saved to models/
```

This creates `models/faiss_index.bin` and `models/chunks.pkl`. Re-run this any time you add or edit files in `knowledge_base/`.

### Step 7 — Run the Chatbot

```bash
python ui/app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## ✅ Testing the Chatbot

Try these messages to verify everything works:

| Test | Message | Expected |
|---|---|---|
| Normal query | `How do I dispute a charge on my bill?` | Billing dispute instructions, no escalation |
| Password reset | `How do I reset my online banking password?` | Password reset steps |
| Fraud | `My card was stolen, what should I do?` | Card freezing instructions |
| Escalation | `I am extremely angry and demand a refund immediately!` | Escalation banner, handoff message |
| Crisis | `I want to kill myself because of these charges` | Immediate escalation |
| Technical | `Why is my mobile banking app not working?` | Troubleshooting steps |

---

## 📊 Running Evaluation

```bash
python evaluation/run_eval.py
```

This reads `evaluation/test_queries.csv`, runs each query through the full pipeline, and saves results (BLEU score, ROUGE-L score, intent, sentiment, escalation flag) to `evaluation/results.csv`.

---

## 🔧 Troubleshooting

**`ModuleNotFoundError: No module named 'faiss'`**
```bash
pip install faiss-cpu
```

**`ModuleNotFoundError: No module named 'ollama'`**
```bash
pip install ollama
```

**Ollama freezes / laptop hangs when running llama3**
Use the smaller model instead:
```bash
ollama pull llama3.2:1b
```
Update `src/llm_generator.py` to use `model="llama3.2:1b"`

**`FileNotFoundError: FAISS index not found`**
Run `python build_index.py` first.

**`FileNotFoundError: models/intent_classifier`**
Download the trained model folder from Colab and place it at `models/intent_classifier/`. See Step 5 above.

**Chatbot gives wrong/unrelated answers**
The FAISS index may be stale. Rebuild it:
```bash
python build_index.py
```

---

## 📝 Academic Note

This project was developed as a final-year group project for EC7203 — Advanced Artificial Intelligence (2026). All models and datasets used are publicly available:
- Banking77 dataset — PolyAI / HuggingFace (`mteb/banking77`)
- BERT — Google / HuggingFace (`bert-base-uncased`)
- DistilBERT SST-2 — HuggingFace (`distilbert-base-uncased-finetuned-sst-2-english`)
- all-MiniLM-L6-v2 — Sentence Transformers
- Llama 3.2 1B — Meta / Ollama