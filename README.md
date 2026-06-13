# Intelligent Customer Support Chatbot with Sentiment Routing & RAG Pipeline

A final-year AI project implementing an end-to-end customer support chatbot for the telecom/banking domain. The system combines:

- **Intent classification** — fine-tuned BERT on the Banking77 dataset (77 intent categories)
- **Sentiment analysis & escalation routing** — DistilBERT-based sentiment detection with automatic human handoff for distressed customers
- **Retrieval-Augmented Generation (RAG)** — FAISS vector search over a custom knowledge base of FAQ documents
- **Prompt engineering** — chain-of-thought and few-shot prompting for grounded, accurate responses
- **LLM backend** — local LLM via Ollama (llama3)
- **Interactive UI** — Gradio Blocks chatbot interface with intent/sentiment display and escalation alerts
- **Evaluation** — BLEU, ROUGE-L, and intent/sentiment accuracy on a test set

---

## 1. Project Structure

```
AI-Chatbot-Project-2026/
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_intent_model.ipynb       # Run on Google Colab (GPU)
│   ├── 03_rag_pipeline.ipynb
│   ├── 04_prompt_engineering.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── intent_classifier.py
│   ├── sentiment.py
│   ├── document_processor.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── prompt_templates.py
│   ├── llm_generator.py
│   └── chatbot_pipeline.py
├── knowledge_base/
│   ├── billing/
│   ├── account/
│   ├── technical/
│   └── fraud/
├── models/
│   ├── intent_classifier/          # downloaded from Colab after training
│   ├── faiss_index.bin             # generated locally
│   └── chunks.pkl                  # generated locally
├── evaluation/
│   ├── test_queries.csv
│   └── results.csv
├── ui/
│   └── app.py
├── build_index.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 2. Prerequisites

- Python 3.10+
- Git
- [Ollama](https://ollama.com/download) installed and running locally
- (For training only) Google Colab account with GPU runtime

---

## 3. Setup Instructions

### 3.1 Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Chatbot-Project-2026.git
cd AI-Chatbot-Project-2026
```

### 3.2 Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Install and set up Ollama (local LLM backend)

1. Download and install Ollama: https://ollama.com/download
2. Pull the model:
   ```bash
   ollama pull llama3
   ```
3. Ollama runs as a background service automatically after installation. No API key is required.

---

## 4. Step-by-Step: Running the Project

### Step 1 — Train the Intent Classifier (Google Colab, GPU)

1. Open `notebooks/02_intent_model.ipynb` in Google Colab.
2. Set Runtime → Change runtime type → GPU.
3. Run all cells. This fine-tunes `bert-base-uncased` on the Banking77 dataset (5 epochs).
4. After training, download the saved `models/intent_classifier/` folder from Colab.
5. Place it into your local project at `models/intent_classifier/`.

> This step only needs to be done once. The trained model is not committed to Git (see `.gitignore`) — it must be downloaded manually each time you set up the project on a new machine.

### Step 2 — Build the Knowledge Base & FAISS Index

The `knowledge_base/` folder already contains FAQ `.txt` files organized by category (billing, account, technical, fraud).

Build the vector index:

```bash
python build_index.py
```

This creates `models/faiss_index.bin` and `models/chunks.pkl`. Re-run this step any time you add or modify files in `knowledge_base/`.

### Step 3 — Test the Pipeline (optional, before launching UI)

```bash
python -c "from src.chatbot_pipeline import chat; print(chat('How do I dispute a charge on my bill?', []))"
```

### Step 4 — Launch the Chatbot UI

```bash
python ui/app.py
```

Open the local URL shown in the terminal (typically `http://127.0.0.1:7860`) in your browser.

The UI displays:
- Chat window with conversation history
- Detected **intent** for each message
- Detected **sentiment** (Positive / Neutral / Negative / Crisis)
- An **escalation alert** banner when the system routes the conversation to a human agent

### Step 5 — Run Evaluation

1. Fill in `evaluation/test_queries.csv` with columns `query,reference` (sample queries and ideal reference responses).
2. Run:
   ```bash
   python evaluation/run_eval.py
   ```
3. Results (BLEU, ROUGE-L, intent, sentiment per query) are saved to `evaluation/results.csv`.

---

## 5. Module Overview

| File | Purpose |
|---|---|
| `src/intent_classifier.py` | Loads the fine-tuned BERT model and classifies user message intent (77 Banking77 categories) |
| `src/sentiment.py` | Classifies sentiment (Positive/Neutral/Negative/Crisis) and determines escalation |
| `src/document_processor.py` | Loads and chunks knowledge base documents (512 tokens, 50 overlap) |
| `src/embedder.py` | Embeds document chunks (all-MiniLM-L6-v2) and builds the FAISS index |
| `src/retriever.py` | Retrieves top-K relevant document chunks for a given query |
| `src/prompt_templates.py` | Chain-of-thought and few-shot prompt templates |
| `src/llm_generator.py` | Calls the local Ollama LLM (llama3) to generate responses |
| `src/chatbot_pipeline.py` | Orchestrates the full pipeline: intent → sentiment → retrieval → prompt → response |
| `ui/app.py` | Gradio Blocks chatbot interface |
| `build_index.py` | One-time script to build the FAISS index from `knowledge_base/` |
| `evaluation/run_eval.py` | Computes BLEU/ROUGE-L over a test query set |

---

## 6. How the Pipeline Works

For each user message:

1. **Intent Classification** — BERT predicts the customer's intent (e.g., "card_payment_fee_charged").
2. **Sentiment Analysis** — DistilBERT scores sentiment; if Negative confidence > 0.75 or a crisis keyword is detected, `escalate = True`.
3. **Escalation Check** — If escalated, the bot responds with a handoff message and skips LLM generation.
4. **Retrieval (RAG)** — Otherwise, FAISS retrieves the top-5 most relevant knowledge base chunks for the query.
5. **Prompt Construction** — A chain-of-thought prompt is built, injecting intent, sentiment, retrieved context, and conversation history.
6. **Response Generation** — The prompt is sent to the local LLM (Ollama/llama3), which generates the final response grounded in the retrieved context.

---

## 7. Notes on Data

- **Banking77** is loaded automatically via `datasets.load_dataset("mteb/banking77")` — no manual download needed.
- **Knowledge base** (`knowledge_base/`) is self-authored FAQ content for the telecom/banking domain.
- **Evaluation test set** (`evaluation/test_queries.csv`) is self-authored, derived from the knowledge base.
- Large files (datasets, model checkpoints, FAISS index) are excluded from Git via `.gitignore` and must be regenerated or downloaded locally.

---

## 8. Troubleshooting

- **`HfUriError` / dataset loading errors**: Use `load_dataset("mteb/banking77")`, not the old `"banking77"` alias.
- **`torchvision` ImportError during training (Colab)**: Add this as the first cell in `02_intent_model.ipynb`:
  ```python
  from datasets import config
  config.TORCHVISION_AVAILABLE = False
  ```
- **`ModuleNotFoundError: ollama`**: Run `pip install ollama` and ensure the Ollama desktop app/service is running.
- **Empty retrieval results**: Ensure `build_index.py` has been run after adding files to `knowledge_base/`, and that `.txt` files are non-empty.

---

## 9. Team Roles (reference)

| Member | Responsibility |
|---|---|
| Member 1 | NLP & Data — preprocessing, BERT intent classifier, sentiment module |
| Member 2 | RAG & Backend — knowledge base, document chunking, FAISS retriever |
| Member 3 | Prompt & LLM — prompt templates, few-shot/CoT design, LLM integration |
| Member 4 | UI, Evaluation & Report — Gradio UI, evaluation framework, report compilation |

---

## 10. License & Academic Note

This project was developed as part of the Advanced Artificial Intelligence module (2026), final-year BSc Computer Engineering group project. All datasets and models used are publicly available under their respective licenses (Banking77 — HuggingFace, BERT — Google/HuggingFace, all-MiniLM-L6-v2 — Sentence-Transformers, llama3 — Meta/Ollama).
