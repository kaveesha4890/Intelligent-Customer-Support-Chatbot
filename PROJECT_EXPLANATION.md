# Intelligent Customer Support Chatbot — Complete Project Explanation

> A beginner-friendly, end-to-end guide to understanding every part of this project.
> No prior AI knowledge is assumed.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Was It Built This Way?](#2-why-was-it-built-this-way)
3. [Technology Stack & Why Each Was Chosen](#3-technology-stack--why-each-was-chosen)
4. [What Changed & Why — Migration Log](#4-what-changed--why--migration-log)
5. [System Architecture Diagram](#5-system-architecture-diagram)
6. [The Full Pipeline — Step by Step](#6-the-full-pipeline--step-by-step)
7. [Deep Dive: Core AI Concepts](#7-deep-dive-core-ai-concepts)
   - 7.1 [Intent Classification](#71-intent-classification)
   - 7.2 [Sentiment Analysis & Escalation](#72-sentiment-analysis--escalation)
   - 7.3 [Greeting Detection via Embedding Similarity](#73-greeting-detection-via-embedding-similarity)
   - 7.4 [Document Chunking](#74-document-chunking)
   - 7.5 [Embeddings — Turning Text into Numbers](#75-embeddings--turning-text-into-numbers)
   - 7.6 [ChromaDB Vector Search](#76-chromadb-vector-search)
   - 7.7 [Intent → Category Routing](#77-intent--category-routing)
   - 7.8 [Context-Enhanced Retrieval](#78-context-enhanced-retrieval)
   - 7.9 [Retrieval-Augmented Generation (RAG)](#79-retrieval-augmented-generation-rag)
   - 7.10 [Prompt Engineering — Few-Shot + Chain-of-Thought](#710-prompt-engineering--few-shot--chain-of-thought)
   - 7.11 [Local LLM via Ollama](#711-local-llm-via-ollama)
   - 7.12 [Conversation History](#712-conversation-history)
8. [Folder & File Structure](#8-folder--file-structure)
9. [What Each File Actually Does (with Code)](#9-what-each-file-actually-does-with-code)
10. [Knowledge Base](#10-knowledge-base)
11. [Assignment Technique Coverage](#11-assignment-technique-coverage)
12. [How to Run the Project](#12-how-to-run-the-project)
13. [One-Paragraph Summary](#13-one-paragraph-summary)
14. [Future Enhancements](#14-future-enhancements)
15. [Suggested Study Order](#15-suggested-study-order)
16. [What is Agentic AI?](#16-what-is-agentic-ai)
17. [What is LangChain?](#17-what-is-langchain)
18. [What is LangGraph?](#18-what-is-langgraph)
19. [Agentic AI Tools](#19-agentic-ai-tools)
20. [What is MCP (Model Context Protocol)?](#20-what-is-mcp-model-context-protocol)
21. [Agentic AI — Full Upgrade Plan for This Project](#21-agentic-ai--full-upgrade-plan-for-this-project)

---

## 1. What Is This Project?

This is an **AI-powered customer support chatbot** designed for banking and telecom-style support.

Imagine a real customer calling a helpdesk and asking:

> *"My card was charged twice. What should I do?"*

A normal chatbot would just search for keywords like "charged" and give a generic reply.

This project is **smarter**. It:
- Detects greetings and closings and responds appropriately without wasting ML processing
- Understands **what the customer actually wants** (intent classification)
- Reads the customer's **emotional tone** (sentiment analysis)
- Routes the retrieval to the **right category** based on intent
- Looks up **real support documents** from a ChromaDB vector store (retrieval)
- Constructs a **thoughtful prompt** with few-shot examples and conversation history
- Returns an **accurate, grounded answer** — or hands off to a human if needed
- Saves every conversation to **SQLite** for analytics and feedback

---

## 2. Why Was It Built This Way?

### Problem with basic chatbots

A basic rule-based chatbot only matches keywords. It breaks easily when customers write differently.

### Problem with a plain LLM

A plain large language model can "hallucinate" — it invents facts that sound convincing but are wrong. In customer support, wrong information damages trust.

### The solution: RAG + Local LLM + Smart Routing

This project combines:
- **Retrieval** — find the right documents from a ChromaDB vector store, filtered by detected intent category
- **Augmented Generation** — pass those documents to the LLM as context
- **Few-shot prompting** — show the LLM worked examples so it learns the right response style
- **Local model** — run everything on your own machine, no external API needed
- **Persistence** — every conversation saved to SQLite for analytics and improvement

---

## 3. Technology Stack & Why Each Was Chosen

| Technology | Role in this Project | Why This Choice? |
|---|---|---|
| **Python** | Main language | Standard for AI/ML; huge library support |
| **HuggingFace Transformers** | DistilBERT intent classifier | Pre-trained weights; fine-tuned on Banking77 |
| **DistilBERT (Banking77)** | Intent classification (77 classes) | 40% smaller than BERT, 2× faster, fits 2GB GPU VRAM |
| **DistilBERT (SST-2)** | Sentiment analysis | Lightweight, fast, accurate for positive/negative |
| **Banking77 Dataset (mteb/banking77)** | Intent training labels | 77 real banking support categories |
| **SentenceTransformers (MiniLM-L6-v2)** | Text embeddings for retrieval + greeting detection | Fast, accurate, 384-dim vectors |
| **ChromaDB** | Vector store for knowledge base | HNSW index, category filtering, dynamic updates, no rebuild needed |
| **SQLite** | Chat history + feedback storage | Built into Python, no server, zero setup |
| **Flask** | Web UI server | Full control over HTML/CSS/JS, supports custom feedback buttons |
| **LangChain** | Document loading & chunking | Clean utilities for splitting text files |
| **Ollama** | Local LLM runtime | Runs Llama 3.2 locally; no cloud API key |
| **Llama 3.2 (1B)** | Response generation | Small enough to run locally on 2GB GPU |
| **NLTK + rouge_score** | Evaluation metrics | Standard NLP evaluation (BLEU, ROUGE-L) |

---

## 4. What Changed & Why — Migration Log

This section documents every significant change made during development, what was used before, what replaced it, and why.

---

### 4.1 UI Framework: Gradio → Flask

| | Before | After |
|--|--------|-------|
| **What** | Gradio web UI | Flask + custom HTML/CSS/JS |
| **File** | `ui/app.py` using `gr.Blocks()` | `ui/app.py` using `Flask()` |
| **Port** | `localhost:7860` | `localhost:5000` |

**Why changed:**
Gradio is great for quick ML demos but offers limited control over the UI. Flask was chosen because:
- We needed custom feedback buttons (👍/👎) below each bot message
- We needed precise control over when buttons appear and disappear
- We needed a `/dashboard` analytics route and `/feedback` POST endpoint
- The chat UI needed session ID tracking for SQLite grouping

---

### 4.2 Vector Store: FAISS → ChromaDB

| | Before | After |
|--|--------|-------|
| **What** | FAISS binary index | ChromaDB persistent vector database |
| **Files** | `models/faiss_index.bin` + `models/chunks.pkl` | `chroma_db/` folder |
| **Module** | `src/embedder.py` + `src/retriever.py` | `src/knowledge_db.py` + `src/retriever.py` |
| **Threshold** | `0.1` (too permissive) | `0.45` (only relevant results) |

**Why changed:**
FAISS is a fast search library but has key limitations:
- Adding one new document required rebuilding the entire index
- No metadata — could not filter by category (fraud vs billing vs account)
- Binary files not human-readable or inspectable

ChromaDB solves all of these:
- Documents can be added/updated/deleted at any time without rebuilding
- Each document stores `category` metadata → filtered search per intent
- Persistent on disk, survives restarts without re-indexing
- Built-in HNSW index scales to millions of documents

**Why threshold changed from 0.1 to 0.45:**
With threshold 0.1, almost any document matched any query. "Hello" was returning phishing fraud documents because even a tiny similarity (0.12) passed. Raising to 0.45 ensures only genuinely relevant documents are returned.

---

### 4.3 Intent Classifier Base Model: BERT-base → DistilBERT

| | Before | After |
|--|--------|-------|
| **What** | `bert-base-uncased` | `distilbert-base-uncased` |
| **Size** | 110M parameters | 66M parameters (40% smaller) |
| **Speed** | Slower | ~2× faster inference |
| **VRAM** | Required >2GB for training | Fits in 2GB GPU (MX330) |
| **Accuracy** | ~95% on Banking77 | ~91% on Banking77 |

**Why changed:**
Training BERT-base on a 2GB GPU was taking 8+ hours per epoch. DistilBERT achieves 91% accuracy (vs ~95% for BERT-base) at half the size, allowing training to complete in ~20 minutes with fp16 precision on the same hardware. For a customer support chatbot the 4% accuracy trade-off is acceptable.

---

### 4.4 Greeting Detection: None → Embedding Similarity

| | Before | After |
|--|--------|-------|
| **What** | No greeting detection | Cosine similarity against prototype sentences |
| **Result** | "Hello" → phishing advice | "Hello" → proper welcome message |
| **Handles** | Nothing | "Hi", "Hi Chatbot", "Good evening sir", "Hey there" |

**Why changed:**
"Hello" and "Thank you" were being classified as banking intents (the Banking77 dataset has no greeting class), causing the retriever to pull phishing documents as the nearest match. A rule-based word list was tried first but missed variations like "Hi Chatbot". The embedding similarity approach checks if the message semantically resembles any greeting prototype sentence — catching natural variations without a hardcoded list.

```python
# Prototypes the model compares against
GREETING_PROTOTYPES = ["Hello", "Hi there", "Good morning", "Hey", "Greetings"]
CLOSING_PROTOTYPES  = ["Thank you", "Thanks", "Goodbye", "Bye", "That's all"]
# If cosine similarity > 0.75 → return canned response, skip ML pipeline
```

---

### 4.5 Chat History Storage: None → SQLite

| | Before | After |
|--|--------|-------|
| **What** | No persistence | SQLite database (`chat_history.db`) |
| **Module** | Nothing | `src/database.py` |
| **Stores** | Nothing | session_id, user_msg, bot_msg, intent, category, sentiment, escalated, feedback, timestamp |

**Why added:**
- Every conversation was lost on page refresh
- No way to analyze what questions users ask most
- No way to track which responses users rated poorly
- No escalation rate monitoring

Now every turn is saved. The `/dashboard` page shows top intents, escalation counts, and thumbs-down responses.

---

### 4.6 User Feedback: None → Smart 👍/👎 Buttons

| | Before | After |
|--|--------|-------|
| **What** | No feedback mechanism | 👍/👎 buttons on the latest bot message |
| **Behavior** | — | Buttons appear on newest reply, disappear when next message arrives |
| **Skips** | — | Greeting and unclear responses (no buttons on "Hello! How can I help?") |
| **After click** | — | Buttons replaced with "Thanks for your feedback", saved to SQLite |

**Why designed this way:**
Showing buttons on every message clutters the UI and is unrealistic. Real support tools (e.g. Zendesk, Intercom) show feedback on the most recent response. Buttons are automatically removed from the previous message when the next one arrives, so only one response can be rated at a time.

---

### 4.7 Conversation History in Prompt: None → Last 4 Turns

| | Before | After |
|--|--------|-------|
| **What** | No history in prompt | Last 4 conversation turns injected into every prompt |
| **Result before** | "I already uninstalled it" → "I don't understand" | "I already uninstalled it" → contextual answer about the app |

**Why changed:**
Follow-up messages like "I already uninstalled it" or "What if it's lost?" have no meaning without context. The LLM needs to see prior turns to understand what "it" refers to. Only the last 4 turns are included to keep the prompt short enough for the 1B parameter model.

---

### 4.8 Confidence Threshold for Unknown Intents

| | Before | After |
|--|--------|-------|
| **What** | No filtering | Return "Could you rephrase?" if confidence < 0.20 |
| **Applied to** | — | First message only — follow-ups always go to LLM |

**Why this design:**
Banking77 has 77 specific intents but doesn't cover everything (e.g. greetings, generic questions). If the first message is truly unrecognizable (score < 20%), asking for clarification is better than guessing. But once a conversation has started, follow-up messages like "How do I do that?" have very low confidence scores yet are meaningful in context — so the threshold is skipped when history exists.

---

### 4.9 Context-Enhanced Retrieval

| | Before | After |
|--|--------|-------|
| **What** | Search ChromaDB with only current message | Search with `last_user_message + current_message` |
| **Example** | "What is the email?" → matched phishing docs | "I cannot log in What is the email?" → matched login/support docs |

**Why changed:**
Short follow-up questions like "What is the email?" or "How do I do that?" are too ambiguous for ChromaDB to find the right documents. Prepending the last user message gives the search enough context to find the correct topic.

---

### 4.10 Few-Shot Examples: Empty → 4 Worked Examples

| | Before | After |
|--|--------|-------|
| **What** | `FEW_SHOT_EXAMPLES = ""` | 4 real examples covering all 4 categories |
| **Result** | Generic responses, repeated answers | Numbered steps, menu paths, contact details |

**Why added:**
The 1B parameter Llama model needs guidance on response style. Without examples, it tended to be vague or repeat itself. The 4 examples show it exactly how to structure responses: numbered steps, menu navigation paths (e.g. `Account > Cards > Freeze`), and contact information format.

---

### 4.11 Intent → Category Routing

| | Before | After |
|--|--------|-------|
| **What** | Search all 51 documents every time | Filter ChromaDB to matching category when confidence ≥ 50% |
| **Example** | "card_not_working" searched fraud + billing + account + technical | "card_not_working" searches technical only |
| **Fallback** | — | If category filter returns nothing, search all categories |

**Why added:**
When the classifier is confident about the intent (≥ 50%), we know approximately which knowledge base category is relevant. Searching only that category returns more precise results and prevents cross-topic contamination (e.g. billing documents appearing in a technical support answer).

---

### 4.12 Knowledge Base: Missing Contact Info → Added

| | Before | After |
|--|--------|-------|
| **What** | No contact/support info document | `knowledge_base/account/16_contact_support.txt` |
| **Result before** | "Our support email is [insert email address here]" | "Email: support@ourbank.com, Phone: 1-800-123-4567" |

**Why added:**
The LLM was hallucinating placeholder text because no document in the knowledge base contained actual contact information. The new file provides the specific details the LLM can quote directly.

---

### 4.13 Offline Model Loading

| | Before | After |
|--|--------|-------|
| **What** | `SentenceTransformer("all-MiniLM-L6-v2")` with network calls | Same + `os.environ["HF_HUB_OFFLINE"] = "1"` |
| **Problem** | Newer sentence_transformers tries to reach huggingface.co on every load | — |
| **Result before** | `RuntimeError: Cannot send a request` when offline | Loads from cache, no network required |

**Why changed:**
A newer version of `sentence_transformers` added a check for PEFT adapter config files on HuggingFace Hub every time a model is loaded. When the network was unavailable, this caused a crash. Setting `HF_HUB_OFFLINE=1` tells all HuggingFace libraries to use locally cached files only.

---

## 5. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER (Web Browser)                             │
│              Types a message at http://127.0.0.1:5000                  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ui/app.py  (Flask UI)                              │
│  - Custom HTML/CSS/JS chat interface                                    │
│  - 👍/👎 feedback buttons on latest bot message only                   │
│  - Shows intent, sentiment, category in metadata bar                   │
│  - /dashboard route for analytics                                       │
│  - /feedback route saves ratings to SQLite                              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  POST /chat
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  src/chatbot_pipeline.py  (Orchestrator)                │
│                                                                         │
│  Step 1: Greeting check (embedding similarity vs prototypes)            │
│          → if greeting/closing: return canned reply immediately         │
│                                                                         │
│  Step 2: Intent classification → confidence check                       │
│          → if first message and confidence < 20%: ask to rephrase      │
│                                                                         │
│  Step 3: Sentiment analysis → escalation check                          │
│          → if escalate: return human handoff message                    │
│                                                                         │
│  Step 4: Intent → category mapping (if confidence ≥ 50%)               │
│  Step 5: Context-enhanced query (prepend last user message)             │
│  Step 6: ChromaDB search (category-filtered or all)                     │
│  Step 7: Build prompt (few-shot + history + context + query)            │
│  Step 8: Ollama LLM → generate response → clean response               │
└──────┬──────────────┬──────────────┬──────────────────────────┬─────────┘
       │              │              │                          │
       ▼              ▼              ▼                          ▼
┌────────────┐ ┌────────────┐ ┌──────────────────────┐ ┌──────────────────┐
│  Intent    │ │ Sentiment  │ │   ChromaDB           │ │  LLM Generator   │
│Classifier  │ │ Analyser   │ │  Vector Store        │ │  (Ollama/Llama)  │
│            │ │            │ │                      │ │                  │
│DistilBERT  │ │DistilBERT  │ │ SentenceTransformers │ │  llama3.2:1b     │
│ Banking77  │ │ SST-2      │ │ all-MiniLM-L6-v2     │ │  local model     │
│ 77 classes │ │ + keywords │ │ HNSW cosine index    │ │                  │
│ + greeting │ │            │ │ category filtering   │ │                  │
│ embeddings │ │            │ │ 51 documents         │ │                  │
└────────────┘ └────────────┘ └──────────────────────┘ └────────────────-─┘
                                                                │
                                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      src/database.py  (SQLite)                          │
│  chat_history.db — saves every turn:                                    │
│  session_id, user_msg, bot_msg, intent, category, sentiment,            │
│  escalated, feedback (👍/👎), timestamp                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Full Pipeline — Step by Step

### Step 1 — User Sends a Message

```
"My card was charged twice. What should I do?"
```

Arrives at `ui/app.py` → `POST /chat` → calls `chat()` in `chatbot_pipeline.py`.

---

### Step 2 — Greeting / Closing Check (NEW)

`chatbot_pipeline.py` encodes the message with MiniLM and computes cosine similarity against prototype sentences:

```python
greeting_score = util.cos_sim(vec, _greeting_vecs).max().item()
# "Hello" → 0.99 → return "Hello! How can I help you today?"
# "My card was charged twice" → 0.21 → continue pipeline
```

If score > 0.75 → return canned reply immediately. No intent classifier, no ChromaDB, no LLM called.

---

### Step 3 — Intent Classification

`src/intent_classifier.py` runs the fine-tuned DistilBERT model:

```python
inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
logits  = model(**inputs).logits
pred_id = torch.argmax(softmax(logits), dim=-1).item()
# → {"intent": "transaction_charged_twice", "confidence": 0.94}
```

If confidence < 0.20 and no conversation history → ask for clarification.

---

### Step 4 — Sentiment Analysis

`src/sentiment.py` checks crisis keywords first, then runs DistilBERT SST-2:

```python
# Crisis keywords → immediate escalation
# NEGATIVE score → escalate
# POSITIVE or low-confidence NEGATIVE → continue
```

---

### Step 5 — Intent → Category Routing (NEW)

```python
# "transaction_charged_twice" → keywords: "transaction", "charge" → "billing"
category = _intent_to_category(intent, confidence)
# confidence ≥ 50% → search only "billing" documents in ChromaDB
```

---

### Step 6 — Context-Enhanced ChromaDB Search (NEW)

```python
# Combine last user message with current for better context
retrieval_query = f"{last_user_msg} {user_message}"
# "I cannot log in What is the email?" → finds login/support docs, not phishing
docs = retrieve_top_k(retrieval_query, k=3, category="billing")
```

---

### Step 7 — Prompt Construction

`src/prompt_templates.py` builds a prompt with 4 layers:

```
[4 Few-shot examples showing desired response style]
[Last 4 conversation turns for context]
[Retrieved documents from ChromaDB]
[Current user question]
[Rules: use only context, numbered steps for HOW questions, etc.]
```

---

### Step 8 — LLM Response Generation

`src/llm_generator.py` sends the prompt to Ollama:

```python
response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": prompt}])
```

---

### Step 9 — Save + Return

`ui/app.py` saves the turn to SQLite and returns the response, intent, sentiment, category, and turn_id to the browser.

---

## 7. Deep Dive: Core AI Concepts

### 7.1 Intent Classification

**What is it?**

Intent classification means figuring out the *purpose* behind a user's message.

**Analogy:** When you call a helpdesk and press "1 for billing, 2 for technical support" — that is manual intent routing. Here, AI does it automatically from free text.

**How it works here:**

1. The Banking77 dataset contains 77 real-world banking support intents (e.g., `card_not_working`, `exchange_rate`, `get_refund`).
2. A DistilBERT model was fine-tuned on thousands of labelled examples from this dataset.
3. At inference time, the model reads your message and assigns one of the 77 labels.

**Why DistilBERT (not BERT-base)?**

DistilBERT (Distilled BERT) is created by training a smaller model to mimic a larger one — a technique called **knowledge distillation**. It is 40% smaller and 2× faster than BERT-base, with ~91% of BERT's accuracy. On a 2GB GPU this matters enormously: BERT-base training took 8+ hours per epoch; DistilBERT takes ~20 minutes.

---

### 7.2 Sentiment Analysis & Escalation

**What is it?**

Sentiment analysis detects the *emotional tone* of a message: positive, neutral, negative, or crisis.

**How it works here:**

```
Text → crisis keyword check → if found: escalate immediately
     → DistilBERT SST-2 → POSITIVE or NEGATIVE score
     → explicit escalation phrases check → if found: escalate
     → ML score alone never triggers escalation (prevents false positives)
```

**Why ML score alone doesn't escalate:**

False positives in escalation are costly — routing a normal question to a human wastes agent time. Only very specific phrases like "I want to speak to a human" or "I will sue" trigger escalation, never just a NEGATIVE score.

---

### 7.3 Greeting Detection via Embedding Similarity

**What is it?**

Instead of a hardcoded word list, the system compares the user's message against prototype sentences using cosine similarity.

**Why embeddings instead of rules?**

A rule list catches "Hello" but misses "Hi Chatbot", "Good evening sir", "Hey there". An embedding-based check understands that these all mean the same thing.

```python
# Prototype sentences (stored as vectors once at startup)
GREETING_PROTOTYPES = ["Hello", "Hi there", "Good morning", "Hey", "Greetings"]

# At inference: encode user message → compare similarity
greeting_score = util.cos_sim(user_vec, prototype_vecs).max().item()
# "Hi Chatbot" → 0.82 > 0.75 threshold → return welcome message
# "My card is blocked" → 0.18 < 0.75 → continue to ML pipeline
```

This is **zero retraining** — the same MiniLM model already used for document retrieval does double duty here.

---

### 7.4 Document Chunking

**What is it?**

Chunking means splitting long documents into smaller, manageable pieces.

**How it works here:**

```python
RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
```

- Each chunk is at most **512 characters**
- Consecutive chunks share **50 characters of overlap** so no sentence is lost at a boundary

---

### 7.5 Embeddings — Turning Text into Numbers

**What is it?**

An embedding is a list of numbers (a vector) that represents the *meaning* of text.

**Analogy:** London and Paris are close on a map. Embeddings do the same for meaning: "reset my password" and "I forgot my login" produce vectors that are close together.

The model `all-MiniLM-L6-v2` converts any text into a **384-dimensional vector**:

```python
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
vector = embed_model.encode("My card was declined")
# shape: (384,) — 384 numbers representing the meaning
```

This same model is used for **two purposes** in this project:
1. Encoding knowledge base documents into ChromaDB
2. Encoding greeting prototypes for chitchat detection

---

### 7.6 ChromaDB Vector Search

**What is it?**

ChromaDB is a vector database that stores document embeddings and finds the most similar ones to a query — using a built-in HNSW (Hierarchical Navigable Small World) index.

**Why ChromaDB instead of FAISS?**

| Feature | FAISS (old) | ChromaDB (new) |
|---------|-------------|----------------|
| Add new document | Rebuild entire index | `collection.add(...)` — instant |
| Filter by category | Not possible | `where={"category": "billing"}` |
| Human-readable | Binary files | SQLite-backed, inspectable |
| Persistence | Manual save/load | Automatic |
| Scales to | Millions (fast) | Millions (fast, HNSW) |

**How it works here:**

```python
# At startup (one-time, skipped if already populated)
collection.add(
    ids=["billing/01_view_bill.txt"],
    documents=["You can view your current bill by..."],
    embeddings=[[0.12, -0.34, ...]],     # 384-dim vector
    metadatas=[{"category": "billing"}]
)

# At query time (category-filtered)
results = collection.query(
    query_embeddings=[query_vec],
    n_results=3,
    where={"category": "billing"},       # only billing documents
)
```

---

### 7.7 Intent → Category Routing

**What is it?**

Using the detected intent to narrow the ChromaDB search to only relevant documents.

**How it works:**

```python
_CATEGORY_RULES = [
    (["fraud", "compromised", "phishing", "stolen"], "fraud"),
    (["charge", "fee", "payment", "transfer", "refund"], "billing"),
    (["card", "pin", "app", "otp", "login", "declined"], "technical"),
    (["account", "password", "personal", "verify", "identity"], "account"),
]

# "transaction_charged_twice" → keywords: "charge", "transaction" → "billing"
# Only searches billing documents in ChromaDB
```

If category search returns no results (e.g. the document doesn't exist), it automatically falls back to searching all 51 documents.

---

### 7.8 Context-Enhanced Retrieval

**What is it?**

Building a richer search query by combining the last user message with the current one.

**Why needed:**

```
User: "I cannot log in even with the correct password"   ← topic established
User: "What is the email?"                               ← ambiguous alone

Without context: "What is the email?" → phishing documents (matches "email")
With context:    "I cannot log in What is the email?" → login/support documents ✓
```

```python
retrieval_query = f"{last_user_msg} {user_message}" if history else user_message
```

---

### 7.9 Retrieval-Augmented Generation (RAG)

**What is it?**

RAG is the most important concept in this project. It combines:

1. **Retrieval** — find relevant documents from ChromaDB
2. **Generation** — use those documents as context when the LLM generates a response

| Without RAG | With RAG |
|---|---|
| LLM uses only training data | LLM uses retrieved, current documents |
| May hallucinate facts | Grounded in real support documents |
| Cannot know company policies | Knows exactly what the FAQs say |

---

### 7.10 Prompt Engineering — Few-Shot + Chain-of-Thought

**Two techniques are used together:**

**1. Few-Shot Examples (NEW)**

4 worked examples are provided in every prompt, one per knowledge base category:

```
Example — Technical:
Customer: My card was declined at a store even though I have enough balance.
Assistant: A card decline despite sufficient balance is usually caused by:
1. Daily spending limit reached — check Account > Cards > Spending Limits.
2. Merchant type blocked...
3. Temporary security hold...

Example — Fraud:
Customer: I received an email asking me to click a link and enter my PIN.
Assistant: Do not click the link. This is a phishing attempt.
Our bank will never ask for your PIN via email...
```

The LLM reads these and learns: *use numbered steps, include menu paths, give contact details*.

**2. Chain-of-Thought via Rules**

The prompt's rules section guides the model's reasoning:

```
Rules:
- If the customer asks HOW to do something, provide numbered steps
- If the customer asks for contact details, provide exact details from context
- Do NOT repeat an answer already given in conversation history
- Use ONLY information from the context above
```

**Why this is Few-Shot Learning:**
The LLM is not retrained. It sees 4 examples at inference time and infers the pattern — that is the definition of few-shot in-context learning.

---

### 7.11 Local LLM via Ollama

Ollama runs `llama3.2:1b` locally. No internet, no API key, no cost per query.

```python
response = ollama.chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": prompt}],
)
```

The 1B model is guided by RAG-retrieved context and few-shot examples, so it doesn't need to "know" banking facts — it just formats the retrieved information into a clear response.

---

### 7.12 Conversation History

The last 4 conversation turns are included in every prompt:

```
Conversation so far:
Customer: My card is not working
Assistant: Your card may have been frozen. Go to Account > Cards > Freeze Card.
Customer: I already uninstalled the app
Assistant: Try reinstalling from the App Store...

Customer question: What if it still doesn't work?
```

This allows the LLM to understand follow-up messages that reference earlier context.

---

## 8. Folder & File Structure

```
Intelligent-Customer-Support-Chatbot/
│
├── knowledge_base/               ← Support FAQ documents (51 files)
│   ├── account/                  ← 16 files (added contact_support.txt)
│   ├── billing/                  ← 15 files
│   ├── technical/                ← 10 files
│   └── fraud/                    ← 10 files
│
├── src/                          ← Core chatbot logic
│   ├── chatbot_pipeline.py       ← Orchestrator (greeting check, routing, history, prompt)
│   ├── intent_classifier.py      ← DistilBERT fine-tuned on Banking77
│   ├── sentiment.py              ← Sentiment analysis + escalation logic
│   ├── retriever.py              ← ChromaDB search interface (wraps knowledge_db.py)
│   ├── knowledge_db.py           ← ChromaDB operations (NEW — replaces FAISS)
│   ├── database.py               ← SQLite chat history + feedback (NEW)
│   ├── document_processor.py     ← Loads and splits knowledge base files (LangChain)
│   ├── embedder.py               ← Legacy FAISS builder (kept for reference)
│   ├── prompt_templates.py       ← FEW_SHOT_EXAMPLES + CHAIN_OF_THOUGHT_TEMPLATE
│   └── llm_generator.py          ← Calls Ollama to generate a response
│
├── models/                       ← ML model artifacts
│   ├── intent_classifier/        ← Fine-tuned DistilBERT (config, safetensors, tokenizer)
│   ├── faiss_index.bin           ← Legacy (no longer used — replaced by ChromaDB)
│   └── chunks.pkl                ← Legacy (no longer used)
│
├── chroma_db/                    ← ChromaDB vector store (NEW — auto-created)
│   └── (binary files managed by ChromaDB)
│
├── chat_history.db               ← SQLite conversation + feedback store (NEW)
│
├── ui/
│   └── app.py                    ← Flask web UI with feedback buttons + /dashboard
│
├── evaluation/
│   ├── run_eval.py               ← Evaluation script (BLEU, ROUGE-L)
│   ├── test_queries.csv          ← Test question/answer pairs
│   └── results.csv               ← Output after running evaluation
│
├── notebooks/                    ← Jupyter notebooks (experiments)
├── train_intent_classifier.py    ← DistilBERT fine-tuning script (plain PyTorch)
└── build_index.py                ← Legacy index builder (now handled by knowledge_db.py)
```

---

## 9. What Each File Actually Does (with Code)

### `src/chatbot_pipeline.py` — The Orchestrator

```python
def chat(user_message: str, history: list):
    # 1. Greeting/closing → canned reply (no ML)
    quick_reply = _check_chitchat(user_message)
    if quick_reply:
        return quick_reply, "greeting", "Neutral", False, "all"

    # 2. Intent + confidence
    intent_result    = classify_intent(user_message)
    sentiment_result = analyse_sentiment(user_message)

    # 3. Low confidence on first message → ask to rephrase
    if intent_result["confidence"] < 0.20 and not history:
        return "I'm not sure I understood that...", "unclear", ..., False, "all"

    # 4. Context-enhanced retrieval query
    retrieval_query = f"{history[-1][0]} {user_message}" if history else user_message

    # 5. Intent → category → ChromaDB search
    category = _intent_to_category(intent_result["intent"], intent_result["confidence"])
    docs = retrieve_top_k(retrieval_query, k=3, category=category)
    if not docs and category:
        docs = retrieve_top_k(retrieval_query, k=3)   # fallback: search all

    # 6. Build prompt with few-shot + history + context
    prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(
        few_shot=FEW_SHOT_EXAMPLES, history=history_block,
        context=context, query=user_message,
    )
    response = generate_response(prompt)
    return response, intent, sentiment, escalated, category
```

---

### `src/knowledge_db.py` — ChromaDB Knowledge Base (NEW)

```python
# Populate on first run (reads all .txt files)
def populate_from_txt_files(embed_model, force=False):
    for category in ["account", "billing", "fraud", "technical"]:
        for filename in os.listdir(kb_dir/category):
            content = open(file).read()
            emb     = embed_model.encode(content, normalize_embeddings=True)
            collection.add(
                ids=[f"{category}/{filename}"],
                documents=[content], embeddings=[emb.tolist()],
                metadatas=[{"category": category}]
            )

# Category-filtered search
def retrieve_similar(query_emb, k=3, threshold=0.45, category=None):
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=k,
        where={"category": category} if category else None,
    )
    # Convert ChromaDB cosine distance (0=identical) to similarity (1=identical)
    return [doc for doc, dist in zip(results["documents"][0], results["distances"][0])
            if 1 - dist >= threshold]
```

---

### `src/database.py` — SQLite Chat History + Feedback (NEW)

```python
def init_db():
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, user_msg TEXT, bot_msg TEXT,
        intent TEXT, category TEXT, sentiment TEXT,
        escalated INTEGER, feedback INTEGER,  -- 1=👍, 0=👎
        timestamp TEXT DEFAULT (datetime('now','localtime'))
    )""")

def save_turn(session_id, user_msg, bot_msg, intent, sentiment, escalated, category) -> int:
    cur = conn.execute("INSERT INTO conversations (...) VALUES (...)")
    return cur.lastrowid   # returned to browser for feedback linking

def save_feedback(turn_id: int, rating: int):
    conn.execute("UPDATE conversations SET feedback = ? WHERE id = ?", (rating, turn_id))
```

---

### `src/retriever.py` — ChromaDB Search Interface

```python
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # use cached models, no network

embed_model = None

def init_retriever():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        init_knowledge_db()
        populate_from_txt_files(embed_model)   # no-op if already done

def retrieve_top_k(query, k=5, threshold=0.45, category=None):
    init_retriever()
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    return retrieve_similar(query_emb, k=k, threshold=threshold, category=category)
```

---

### `src/prompt_templates.py` — Few-Shot + Chain-of-Thought

```python
FEW_SHOT_EXAMPLES = """
Examples of good responses:

Customer: My card was declined at a store even though I have enough balance.
Assistant: A card decline despite sufficient balance is usually caused by:
1. Daily spending limit reached — check Account > Cards > Spending Limits.
...

Customer: I received an email asking me to click a link and enter my PIN.
Assistant: Do not click the link. This is a phishing attempt.
...
"""

CHAIN_OF_THOUGHT_TEMPLATE = """
You are a customer support assistant. Answer using ONLY the context below.

{few_shot}
{history}
Context:
{context}

Customer question: {query}

Rules:
- If customer asks HOW, provide numbered steps
- Provide exact contact details when asked
- Do not repeat answers already given in history
- Use ONLY context above, output answer directly

Answer:"""
```

---

### `ui/app.py` — Flask Web Interface

```python
# Three routes
@app.route('/')          # → serves custom HTML/CSS/JS chat page
@app.route('/chat')      # → calls chat(), saves to SQLite, returns turn_id
@app.route('/feedback')  # → saves 👍/👎 rating against turn_id
@app.route('/dashboard') # → renders analytics HTML page
@app.route('/stats')     # → returns JSON stats (total, escalated, top intents, bad responses)

# JS: only show feedback buttons on the latest bot message
lastFeedbackRow = null
# When new message arrives → remove buttons from lastFeedbackRow
# Don't show buttons for greeting/unclear intents
# After rating → replace buttons with "Thanks for your feedback"
```

---

## 10. Knowledge Base

51 plain `.txt` files organized into four topic folders:

| Folder | Files | Example Topics |
|---|---|---|
| `account/` | 16 | reset password, lost card, PIN change, joint account, **contact support (NEW)** |
| `billing/` | 15 | view bill, dispute charge, duplicate charge, late fee, autopay, refund |
| `technical/` | 10 | app issues, login, OTP, card declined, biometric login, session timeout |
| `fraud/` | 10 | report fraud, phishing, unauthorized login, identity theft, crisis support |

**New file added:** `account/16_contact_support.txt`
Contains: `support@ourbank.com`, phone `1-800-123-4567`, live chat hours, fraud hotline.
This was added because the LLM was hallucinating placeholder text (`[insert email address here]`) when asked for contact information.

---

## 11. Assignment Technique Coverage

The project satisfies **4 out of 5** required technique categories (minimum: 3):

| Requirement | Met | How |
|-------------|-----|-----|
| **NLP — Text processing** | ✓ | Tokenization, text chunking (512-char), response cleaning (`_clean_response`) |
| **NLP — Word embeddings** | ✓ | `all-MiniLM-L6-v2` → 384-dim sentence embeddings stored in ChromaDB |
| **Transformer Models — BERT** | ✓✓ | DistilBERT for intent (fine-tuned) + DistilBERT SST-2 for sentiment |
| **Transformer Models — LLM** | ✓ | Ollama `llama3.2:1b` generates final responses |
| **Transfer Learning / Fine-tuning** | ✓ | DistilBERT pre-trained weights → fine-tuned on Banking77 (77 classes) in `train_intent_classifier.py` |
| **Prompt Engineering — Systematic design** | ✓ | `CHAIN_OF_THOUGHT_TEMPLATE` with explicit rules |
| **Prompt Engineering — Chain-of-thought** | ✓ | Rules guide LLM reasoning (numbered steps, no hallucination, history awareness) |
| **Prompt Engineering — In-context (few-shot)** | ✓ | `FEW_SHOT_EXAMPLES` — 4 worked examples per category shown at inference |
| Generative AI (VAE/GAN/Diffusion) | ✗ | Not applicable to a text chatbot |

---

## 12. How to Run the Project

### Prerequisites

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install transformers torch datasets sentence-transformers
pip install langchain langchain-community chromadb flask ollama
pip install sklearn nltk rouge-score pandas

# 3. Install Ollama (https://ollama.com) and pull the model
ollama pull llama3.2:1b

# 4. Train the intent classifier (one-time, ~20 minutes on GPU)
python train_intent_classifier.py
```

### Start the app

```bash
python ui/app.py
# Open http://127.0.0.1:5000
```

ChromaDB will auto-populate from the knowledge base `.txt` files on first message. No need to run `build_index.py`.

### Analytics dashboard

```
http://127.0.0.1:5000/dashboard
```

### Files created automatically

| File / Folder | Created By | Purpose |
|---|---|---|
| `models/intent_classifier/` | `train_intent_classifier.py` | Fine-tuned DistilBERT |
| `chroma_db/` | First chat message | ChromaDB vector store |
| `chat_history.db` | App startup | SQLite conversation log |

---

## 13. One-Paragraph Summary

This project is an AI customer support chatbot for the banking domain that combines six AI techniques. First, it uses **embedding-based similarity** (MiniLM) to detect greetings and closings without any ML overhead. Second, it classifies the customer's intent using a **fine-tuned DistilBERT** model trained via transfer learning on the Banking77 dataset (77 classes). Third, it detects sentiment using a second **DistilBERT** model and escalates to a human agent if the customer sounds distressed. Fourth, the classified intent is mapped to a ChromaDB category, and a **context-enhanced query** (last message + current message combined) searches the **ChromaDB vector store** for relevant FAQ documents — this is the RAG (Retrieval-Augmented Generation) component. Fifth, a carefully engineered prompt combines **four few-shot examples** (one per category), the last 4 conversation turns, retrieved documents, and the customer's question, applying both chain-of-thought rules and in-context learning. Sixth, the prompt is sent to a local **Llama 3.2 (1B)** model running via Ollama, and the response is displayed in a **Flask** web UI with 👍/👎 feedback buttons. Every conversation turn is saved to **SQLite** for analysis on a live `/dashboard` page. The entire system runs locally with no external API dependencies.

---

## 14. Future Enhancements

### Completed During Development ✓

| Enhancement | Status |
|---|---|
| Persistent conversation sessions | ✓ Done — SQLite `chat_history.db` |
| Analytics dashboard | ✓ Done — `/dashboard` route |
| User feedback collection | ✓ Done — 👍/👎 saved to SQLite |
| ChromaDB vector store | ✓ Done — replaced FAISS |
| Category-filtered retrieval | ✓ Done — intent → category routing |
| Few-shot examples in prompt | ✓ Done — 4 examples per category |
| Conversation history in prompt | ✓ Done — last 4 turns |
| Greeting detection | ✓ Done — embedding similarity |

### Short-term Remaining

| Enhancement | What It Would Do | Effort |
|---|---|---|
| **Response streaming** | Show LLM output token-by-token instead of waiting | Low |
| **Larger Llama model** | `llama3.2:3b` or `llama3:8b` for better quality | Very low |
| **Re-ranking** | Cross-encoder to re-score ChromaDB results for precision | Low |
| **Multilingual support** | Use multilingual MiniLM for non-English queries | Medium |

### Medium-term

| Enhancement | What It Would Do | Effort |
|---|---|---|
| **Human handoff integration** | Route escalated chats to Zendesk/ticket system | High |
| **Hybrid search (BM25 + ChromaDB)** | Keyword + semantic search combined | Medium |
| **LangGraph agentic pipeline** | Replace fixed pipeline with conditional graph | Medium |
| **Active learning** | Use low-confidence predictions to improve intent model | High |

---

## 15. Suggested Study Order

| Step | File / Folder | What You Will Learn |
|---|---|---|
| 1 | `knowledge_base/account/*.txt` | What the raw support documents look like |
| 2 | `src/document_processor.py` | How documents are loaded and chunked (LangChain) |
| 3 | `src/knowledge_db.py` | How ChromaDB stores and searches documents |
| 4 | `src/retriever.py` | How a user query finds relevant chunks |
| 5 | `src/intent_classifier.py` | How DistilBERT predicts intent |
| 6 | `src/sentiment.py` | How sentiment and escalation work |
| 7 | `src/prompt_templates.py` | How few-shot + chain-of-thought prompts are structured |
| 8 | `src/chatbot_pipeline.py` | How all the above connect: greeting → intent → category → ChromaDB → prompt → LLM |
| 9 | `src/database.py` | How SQLite stores conversations and feedback |
| 10 | `src/llm_generator.py` | How Ollama generates the response |
| 11 | `ui/app.py` | How Flask serves the UI, handles feedback, and renders the dashboard |
| 12 | `train_intent_classifier.py` | How DistilBERT was fine-tuned on Banking77 |
| 13 | `evaluation/run_eval.py` | How quality is measured (BLEU, ROUGE-L) |

---

## 16. What is Agentic AI?

### The Simple Idea

A normal AI system works like a vending machine:

```
You press a button (input) → machine does one fixed thing → result comes out
```

An **Agentic AI** works more like a human employee:

```
You give it a goal → it figures out what steps to take → it uses tools →
it checks the result → it decides what to do next → it tries again if needed
```

**The most important difference:**

| Normal AI (pipeline) | Agentic AI |
|---|---|
| Follows a fixed sequence of steps | Decides its own steps based on the situation |
| Cannot change its behavior mid-run | Can loop, retry, or change direction |
| Cannot use external tools | Can call tools: search, database, APIs |
| One input → one output | Goal in → series of actions → final answer |

---

### 16.1 How an Agent Thinks — The ReAct Loop

```
THOUGHT: "The user wants to reset their card PIN."
    │
    ▼
ACTION: Call tool → search_knowledge_base("PIN reset")
    │
    ▼
OBSERVATION: "To reset PIN, go to Settings > Card >..."
    │
    ▼
THOUGHT: "I have enough info. I can answer now."
    │
    ▼
FINAL ANSWER: "To reset your PIN, go to Settings..."
```

---

### 16.2 Agentic vs Non-Agentic

This project currently uses a **non-agentic pipeline**. Every step always runs in the same order. An agentic version would let the LLM decide which steps to run based on the question.

### 16.3 Types of AI Agents

| Agent Type | How It Decides | Best For |
|---|---|---|
| **ReAct Agent** | Reason then act in a loop | General-purpose question answering |
| **Tool-calling Agent** | LLM picks from a list of tools | Specific external APIs |
| **Plan-and-Execute Agent** | Makes a full plan first | Complex multi-step tasks |
| **LangGraph Agent** | State machine with conditional branching | Workflows with clear decision points |

---

## 17. What is LangChain?

LangChain is a Python framework for building applications with language models. It provides ready-made building blocks.

**Already used in this project:**

```python
# src/document_processor.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

**How it could replace the entire pipeline:**

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | Ollama(model="llama3.2:1b")
    | StrOutputParser()
)
response = rag_chain.invoke("My card was charged twice")
```

---

## 18. What is LangGraph?

LangGraph builds stateful, multi-step workflows as graphs — nodes connected by conditional edges.

**How it would replace `src/chatbot_pipeline.py`:**

```
         START
           │
    ┌─────────────┐
    │ intent_node │
    └──────┬──────┘
           │
    ┌──────────────────┐
    │  sentiment_node  │
    └──────────────────┘
           │
     ┌─────┴──────┐
  escalate     normal
     │            │
     ▼            ▼
┌──────────┐ ┌───────────────┐
│ escalate │ │ retrieval_node│
│   node   │ └───────┬───────┘
└────┬─────┘         │
     │               ▼
     │       ┌───────────────┐
     │       │ generate_node │
     │       └───────┬───────┘
     └───────┬────────┘
            END
```

Each node is a Python function that reads and updates a shared state dictionary. Conditional edges route to `escalate_node` or `retrieval_node` based on the sentiment result.

---

## 19. Agentic AI Tools

A **tool** is a function that an AI agent can choose to call. The agent reads the tool description and decides whether to use it.

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the customer support FAQ documents for relevant information."""
    docs = retrieve_top_k(query, k=5)
    return "\n\n".join(docs) if docs else "No relevant documents found."

@tool
def create_support_ticket(issue_description: str, priority: str = "normal") -> str:
    """Create a support ticket for issues requiring human follow-up."""
    ticket_id = f"TKT-{hash(issue_description) % 100000:05d}"
    return f"Ticket {ticket_id} created. A human agent will contact you within 24 hours."
```

The LLM reads the **docstring** to decide when to call each tool.

---

## 20. What is MCP (Model Context Protocol)?

MCP is an open standard by Anthropic that defines how AI models connect to external data sources — like USB standardized device connections.

```
Without MCP: AI App ──custom code──→ Database
             AI App ──custom code──→ File system
             AI App ──custom code──→ External API

With MCP:    AI App ──MCP──→ MCP Server for Database
             AI App ──MCP──→ MCP Server for File system
             (same protocol, plug and play)
```

**MCP Server for this project:**

```python
from mcp.server import Server
server = Server("customer-support")

@server.list_tools()
async def list_tools():
    return [Tool(name="search_knowledge_base", description="...", inputSchema={...})]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "search_knowledge_base":
        results = retrieve_top_k(arguments["query"])
        return [TextContent(type="text", text="\n\n".join(results))]
```

Any MCP-compatible client (Claude Desktop, etc.) can then use your knowledge base as a tool without custom integration code.

---

## 21. Agentic AI — Full Upgrade Plan for This Project

| Stage | What It Adds | Technology |
|---|---|---|
| **1 (current)** | Smart pipeline with ChromaDB, SQLite, few-shot, category routing | Flask + ChromaDB + SQLite |
| **2** | Conditional routing graph (escalate vs answer vs clarify) | LangGraph |
| **3** | LLM picks which tool to call dynamically | LangChain Tools + `@tool` decorator |
| **4** | Knowledge base becomes a reusable external service | MCP Server |
| **5** | Multiple specialized agents (FAQ agent, account agent, escalation agent) | Multi-Agent LangGraph |

> The current project is a strong foundation. Stage 2 (LangGraph) is achievable in a single afternoon and would make the routing logic visual and easy to extend.

---

*This document reflects the complete current state of the project including all improvements made during development. A reader with no prior AI background should be able to understand the full system after reading it.*
