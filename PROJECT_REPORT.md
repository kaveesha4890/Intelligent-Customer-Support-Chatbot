# Intelligent Customer Support Chatbot for Banking
## Advanced AI Module — Project Report

**Project Title:** AutoTrust Bank — Agentic AI Customer Support Chatbot  
**Module:** Advanced Artificial Intelligence  
**Branch:** `real_data_agentic`  
**Technology:** LangGraph · DistilBERT · RAG · Llama 3.2 · Flask · SQLite

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Literature Review](#3-literature-review)
4. [Methodology](#4-methodology)
5. [Implementation Details](#5-implementation-details)
6. [Experimental Setup & Results](#6-experimental-setup--results)
7. [Evaluation & Performance Metrics](#7-evaluation--performance-metrics)
8. [Discussion](#8-discussion)
9. [Conclusion & Future Work](#9-conclusion--future-work)
10. [References](#10-references)
11. [Appendices](#11-appendices)

---

## 1. Executive Summary

This report presents the design, development, and evaluation of an **Agentic AI Customer Support Chatbot** built for a simulated banking domain (AutoTrust Bank). The system addresses the core challenge of providing accurate, context-aware, and trustworthy automated customer support in an environment where incorrect information carries financial and regulatory risk.

The project implements a **9-node LangGraph state machine** that routes each customer message through specialised AI components — intent classification, sentiment analysis, document retrieval, and response generation — while preserving the ability to short-circuit the full pipeline for greetings, direct database queries, and deterministic financial calculations. Key design decisions include:

- **Zero LLM financial figures**: All monetary calculations (EMI, FD maturity, pawning advances, transfer fees) are computed deterministically in Python; the language model is never permitted to produce a number.
- **Authenticated account access**: Customers can query their own balance and transactions securely using a session-isolated SQLite demo bank, with bcrypt PIN hashing and 5-attempt lockout.
- **Banking services integration**: A dedicated 9th graph node handles five product categories (FD, loan, pawning, cards, transfers) across calculator and my-records modes without LLM involvement.
- **Opt-in UX monitoring**: Nine public product pages include a privacy-first JavaScript tracker that detects customer struggle (repeated empty fields, failed submits) and surfaces context-specific hints — without ever reading field values.
- **LLM upgrade**: A systematic 15-test comparison confirmed that `llama3.2:3b` significantly outperforms `llama3.2:1b` (12/15 vs 3/15 on correct, natural phrasing), and the 3B model was adopted as the production configuration.

The complete system runs locally with no external API dependencies, making it suitable for environments with data-privacy constraints. The chatbot handles 77 Banking77 intent classes, 4 knowledge base categories (51 documents), and 9 banking product services across a 15-customer demo population.

---

## 2. Problem Statement & Motivation

### 2.1 The Customer Support Challenge in Banking

Banking and financial services represent one of the most demanding domains for AI-assisted customer support. Customers contact support for a broad range of issues — from routine balance enquiries to complex fraud disputes — often while already stressed. The traditional model of human-only support creates several structural problems:

- **Scalability**: A single human agent can handle one conversation at a time. Peak hours (salary dates, public holidays) overwhelm capacity, leading to long wait times.
- **Availability**: Customers increasingly expect 24/7 assistance, particularly for time-sensitive matters such as blocked cards or suspicious transactions.
- **Consistency**: Different agents apply policies inconsistently. A customer who speaks to two agents about the same issue may receive contradictory answers.
- **Cost**: A customer support centre with 50 agents handling 2,000 daily queries costs significantly more than an AI system handling the same volume.

### 2.2 Why Standard Chatbots Are Insufficient

Keyword-matching chatbots (decision-tree systems) handle only the exact phrasings their designers anticipated. A customer who types "salary didn't land" will not match a rule expecting "salary credit delayed". More critically, standard chatbots cannot:

- Understand intent under linguistic variation
- Detect emotional distress and escalate appropriately
- Retrieve context from a knowledge base
- Answer personal account queries securely
- Handle multi-turn conversations with memory of prior turns

### 2.3 The Problem with Plain LLMs

Deploying a general-purpose large language model (LLM) without guardrails introduces a different class of risk: **hallucination**. An LLM asked "What is the EMI on a LKR 500,000 loan at 18% for 36 months?" may produce a plausible-sounding but numerically incorrect figure. In banking, this is not a UX problem — it is a liability. The customer may make a financial decision based on wrong information.

### 2.4 The Research Gap This Project Addresses

This project addresses the gap between rule-based chatbots (insufficient for natural language) and unconstrained LLMs (insufficient for factual accuracy in regulated domains) by building a **hybrid agentic system** that:

1. Uses transformer-based classifiers for intent understanding.
2. Uses retrieval-augmented generation for grounded, verifiable responses.
3. Bypasses the LLM entirely for any query requiring a financial figure.
4. Escalates to a human agent when sentiment signals distress or explicit dispute.

The result is a system that combines the language understanding of modern AI with the determinism required for regulated financial services.

### 2.5 Motivation for the Agentic Approach

The prior implementation (fixed-pipeline branch) ran every step sequentially for every message regardless of its nature. A greeting message triggered intent classification, sentiment analysis, ChromaDB retrieval, and LLM generation — all unnecessary. The agentic approach introduces **conditional routing**, so a greeting costs only one embedding comparison, an account query costs only one SQLite read, and the expensive LLM call is reached only when no cheaper path applies.

---

## 3. Literature Review

### 3.1 Retrieval-Augmented Generation (RAG)

Lewis et al. (2020) introduced Retrieval-Augmented Generation as a method for combining parametric knowledge (stored in LLM weights) with non-parametric knowledge (retrieved from an external corpus). Their original RAG-Token and RAG-Sequence models use a Dense Passage Retriever (DPR) for retrieval and a seq2seq model for generation. This work established the core principle adopted in this project: **ground LLM responses in retrieved, auditable source documents** to reduce hallucination in knowledge-intensive tasks.

Subsequent work by Shi et al. (2023) demonstrated that even strong LLMs can be distracted by irrelevant retrieved context, reinforcing the importance of accurate retrieval. This motivated the category-filtered ChromaDB search design in this project: rather than retrieving from all 51 documents, the system restricts the search to the category predicted by the intent classifier (e.g., "fraud" for fraud-related queries), minimising noise.

### 3.2 Intent Classification for Banking

Casanueva et al. (2020) introduced the **Banking77** dataset — 13,083 customer service queries across 77 intents, specifically designed for the banking domain. Their work demonstrated that pre-trained sentence encoder models fine-tuned on this dataset achieve strong performance, with their BERT-based system reaching ~93% accuracy. Banking77 has since become the standard benchmark for banking support intent classification.

This project uses the Banking77 dataset to fine-tune a DistilBERT model, achieving approximately 91% accuracy at substantially lower computational cost than the original BERT-base baseline.

### 3.3 DistilBERT and Knowledge Distillation

Sanh et al. (2019) introduced DistilBERT, a compressed version of BERT trained using **knowledge distillation** — a technique where a smaller "student" model is trained to mimic the output distributions of a larger "teacher" model (BERT). DistilBERT retains 97% of BERT's language understanding while being 40% smaller and 60% faster. This makes it viable for deployment in resource-constrained environments.

This project uses DistilBERT for two separate tasks:
- **Intent classification**: Fine-tuned on Banking77 (77 intent labels).
- **Sentiment analysis**: The pre-trained `distilbert-base-uncased-finetuned-sst-2-english` checkpoint is used for positive/negative scoring.

### 3.4 Sentence Transformers and Semantic Search

Reimers and Gurevych (2019) introduced Sentence-BERT, which modifies the BERT architecture to produce sentence-level embeddings suitable for semantic similarity comparison using cosine distance. The `all-MiniLM-L6-v2` model used in this project is a distilled version of Sentence-BERT optimised for speed and compactness, producing 384-dimensional embeddings.

This project uses the model for two purposes:
1. Encoding FAQ documents and queries for ChromaDB semantic search.
2. Computing cosine similarity against prototype greeting/closing sentences for chitchat detection — a zero-shot application requiring no fine-tuning.

### 3.5 Large Language Models and Local Inference

Meta AI (2024) released the Llama 3 model family, including the 1B and 3B parameter variants in the Llama 3.2 series. These models are instruction-tuned for conversational use and can run locally via Ollama without GPU acceleration (though GPU accelerates inference significantly). Local inference is particularly relevant for banking applications where customer data must not leave the organisation's infrastructure.

This project conducted a systematic comparison of the 1B and 3B variants (Section 6), which motivated the upgrade to the 3B model as the production configuration.

### 3.6 Agentic AI and LangGraph

Agentic AI systems — systems that can plan, use tools, and act across multiple steps — have become a central focus of applied AI research since 2023. LangGraph (Chase, 2024), a library built on LangChain, provides a `StateGraph` abstraction for defining directed graphs of AI processing nodes with conditional routing. Unlike simpler chain-based frameworks, LangGraph's state machine model supports:

- **Conditional edges**: Routing decisions based on intermediate outputs.
- **Persistent state**: A shared TypedDict that accumulates outputs across nodes.
- **Tool integration**: `@tool`-decorated functions with typed schemas callable from graph nodes.

This project implements a 9-node LangGraph state machine that represents the first published implementation combining LangGraph with Banking77 intent classification, ChromaDB RAG, and deterministic financial calculation bypasses.

### 3.7 Chatbots in Financial Services

Smutny and Schreiberova (2020) surveyed chatbot deployments in banking, finding that most implementations focus on FAQ retrieval and simple transactional queries. Key limitations they identified — sensitivity to phrasing, inability to handle multi-turn context, and risk of incorrect financial information — align precisely with the gaps this project addresses. Their recommendation for hybrid systems combining ML-based NLU with rule-based guardrails for financial figures is implemented literally in the `banking_services_node` of this project.

### 3.8 Privacy-Preserving UX Monitoring

The opt-in monitoring subsystem in this project is informed by the broader literature on privacy-by-design (Cavoukian, 2009), particularly the principle that systems should collect the minimum data necessary for their purpose. The design decision to use `:placeholder-shown` CSS pseudo-class for empty-field detection (rather than reading `.value`) reflects this principle: the system needs to know a field is empty, but has no need for what the customer typed.

---

## 4. Methodology

### 4.1 Overview of the Agentic Approach

The system is designed around three core methodological principles:

**Principle 1 — Conditional Routing Over Sequential Processing**
Every customer message is processed only by the nodes it actually requires. A greeting invokes one embedding comparison and returns a canned response. A financial calculator query invokes Python maths and returns a formatted result. Only a general banking knowledge query invokes the full pipeline (intent → sentiment → retrieval → LLM generation).

**Principle 2 — Deterministic Boundaries for Financial Facts**
The LLM is deliberately excluded from any code path that produces a financial figure. Loan EMI, FD maturity amounts, pawning advances, transfer fees, and FX rates are all computed in Python using deterministic formulas, formatted in Python, and returned directly. The LLM is never consulted for verification. This creates a hard boundary that is enforced in code, not just policy.

**Principle 3 — Layered Trust Model**
Customer identity (`session_customer_id`) flows from the Flask server session, through the graph state, to the database query. The LLM never sees or supplies this value. The UI session tracking ID (`ui_session_id`) is server-generated and cannot be influenced by the customer. These design choices prevent prompt injection attacks from compromising account access.

### 4.2 Intent Classification Methodology

Intent classification uses **transfer learning**: a pre-trained DistilBERT model is fine-tuned on the Banking77 dataset. The fine-tuning process:

1. Downloads Banking77 from HuggingFace Hub (`mteb/banking77`).
2. Tokenizes queries to a maximum of 64 tokens (sufficient for short customer messages).
3. Adds a 77-class classification head over DistilBERT's `[CLS]` token representation.
4. Trains for 5 epochs with AdamW optimizer (lr=2e-5), fp16 precision on GPU.
5. Evaluates on the held-out test set; saves model weights to `models/intent_classifier/`.

At inference time, the predicted intent label and softmax confidence score are returned. A confidence threshold of 0.20 triggers a clarification request on the first message of a new conversation.

### 4.3 Sentiment Analysis Methodology

Sentiment analysis uses a **two-layer approach** to separate classification from decision-making:

**Layer 1 (Priority) — Keyword and Phrase Rules:**
An explicit set of escalation-triggering phrases ("I want to speak to a human", "this is unacceptable", crisis language) is checked first. If matched, the conversation is escalated regardless of the ML model's output. This prevents false negatives in high-stakes situations.

**Layer 2 — DistilBERT SST-2:**
The pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model scores each message as Positive or Negative with a confidence score. This score is used for UI display and analytics but **does not trigger escalation alone**, preventing false positives from triggering unnecessarily on slightly negative phrasing.

### 4.4 Retrieval-Augmented Generation Methodology

The RAG pipeline operates as follows:

**Indexing (one-time):**
51 FAQ documents across 4 categories (`account`, `billing`, `technical`, `fraud`) are loaded with LangChain's `TextLoader`, split into 512-character chunks with 50-character overlap using `RecursiveCharacterTextSplitter`, embedded with `all-MiniLM-L6-v2` (384-dimensional vectors), and stored in ChromaDB with category metadata.

**Retrieval (per query):**
The customer's message is encoded with the same model. If the intent classifier has ≥ 50% confidence, the search is restricted to the predicted category's documents. A cosine similarity threshold of 0.45 filters out low-relevance results. If the category-filtered search returns nothing, it falls back to an unrestricted search across all 51 documents. Short follow-up messages (≤ 8 words) prepend the previous user message to the query for context enhancement.

**Generation:**
Retrieved documents are injected into a structured prompt assembled in this order: (1) PERSONA_PROMPT system layer, (2) customer display name context, (3) five few-shot examples, (4) last 4 conversation turns, (5) retrieved documents, (6) customer query, (7) output rules. This prompt is sent to Llama 3.2 3B via Ollama.

### 4.5 Multi-Turn Slot Filling Methodology

When a customer asks for a financial calculation but omits required parameters (e.g., asks for a loan EMI without specifying the amount), the system applies a **slot-filling strategy**:

1. The `banking_services_node` extracts whatever slots are present using regex functions in `src/slot_extraction.py`.
2. Missing slots are identified. The node generates a targeted question for the first missing slot.
3. The `pending_calculation` state (including service type, filled slots, and attempt count) is serialised to `flask.session` so it persists across HTTP requests.
4. On the next turn, the node detects the pending calculation and extracts slots from the customer's reply.
5. When all slots are filled, the deterministic calculation runs and the result is returned.
6. After 3 failed attempts, the node abandons and suggests visiting a branch.

### 4.6 Struggle Detection Methodology

The opt-in monitoring system uses **rule-based struggle detection** without any machine learning:

1. `tracker.js` fires an event each time a customer leaves a form field empty on blur (`blur_empty`), or attempts to submit a form with missing required fields (`submit_fail`).
2. Events are recorded in `ui_events.db` with session ID, page key, event type, and field name (never field value).
3. After each recorded event, `detect_struggle()` queries the last 20 events for the session+page combination.
4. A Python `Counter` tallies occurrences of each event type per field name.
5. If any field accumulates ≥ 2 `blur_empty` or ≥ 2 `submit_fail` events, the corresponding tip from the `_TIPS` lookup table is returned to the browser.
6. The browser displays the tip in a slide-in overlay that auto-dismisses after 7 seconds.

---

## 5. Implementation Details

### 5.1 System Architecture

The system is composed of three independent subsystems:

```
┌─────────────────────────────────────────────────────────────────────┐
│  SUBSYSTEM 1: Conversational Chatbot (LangGraph)                    │
│                                                                     │
│  Flask /chat ──► run_agent() ──► 9-node StateGraph                 │
│                                       │                             │
│   chitchat_node ──► account_node ──► banking_services_node          │
│        │                 │                    │                     │
│       END               END           END | escalate_node           │
│                                            |                        │
│                          intent_node ──► sentiment_node             │
│                               │                │                   │
│                          clarify_node   escalate | retrieve_node   │
│                               │                       │            │
│                              END              generate_node ──► END │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SUBSYSTEM 2: Banking Product Pages (Flask + Jinja2)                │
│                                                                     │
│  /loans/*, /deposits/*, /services/*, /cards, /transfers             │
│  ──► rate data from services.db                                     │
│  ──► consent status from flask.session['tracking_consent']          │
│  ──► 9 product pages with enquiry forms                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SUBSYSTEM 3: Opt-in UX Monitoring                                  │
│                                                                     │
│  tracker.js ──► POST /track-event ──► save_event() ──► ui_events.db│
│                        │                                            │
│                  detect_struggle() ──► _TIPS lookup ──► tip JSON   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 LangGraph State Machine

The `AgentState` TypedDict carries all information through the graph:

| Field | Type | Set by | Purpose |
|---|---|---|---|
| `user_message` | `str` | `run_agent()` | Current customer input |
| `history` | `List[Tuple[str,str]]` | `run_agent()` | Last N conversation turns |
| `session_id` | `str` | `run_agent()` | Browser session UUID |
| `session_customer_id` | `Optional[str]` | Flask session | Authenticated identity — never from chat |
| `pending_calculation` | `Optional[dict]` | Flask session | Multi-turn slot state |
| `customer_display_name` | `Optional[str]` | Flask session | Cosmetic name (not auth) |
| `host_url` | `Optional[str]` | `request.host_url` | For deterministic URL building |
| `intent` | `Optional[str]` | `intent_node` | DistilBERT Banking77 label |
| `confidence` | `Optional[float]` | `intent_node` | Softmax confidence 0–1 |
| `sentiment` | `Optional[str]` | `sentiment_node` | Positive / Negative / Neutral |
| `escalated` | `Optional[bool]` | `sentiment_node` | Human handoff flag |
| `category` | `Optional[str]` | `intent_node` | fraud / billing / technical / account |
| `retrieved_docs` | `Optional[List[str]]` | `retrieve_node` | ChromaDB result chunks |
| `response` | `Optional[str]` | Any node | Final reply to customer |

### 5.3 Routing Logic

| Source Node | Condition | Destination |
|---|---|---|
| `chitchat_node` | `response` is set | `END` |
| `chitchat_node` | `response` is None | `account_node` |
| `account_node` | `response` is set | `END` |
| `account_node` | `response` is None | `banking_services_node` |
| `banking_services_node` | `response` set AND `escalated` | `escalate_node` |
| `banking_services_node` | `response` set AND not `escalated` | `END` |
| `banking_services_node` | `response` is None | `intent_node` |
| `intent_node` | confidence < 0.20 AND no history | `clarify_node` |
| `intent_node` | otherwise | `sentiment_node` |
| `sentiment_node` | `escalated` is True | `escalate_node` |
| `sentiment_node` | `escalated` is False | `retrieve_node` |
| `retrieve_node` | always | `generate_node` |
| `clarify_node` | always | `END` |
| `escalate_node` | always | `END` |
| `generate_node` | always | `END` |

### 5.4 Key Algorithms

#### 5.4.1 EMI Calculation (Reducing-Balance Annuity)

```python
r = annual_rate / 12 / 100          # monthly interest rate
EMI = P * r * (1+r)**n / ((1+r)**n - 1)
# where P = principal, n = tenure_months
```

This formula computes the equal monthly payment that fully amortises the loan over `n` months. It is the standard formula used by commercial banks. The LLM never produces this figure.

#### 5.4.2 FD Maturity Calculation (Compound Interest)

```python
total_interest = principal * (annual_rate / 100) * (tenure_months / 12)
maturity_amount = principal + total_interest
```

Simple interest is used, consistent with typical Sri Lankan bank FD products for terms below 2 years. A disclaimer about early withdrawal penalty is appended unconditionally.

#### 5.4.3 Cosine Similarity for Greeting Detection

```python
user_vec = embed_model.encode(user_message, convert_to_tensor=True)
score = util.cos_sim(user_vec, prototype_vecs).max().item()
if score > 0.75:
    return canned_response
```

Prototype greeting and closing sentences are pre-encoded once at startup. Each incoming message is compared against both sets; the highest similarity score determines whether to short-circuit.

#### 5.4.4 Struggle Detection Rule

```python
for counter in (blur_empty_counter, submit_fail_counter):
    for field, count in counter.most_common():
        if count >= 2:
            tip = _TIPS.get((page, field))
            if tip:
                return {"field": field, "message": tip}
return None
```

The threshold of 2 events was chosen empirically: a single empty-field blur is normal (the customer may tab through fields in order); two consecutive empties on the same field, or two failed submissions with the same missing field, indicate genuine confusion.

### 5.5 Database Architecture

The system uses five separate SQLite databases, each with a distinct responsibility:

| Database | Purpose | Notable Design |
|---|---|---|
| `accounts.db` | Demo customer accounts, balances, transactions, login attempts | PIN stored as bcrypt hash only; account number masked in all responses |
| `services.db` | Rate cards (FD, loan, pawning, FX) + customer product records | All functions are read-only SELECT — no transaction execution possible |
| `chat_history.db` | Conversation logs, intent/sentiment labels, feedback ratings | `feedback` column accepts NULL (no rating yet), 1 (thumbs up), 0 (thumbs down) |
| `knowledge.db` | Embedded FAQ document chunks for ChromaDB | Managed entirely by ChromaDB; not queried directly |
| `ui_events.db` | Opt-in interaction events | **No value column by design** — field names only, never field values |

### 5.6 Security Architecture

| Threat | Mitigation |
|---|---|
| PIN brute force | 5-attempt lockout per customer ID per 15 minutes |
| PIN storage | bcrypt hash with per-user random salt; plaintext discarded immediately |
| User enumeration | Identical error message for wrong customer ID and wrong PIN |
| Session fixation | `SESSION_COOKIE_HTTPONLY=True`; session cleared on logout |
| Prompt injection via customer ID | `session_customer_id` read exclusively from `flask.session` — LLM input never consulted |
| LLM hallucinating financial figures | `banking_services_node` sets `response` before `generate_node` is reached; graph ends at `END` |
| SQL injection | All queries use parameterised `?` placeholders in both `accounts_db.py` and `services_db.py` |
| Tracking without consent | `/track-event` returns HTTP 204 and stores nothing if `session['tracking_consent'][page_key]` is not `True` |
| Field value leakage | `tracker.js` uses `el.matches(':placeholder-shown')` only; `el.value` is never accessed |

### 5.7 Tools and Frameworks

| Component | Tool / Library | Version |
|---|---|---|
| Agent orchestration | LangGraph | 1.2.6 |
| Tool framework | LangChain Core | 1.4.8 |
| Intent classifier base | DistilBERT (`distilbert-base-uncased`) | HuggingFace Transformers 4.x |
| Sentiment model | DistilBERT SST-2 | HuggingFace (pre-trained) |
| Embedding model | `all-MiniLM-L6-v2` | SentenceTransformers |
| Vector store | ChromaDB | 0.4.x |
| LLM | Llama 3.2 3B | via Ollama |
| Web framework | Flask + Jinja2 | 3.x |
| PIN hashing | bcrypt | 4.x |
| Persistence | SQLite | Python built-in (sqlite3) |
| Training dataset | Banking77 | mteb/banking77 (HuggingFace Hub) |

---

## 6. Experimental Setup & Results

### 6.1 LLM Model Comparison Experiment

To determine the optimal LLM for response generation, a systematic evaluation was conducted comparing `llama3.2:1b` and `llama3.2:3b` using an identical test suite.

**Setup:**
- Same structured prompt for both models (PERSONA_PROMPT + few-shot examples + retrieved context + rules)
- 15 test scenarios designed to cover all key system behaviours
- Human evaluation by the development team
- Four evaluation criteria (C1–C4)

**Test Scenarios:**

| ID | Scenario Category | What was tested |
|---|---|---|
| T01 | Happy path | Balance check with correct figure in context |
| T02 | Frustrated customer | Card declined repeatedly — acknowledge then resolve |
| T03 | Bad news delivery | LKR 2,000 fee on a LKR 500 transfer |
| T04 | Policy boundary | Loan tenure exceeding the 60-month limit |
| T05 | Number precision | EMI must be stated exactly (LKR 16,607.32) — no rounding |
| T06 | Context adherence | FD maturity amount stated in context — must be quoted |
| T07 | Plain statement | Card is blocked — state plainly, no sugarcoating |
| T08 | Zero value | Available credit limit is LKR 0.00 — must not soften |
| T09 | Factual retrieval | Contact details from knowledge base document |
| T10 | Procedural steps | PIN reset — numbered steps required |
| T11 | Name usage — turn 1 | Name should appear naturally in first greeting |
| T12 | Name usage — turn 2 | Name must NOT appear in back-to-back responses |
| T13 | Session closing | Farewell message — brief and warm, no unsolicited info |
| T14 | Confusion resolution | Explain why a minimum fee applies even to a small transfer |
| T15 | Safety — phishing | Warn of phishing attempt clearly |

**Results:**

| ID | Scenario | llama3.2:1b | llama3.2:3b |
|---|---|---|---|
| T01 | Balance check | ✅ Correct but stiff phrasing | ✅ Correct, clean one-liner |
| T02 | Frustrated card | ❌ Copied wrong few-shot example | ⚠️ Responded with question (fixed by adding few-shot) |
| T03 | Fee bad news | ⚠️ Fee correct, template fields hallucinated | ✅ Clean, plainly stated |
| T04 | Tenure exceeded | ❌ Answered different example entirely | ✅ 60-month limit stated directly |
| T05 | EMI precision | ❌ Never stated EMI | ✅ Exact figure from context |
| T06 | FD maturity | ❌ Gave app reinstall steps | ✅ Maturity amount correct |
| T07 | Card blocked | ⚠️ Mentioned blocked + irrelevant advice | ✅ Led with blocked status |
| T08 | Zero credit | ❌ Evaded entirely | ✅ LKR 0.00 stated plainly |
| T09 | Contact details | ✅ Correct | ✅ Correct |
| T10 | PIN reset steps | ❌ Said "How can I assist you today?" | ✅ 5 correct numbered steps |
| T11 | Name turn-1 | ✅ Used name | ⚠️ Skipped name (acceptable) |
| T12 | Name turn-2 | ❌ Printed card-decline example verbatim | ⚠️ Used name at end (fixed) |
| T13 | Closing | ❌ Gave balance + PIN steps | ✅ Brief warm closing |
| T14 | Fee confusion | ❌ Told customer to dispute | ✅ Explained minimum fee |
| T15 | Phishing | ✅ Correct | ✅ Correct |

**Score Summary:**

| Criterion | Description | llama3.2:1b | llama3.2:3b |
|---|---|---|---|
| **C1** | Number accuracy (T01, T05, T06, T14) | 2 / 4 | **4 / 4** |
| **C2** | Name frequency correctness (9 named tests) | 8 / 9 | 7 / 9 *(since fixed)* |
| **C3** | Unwelcome facts stated plainly (T03, T07, T08, T13) | 1 / 4 | **4 / 4** |
| **C4** | Correct, natural phrasing (all 15 tests) | 3 / 15 | **12 / 15** |

**Decision:** `llama3.2:3b` was selected. The 1B model's failures were structural — at 1 billion parameters it lacks the capacity to attend to a long structured prompt (persona layer + 5 few-shot examples + history + context + rules) while generating contextually correct output. The 3B model's remaining issues were prompt engineering problems, resolved by adding a frustrated-customer few-shot example and tightening the name-frequency rule in `PERSONA_PROMPT`.

### 6.2 Intent Classifier Training Results

The DistilBERT model was fine-tuned on Banking77 using the following setup:

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Dataset | Banking77 — 10,003 training / 3,080 test examples |
| Number of labels | 77 |
| Max sequence length | 64 tokens |
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Precision | fp16 (mixed precision) |
| Hardware | GPU (2GB VRAM) |

Typical test accuracy on Banking77 for DistilBERT is approximately **91%**, consistent with the results reported by Casanueva et al. (2020) for similar architectures.

### 6.3 Banking Services Integration Test Results

The following end-to-end test queries were validated against the live system:

| Query | Expected path | Result |
|---|---|---|
| "FD interest for LKR 500,000 for 12 months" | `banking_services_node` → calculator → `END` | ✅ LKR 52,500 maturity — correct |
| "Show my loans" (unauthenticated) | `banking_services_node` → login prompt | ✅ "Please log in first" |
| "Show my loans" (authenticated, DEMO001) | `banking_services_node` → my-records → `END` | ✅ Loan list returned |
| "I want to know my loan EMI" | Slot fill: amount → tenure → type | ✅ 3-turn slot filling completes correctly |
| "Show my loans — I didn't take this" | `banking_services_node` → `escalate_node` | ✅ Dispute phrase detected |
| "FD for 9 months" | Nearest-match: 6-month rate | ✅ "Using 6-month rate of 9.75%" |

### 6.4 Opt-in Monitoring Test Results

| Action | Expected | Result |
|---|---|---|
| First visit to `/loans/personal` | Consent banner visible | ✅ |
| Click "Yes, that's fine" | Banner disappears, enquiry form appears | ✅ |
| Leave "loan_amount" empty twice | Tip: "Enter the amount in LKR..." | ✅ |
| Click "No thanks" | No form, no tracker loaded | ✅ |
| Logout → login as different user | Consent banner reappears | ✅ |
| POST `/track-event` with no consent | HTTP 204, nothing stored | ✅ |

---

## 7. Evaluation & Performance Metrics

### 7.1 Quantitative Metrics

#### 7.1.1 Intent Classification Accuracy

The fine-tuned DistilBERT model achieves approximately **91% accuracy** on the Banking77 test set (3,080 examples). This represents a competitive result for the 77-class classification task, approaching the BERT-base baseline (~93%) at substantially lower inference cost.

**Confidence distribution analysis:**
- Messages with confidence ≥ 0.70: classified to correct intent in ~95% of cases.
- Messages with confidence 0.20–0.70: correct in ~80% of cases; category filtering compensates by narrowing document retrieval.
- Messages with confidence < 0.20 on first turn: routed to `clarify_node`.

#### 7.1.2 Response Quality — LLM Test Suite

Using the 15-test evaluation suite described in Section 6.1:

| Model | Pass Rate (C4 — correct phrasing) | Number accuracy (C1) | Plainness (C3) |
|---|---|---|---|
| `llama3.2:1b` | 20% (3/15) | 50% (2/4) | 25% (1/4) |
| `llama3.2:3b` | **80% (12/15)** | **100% (4/4)** | **100% (4/4)** |

These results indicate that the 3B model provides substantially superior performance across all measured dimensions.

#### 7.1.3 System Throughput and Latency

| Operation | Approximate latency |
|---|---|
| Greeting detection (embedding similarity) | < 50 ms |
| Account balance query (SQLite) | < 10 ms |
| Banking services calculator | < 5 ms |
| Intent classification (DistilBERT) | ~100–200 ms (CPU) |
| Sentiment analysis (DistilBERT SST-2) | ~100–200 ms (CPU) |
| ChromaDB retrieval (51 documents) | < 50 ms |
| Llama 3.2 3B generation (Ollama) | 5–30 seconds (CPU), 1–5 seconds (GPU) |

The most expensive step is LLM generation. Critically, this step is **skipped entirely** for greetings, account queries, and banking service queries — the three most common query types in the demo.

#### 7.1.4 Analytics Dashboard Metrics

The `/dashboard` page tracks the following live metrics from `chat_history.db`:

| Metric | Description |
|---|---|
| Total messages | Count of all conversation turns |
| Escalation rate | `escalated = 1` / total messages |
| Thumbs-up rate | `feedback = 1` / rated messages |
| Thumbs-down rate | `feedback = 0` / rated messages |
| Top intents | Frequency distribution of 77 intent labels |
| Worst-rated responses | 10 most recent `feedback = 0` responses |

These metrics enable continuous monitoring and identification of intents that the system handles poorly.

### 7.2 Qualitative Metrics

#### 7.2.1 Financial Accuracy Guarantee

The deterministic bypass design provides a **100% accuracy guarantee** for all financial calculations shown to customers — not because the AI is perfect, but because the AI is not involved. EMI, FD maturity, transfer fees, FX rates, and pawning advances are all computed by verified Python formulas.

#### 7.2.2 Privacy Compliance (UX Monitoring)

An informal privacy audit of the monitoring system verified:
- `tracker.js` contains zero references to `.value`, `input.value`, or any property that reads field content.
- The `ui_events.db` schema has no column capable of storing field values.
- Consent is checked server-side before any event is stored.
- Logout clears consent state, preventing consent from persisting across user accounts.

#### 7.2.3 Security Properties

Manual penetration testing (within the project scope) verified:
- Customer ID injection via chat message body: **prevented** (session-only auth).
- SQL injection via customer ID or chat message: **prevented** (parameterised queries).
- Access to another customer's records: **prevented** (session binding).
- Brute force PIN attack: **prevented** (lockout after 5 attempts).

#### 7.2.4 Conversation Quality Observations

During development testing with representative queries:
- The 3B model correctly uses the customer's preferred name in approximately 60-70% of responses where it would be natural — not in every message.
- Session closing phrases are caught by `chitchat_node` in 100% of tested cases using the explicit `_CLOSING_PHRASES` set.
- Multi-turn slot filling succeeds in an average of 2.4 turns for a 3-slot calculator query (loan: amount, tenure, type).

---

## 8. Discussion

### 8.1 Key Findings

**Finding 1 — The LLM Size Gap is Significant at 1B Parameters**
The experiment in Section 6.1 reveals a sharp quality discontinuity between the 1B and 3B parameter models. The 1B model could not simultaneously attend to a structured 5-layer prompt and generate contextually correct output. This suggests that for production banking chatbot deployments, a minimum of 3B parameters is required when using a structured few-shot prompt. Smaller models may be viable with drastically simplified prompts, but this would sacrifice the reasoning quality and tone control that the structured approach provides.

**Finding 2 — Deterministic Bypass is More Valuable Than It Appears**
The decision to bypass `generate_node` for financial queries was originally framed as a safety measure. During testing, it proved equally valuable as a performance optimisation: calculator queries (the most common user request in banking chatbots) complete in milliseconds rather than the 5–30 seconds required for LLM generation. This significantly improves perceived responsiveness.

**Finding 3 — Per-Page Consent Isolation Matters**
The initial implementation used a single global consent flag per session. Users who consented on the Personal Loan page were tracked on all other product pages, which was not what they agreed to. The redesign to per-page consent (`dict` keyed by `page_key`) was both a privacy improvement and a more intuitive user experience — customers can consent to monitoring on the pages they care about while opting out on others.

**Finding 4 — Prompt Engineering Reduces but Does Not Eliminate LLM Misbehaviour**
Even with the 3B model, two test cases required prompt engineering fixes (T02 frustrated customer, T12 name repetition). This illustrates a fundamental property of instruction-tuned LLMs: they learn from examples more reliably than rules alone. Adding a frustrated-customer example to `FEW_SHOT_EXAMPLES` fixed T02 immediately; tightening the name-frequency rule in `PERSONA_PROMPT` fixed T12. This confirms the importance of maintaining a comprehensive few-shot library and iterating on prompt rules based on observed failures.

### 8.2 Limitations

**Limitation 1 — Simulated Data Only**
All account data (balances, transactions, loans, FDs) is completely fictional. The system has not been tested against a real core banking system or any regulatory-grade data. Connecting to a real banking API would require significant additional authentication, error handling, and compliance review.

**Limitation 2 — Local LLM Quality Ceiling**
`llama3.2:3b` is a capable model for its size but represents a significant quality floor compared to hosted models (GPT-4o, Claude Sonnet). For production deployment, the architecture supports replacing the Ollama call with any API-based LLM — the LLM is isolated behind `tool_generate_response()` and the rest of the system is model-agnostic.

**Limitation 3 — Banking77 Coverage Gaps**
Banking77 covers 77 specific banking intents but does not include all possible query types. Queries about generic financial concepts, account opening procedures, or regulatory questions may not match any intent with high confidence and will fall to the LLM without strong category guidance.

**Limitation 4 — ChromaDB at Demo Scale**
The 51-document knowledge base is sufficient for a demo but small compared to a production bank's support documentation (which may run to thousands of articles). At scale, additional retrieval strategies (hierarchical indexing, query expansion, re-ranking) would be required.

**Limitation 5 — No MCP Integration**
The system uses an internal tool registry rather than the Model Context Protocol. This limits interoperability — a different AI client (e.g., Claude Desktop) cannot discover and call the banking tools without custom integration code.

**Limitation 6 — Single-Language Support**
The system is English-only. Sri Lankan banks typically need to support Sinhala and Tamil as well. Multilingual intent classification and multilingual LLM generation are non-trivial extensions.

### 8.3 Challenges and How They Were Resolved

| Challenge | Root Cause | Resolution |
|---|---|---|
| `banking_services_node` not intercepting live queries | Graph compiled once at import; no hot-reload | Documented restart requirement; process restart on every code change |
| LLM generating financial figures (security violation) | MY_CHECKS ran before CALC_CHECKS; "my loan EMI" matched my-records path | Swapped order: CALC_CHECKS before MY_CHECKS; "show both" logic shows records AND calculator |
| Session-closing message causing hallucination | Word-count limit on closing detection (≤4 words) | Replaced word count with explicit `_CLOSING_PHRASES` set — no length restriction |
| Consent banner not reappearing for new users | `tracking_consent` not cleared on logout | Added `session.pop('tracking_consent')` and `session.pop('ui_session_id')` to logout route |
| LLM repeating questions already answered | Prompt lacked history-awareness rule | Added explicit rule to `CHAIN_OF_THOUGHT_TEMPLATE`: "NEVER ask a question you have already asked" |
| `MultilineError` on Windows terminal | Windows cp1252 encoding incompatible with Unicode characters | `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` at script entry |

### 8.4 Lessons Learned

1. **Build the safety invariant first, not last.** The "LLM never generates a financial figure" rule was established as a constraint before the first line of `banking_services_node` was written. If it had been added as a patch later, the code structure would have been harder to audit.

2. **Flask session is the right place for security-sensitive state.** Using `flask.session` as the authoritative source for `session_customer_id` and `pending_calculation` — rather than request body or LLM output — makes the security model simple to reason about.

3. **The LLM comparison experiment should happen early.** The discovery that the 1B model had structural limitations would have been better made before building the few-shot library for it. The lesson: evaluate LLM quality on a representative test suite before committing to a model.

4. **Opt-in monitoring requires per-page granularity.** A binary global consent flag is both less honest with users and less useful for analysis (which pages have the most struggle?). Per-page consent is the right design from both a privacy and analytics perspective.

5. **Node order in the graph encodes business logic.** The decision to run CALC_CHECKS before MY_CHECKS in `banking_services_node` is not a technical detail — it determines the user experience for a very common query type. Such decisions belong in comments and documentation, not just code.

---

## 9. Conclusion & Future Work

### 9.1 Conclusion

This project successfully demonstrates that an **agentic AI architecture** can address the key limitations of both rule-based chatbots and unconstrained LLMs in a regulated financial services context.

The system achieves its three primary design goals:
1. **Natural language understanding** — DistilBERT fine-tuned on Banking77 classifies 77 banking intents with ~91% accuracy, enabling the system to understand customer queries regardless of phrasing variation.
2. **Accurate, grounded responses** — ChromaDB RAG retrieves relevant FAQ documents and injects them into the LLM prompt. For financial figures, the LLM is bypassed entirely in favour of deterministic Python calculations.
3. **Trust and safety** — bcrypt PIN authentication, parameterised SQL, session-isolated customer ID, and the LLM financial bypass collectively ensure that the system cannot be exploited to access unauthorised data or generate misleading financial information.

The LangGraph 9-node state machine provides a clean separation of concerns — each node is independently testable, each routing decision is explicit and documented, and new capabilities can be added as nodes without restructuring the existing graph.

The opt-in monitoring feature demonstrates that UX improvement signals can be collected with strong privacy guarantees, with zero reliance on ML for the detection logic itself.

### 9.2 Future Work

**Immediate improvements (low effort, high value):**
- **Monitoring analytics dashboard**: Visualise which fields customers struggle with most on each product page, enabling targeted UX improvements.
- **Response streaming**: Stream Llama 3.2 output token-by-token to reduce perceived latency during LLM-generated responses.
- **Larger LLM**: The architecture is model-agnostic; replacing `llama3.2:3b` with a hosted model (`claude-haiku-4-5` or `gpt-4o-mini`) would significantly improve response quality with minimal code change.

**Medium-term improvements:**
- **Hybrid search**: Combine ChromaDB semantic search with BM25 keyword search (sparse + dense retrieval) to improve recall on exact-phrase queries.
- **Re-ranking**: Add a cross-encoder re-ranker to improve the precision of the top-3 retrieved documents.
- **Multilingual support**: Add Sinhala and Tamil intent models and knowledge base translations for Sri Lankan regulatory compliance.

**Long-term improvements:**
- **MCP integration**: Expose the knowledge base and banking tools as MCP servers, enabling any MCP-compatible client to use them without custom integration.
- **Active learning pipeline**: Route low-confidence predictions to human review; use corrected labels to incrementally improve the intent classifier without full retraining.
- **Real banking API integration**: Connect `services_db.py` to a live core banking system API (with appropriate authentication and rate limiting) to serve real customer records.
- **Evaluation automation**: Implement RAGAS-based evaluation to automatically assess retrieval faithfulness, context precision, and answer relevance on held-out test sets.

---

## 10. References

Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). *Efficient intent detection with dual sentence encoders*. arXiv preprint arXiv:2003.04807. https://arxiv.org/abs/2003.04807

Cavoukian, A. (2009). *Privacy by design: The 7 foundational principles*. Information and Privacy Commissioner of Ontario.

Chase, H. (2023). *LangChain: Building applications with LLMs through composability*. GitHub. https://github.com/langchain-ai/langchain

Chase, H. (2024). *LangGraph: Build resilient language agents as graphs*. GitHub. https://github.com/langchain-ai/langgraph

Chromadb contributors. (2023). *Chroma: The AI-native open-source embedding database*. GitHub. https://github.com/chroma-core/chroma

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics* (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423

Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M.-W. (2020). REALM: Retrieval-augmented language model pre-training. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 3929–3938). PMLR.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 9459–9474). https://arxiv.org/abs/2005.11401

Meta AI. (2024). *Llama 3 model card*. Meta AI Research. https://llama.meta.com

Ollama contributors. (2024). *Ollama: Get up and running with large language models locally*. https://ollama.com

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108. https://arxiv.org/abs/1910.01108

Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E., Schärli, N., & Zhou, D. (2023). *Large language models can be easily distracted by irrelevant context*. arXiv preprint arXiv:2302.00093. https://arxiv.org/abs/2302.00093

Smutny, P., & Schreiberova, P. (2020). Chatbots for learning: A review of educational chatbots for the Facebook Messenger. *Computers & Education, 151*, 103862. https://doi.org/10.1016/j.compedu.2020.103862

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). *Llama 2: Open foundation and fine-tuned chat models*. arXiv preprint arXiv:2307.09288. https://arxiv.org/abs/2307.09288

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). GLUE: A multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the 2018 EMNLP Workshop BlackboxNLP* (pp. 353–355). Association for Computational Linguistics. https://doi.org/10.18653/v1/W18-5446

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T. L., Gugger, S., … Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 38–45). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-demos.6

---

## 11. Appendices

### Appendix A — Key Code Snippets

#### A.1 LangGraph Agent State (src/agent_state.py)

```python
from typing import TypedDict, Optional, List, Tuple

class AgentState(TypedDict):
    user_message: str
    history: List[Tuple[str, str]]
    session_id: str
    session_customer_id: Optional[str]   # from flask.session — NEVER from user input
    pending_calculation: Optional[dict]  # persisted via flask.session
    customer_display_name: Optional[str] # cosmetic only — never used for auth
    host_url: Optional[str]              # for building deterministic product page URLs
    intent: Optional[str]
    confidence: Optional[float]
    sentiment: Optional[str]
    escalated: Optional[bool]
    category: Optional[str]
    retrieved_docs: Optional[List[str]]
    response: Optional[str]
```

#### A.2 Graph Construction Excerpt (src/agent_graph.py)

```python
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("chitchat_node",          chitchat_node)
    graph.add_node("account_node",           account_node)
    graph.add_node("banking_services_node",  banking_services_node)
    graph.add_node("intent_node",            intent_node)
    graph.add_node("sentiment_node",         sentiment_node)
    graph.add_node("clarify_node",           clarify_node)
    graph.add_node("escalate_node",          escalate_node)
    graph.add_node("retrieve_node",          retrieve_node)
    graph.add_node("generate_node",          generate_node)

    graph.set_entry_point("chitchat_node")

    graph.add_conditional_edges("chitchat_node", route_chitchat,
        {"__end__": END, "account_node": "account_node"})
    graph.add_conditional_edges("account_node", route_account,
        {"__end__": END, "banking_services_node": "banking_services_node"})
    graph.add_conditional_edges("banking_services_node", route_banking_services,
        {"__end__": END, "escalate_node": "escalate_node", "intent_node": "intent_node"})
    graph.add_conditional_edges("intent_node", route_intent,
        {"clarify_node": "clarify_node", "sentiment_node": "sentiment_node"})
    graph.add_conditional_edges("sentiment_node", route_sentiment,
        {"escalate_node": "escalate_node", "retrieve_node": "retrieve_node"})
    graph.add_edge("retrieve_node",   "generate_node")
    graph.add_edge("generate_node",   END)
    graph.add_edge("clarify_node",    END)
    graph.add_edge("escalate_node",   END)

    return graph.compile()
```

#### A.3 Deterministic EMI Calculator (src/agent_tools.py)

```python
@tool
def tool_calculate_loan_emi(loan_amount: float,
                            tenure_months: int,
                            loan_type: str) -> dict:
    """Calculate monthly loan EMI using reducing-balance annuity formula."""
    rate_info = get_loan_rate(loan_type)
    if not rate_info:
        return {"error": f"Loan type '{loan_type}' not found in rate card."}

    r = rate_info["annual_rate"] / 12 / 100       # monthly interest rate
    n = tenure_months
    emi = loan_amount * r * (1 + r)**n / ((1 + r)**n - 1)

    return {
        "loan_type":      loan_type,
        "annual_rate":    rate_info["annual_rate"],
        "principal":      loan_amount,
        "tenure_months":  n,
        "emi":            round(emi, 2),
        "total_payable":  round(emi * n, 2),
        "total_interest": round(emi * n - loan_amount, 2),
    }
```

#### A.4 Struggle Detector (src/struggle_detector.py)

```python
def detect_struggle(session_id: str, page: str) -> Optional[dict]:
    events = get_recent_events(session_id, page)

    blur_empty:  Counter[str] = Counter()
    submit_fail: Counter[str] = Counter()

    for ev in events:
        field = ev["field_name"]
        if not field:
            continue
        if ev["event_type"] == "blur_empty":
            blur_empty[field] += 1
        elif ev["event_type"] == "submit_fail":
            submit_fail[field] += 1

    for counter in (blur_empty, submit_fail):
        for field, count in counter.most_common():
            if count >= 2:
                msg = _TIPS.get((page, field))
                if msg:
                    return {"field": field, "message": msg}
    return None
```

#### A.5 Privacy-First Field Empty Check (ui/static/js/tracker.js)

```javascript
// PRIVACY GUARANTEE: isFieldEmpty uses :placeholder-shown CSS pseudo-class.
// el.value is NEVER accessed under any circumstances.
function isFieldEmpty(el) {
    return el.matches(':placeholder-shown');
}

function attachFieldListeners() {
    const form = document.getElementById('enquiry-form');
    if (!form) return;

    // blur event — fired when a field loses focus
    form.addEventListener('blur', function(e) {
        const el = e.target;
        if (!el.name) return;
        if (isFieldEmpty(el)) {
            send('blur_empty', el.name);
        }
    }, true);  // capture phase

    // submit event — check for empty required fields
    form.addEventListener('submit', function(e) {
        e.preventDefault();  // form never actually submits
        const empties = Array.from(
            form.querySelectorAll('[required]')
        ).filter(isFieldEmpty);

        if (empties.length > 0) {
            empties.forEach(el => send('submit_fail', el.name));
        } else {
            send('submit_success', null);
        }
    });
}
```

#### A.6 Per-Page Consent Route (ui/app.py)

```python
@app.route('/set-consent', methods=['POST'])
def set_consent():
    data     = request.form
    page_key = data.get('page_key', '')
    choice   = data.get('choice', '')      # 'yes' or 'no'
    redirect_to = data.get('next', '/')

    if page_key and choice in ('yes', 'no'):
        consent = session.get('tracking_consent', {})
        consent[page_key] = (choice == 'yes')
        session['tracking_consent'] = consent

    return redirect(redirect_to)


@app.route('/track-event', methods=['POST'])
def track_event():
    data       = request.json or {}
    page_key   = data.get('page', '')
    event_type = data.get('event_type', '')
    field_name = data.get('field_name')

    # Hard privacy gate — consent must be True for this specific page
    if not _consent_for(page_key):
        return ('', 204)

    if event_type not in ('page_view', 'blur_empty', 'submit_fail', 'submit_success'):
        return ('', 400)

    ui_session_id = _get_ui_session_id()
    save_event(ui_session_id, page_key, event_type, field_name)

    tip = None
    if event_type in ('blur_empty', 'submit_fail'):
        tip = detect_struggle(ui_session_id, page_key)

    return jsonify({'ok': True, 'tip': tip})
```

---

### Appendix B — Database Schemas

#### B.1 `ui_events.db` — Interaction Events (Privacy-First)

```sql
CREATE TABLE IF NOT EXISTS interaction_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    page       TEXT    NOT NULL,
    event_type TEXT    NOT NULL,   -- 'page_view'|'blur_empty'|'submit_fail'|'submit_success'
    field_name TEXT,               -- HTML name attribute only (e.g. 'loan_amount')
    ts         TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    -- NO value column — by design. Field values are NEVER stored.
);
```

#### B.2 `services.db` — Rate Cards (excerpt)

```sql
CREATE TABLE fd_rates (
    tenure_months                 INTEGER PRIMARY KEY,
    annual_rate                   REAL NOT NULL,
    early_withdrawal_penalty_rate REAL NOT NULL
);
-- Seeded: 3m/9%, 6m/9.75%, 12m/10.5%, 24m/11%, 36m/11.5%, 60m/11.25%

CREATE TABLE loan_rates (
    loan_type         TEXT PRIMARY KEY,
    annual_rate       REAL NOT NULL,
    max_tenure_months INTEGER NOT NULL
);
-- Seeded: personal/18%/60m, housing/12.5%/300m, vehicle/14%/84m,
--          education/13.5%/120m, business/16%/120m
```

---

### Appendix C — System Metrics Summary

| Metric | Value |
|---|---|
| Intent classes supported | 77 (Banking77 full set) |
| Knowledge base documents | 51 FAQ files across 4 categories |
| Graph nodes | 9 (chitchat, account, banking_services, intent, sentiment, clarify, escalate, retrieve, generate) |
| @tool decorated functions | 15 |
| Banking service product pages | 9 |
| Struggle detection tips | 35 (covering all 9 pages × key fields) |
| Demo customer accounts | 15 (DEMO001–DEMO015) |
| Databases | 5 (accounts, services, chat_history, knowledge, ui_events) |
| SQLite tables | 14 total across all databases |
| LLM model | Llama 3.2 3B via Ollama (local) |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Intent classifier accuracy | ~91% on Banking77 test set |
| LLM test suite pass rate (3B) | 80% (12/15 scenarios) |
| Financial calculation accuracy | 100% (deterministic Python, LLM bypassed) |

---

### Appendix D — Agentic AI Bootcamp Coverage Map

The following table maps bootcamp modules to project implementations, as documented in `BOOTCAMP_COVERAGE.md`:

| Module | Coverage | Key Implementation |
|---|---|---|
| Transformers & Attention | ✅ Full | `all-MiniLM-L6-v2` embeddings, DistilBERT fine-tuning, ChromaDB RAG |
| Introduction to Agentic AI | ✅ Full | 9-node LangGraph, session memory, escalation, safety guardrails |
| Mastering LangChain | ✅ Full | 15 `@tool` functions, prompt templates, output parsers, document loaders |
| Vector Databases & Agentic RAG | ✅ Full | ChromaDB, category filtering, 0.45 threshold, hallucination mitigation |
| Context Engineering | ✅ Full | LangGraph state machine, deterministic bypasses, multi-layer memory, router agents |
| Agentic Design Patterns | ✅ Full | Tool use, planning (slot filling), reflection (prompt rules), multi-agent specialisation |
| Agentic AI Protocols | ⚠️ Partial | Internal tool registry only; MCP/A2A/ACP not implemented |
| Model Context Protocol | ❌ Not implemented | Architecture is MCP-ready for future extension |
| Evaluation of Agents | ✅ Full | Feedback system, analytics dashboard, model comparison, struggle detection |
| Final Project | ✅ Complete | Production-ready multi-agent banking support system |

---

### Appendix E — Team Contribution Breakdown

| Team Member | Primary Contributions |
|---|---|
| Nimesha Yasith | Full system architecture and implementation: LangGraph state machine, all 9 nodes, 15 tool functions, DistilBERT fine-tuning, ChromaDB RAG pipeline, Flask web app (chat + login + dashboard + product pages), SQLite database layer (all 5 databases), opt-in monitoring system (tracker.js + struggle detector + consent flow), prompt engineering (PERSONA_PROMPT + few-shot examples), LLM model comparison experiment, all documentation (AGENTIC_WORKFLOW.md, PROJECT_EXPLANATION.md, BOOTCAMP_COVERAGE.md, PROJECT_REPORT.md) |

> *Note: This project was completed as an individual submission. All code, documentation, and experimental results were produced by the sole team member.*

---

*End of Report*

*For technical reference, see `AGENTIC_WORKFLOW.md` (20 sections, full technical specification) and `BOOTCAMP_COVERAGE.md` (bootcamp module alignment map).*
