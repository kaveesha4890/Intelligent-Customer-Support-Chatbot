
# Intelligent Customer Support Chatbot — Agentic AI Workflow

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Agent State](#4-agent-state)
5. [Agent Graph — Full Workflow](#5-agent-graph--full-workflow)
6. [Node Descriptions](#6-node-descriptions)
7. [Tool Definitions](#7-tool-definitions)
8. [RAG Pipeline](#8-rag-pipeline-retrieval-augmented-generation)
9. [Account Banking Feature](#9-account-banking-feature)
10. [Banking Services Feature](#10-banking-services-feature)
11. [Conversational UX Features](#11-conversational-ux-features)
12. [Security Model](#12-security-model)
13. [Database Schemas](#13-database-schemas)
14. [Flask API Endpoints](#14-flask-api-endpoints)
15. [ML Models](#15-ml-models)
16. [How to Run](#16-how-to-run)
17. [Demo Credentials](#17-demo-credentials)
18. [Challenges, Debugging & Fixes](#18-challenges-debugging--fixes)
19. [LLM Model Comparison — llama3.2:1b vs llama3.2:3b](#19-llm-model-comparison--llama321b-vs-llama323b)

---

## 1. System Overview

This system is an **Agentic AI Customer Support Chatbot** built for a banking domain. It uses a **LangGraph state machine** to route each customer message through a series of specialised AI nodes — each node handles one focused task (greeting detection, intent classification, sentiment analysis, document retrieval, response generation).

Unlike a fixed pipeline where every message goes through every step in order, the agentic graph makes **conditional routing decisions** at each node. A greeting skips the entire ML pipeline. A distressed customer is immediately escalated without hitting retrieval. An account balance query is answered directly from the database without touching the LLM at all.

### Key Capabilities
| Capability | How it works |
|---|---|
| Greeting / closing detection | Embedding similarity (all-MiniLM-L6-v2) |
| Acknowledgement handling ("okay", "got it") | Keyword matching on short messages |
| Session-closing detection ("ok thanks, that's all") | Explicit phrase set — bypasses LLM entirely |
| Continuation handling ("yes" after bot closing) | Context-aware keyword matching |
| Customer display name collection | First-message prompt → regex extraction → Flask session persistence |
| Warm tone / persona | PERSONA_PROMPT system layer injected into every generate_node call |
| Intent classification | Fine-tuned DistilBERT on Banking77 (77 classes) |
| Sentiment analysis + escalation | DistilBERT SST-2 + explicit keyword rules |
| Knowledge retrieval | ChromaDB + sentence embeddings (RAG) |
| Response generation | Local Llama 3.2 3B via Ollama |
| Account balance / transactions | SQLite query on authenticated session |
| FD / loan / pawning / transfer / FX calculators | Deterministic Python math — LLM never generates figures |
| "Show both" for logged-in calculator queries | Personal records prepended above calculator result |
| My-records (FDs, loans, pawning, cards) | SQLite query on authenticated session (read-only) |
| Dispute detection | Phrase matching → escalated=True → escalate_node |
| Human escalation | Triggered by negative sentiment / crisis keywords / account disputes |
| Currency | All monetary figures shown in **LKR** throughout |

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph 1.2.6 (StateGraph) |
| Tool framework | LangChain Core 1.4.8 (`@tool` decorator) |
| Web framework | Flask |
| Intent classifier | DistilBERT fine-tuned on Banking77 |
| Sentiment analyser | DistilBERT SST-2 (HuggingFace) |
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector store | ChromaDB (local, persistent) |
| LLM | **Llama 3.2 3B** via Ollama *(upgraded from 1B — see Section 19)* |
| Account database | SQLite (`accounts.db`) |
| Banking services database | SQLite (`services.db`) |
| Chat history database | SQLite (`chat_history.db`) |
| PIN hashing | bcrypt |
| Session management | Flask server-side session (signed cookie) |
| Python version | 3.13.2 (venv) |

---

## 3. Project Structure

```
Intelligent-Customer-Support-Chatbot/
│
├── ui/
│   └── app.py                  # Flask app — routes, login UI, chat UI, dashboard
│
├── src/
│   ├── agent_state.py          # AgentState TypedDict — shared data bag for the graph
│   ├── agent_graph.py          # LangGraph graph definition — nodes, edges, run_agent()
│   ├── agent_tools.py          # @tool decorated functions called by graph nodes
│   ├── prompt_templates.py     # PERSONA_PROMPT + few-shot + chain-of-thought template
│   ├── accounts_db.py          # SQLite operations for banking demo (auth, balance, txns)
│   ├── services_db.py          # SQLite operations for rate cards + customer product records
│   ├── slot_extraction.py      # Regex-based slot extractors (amount, tenure, carat, etc.)
│   ├── intent_classifier.py    # DistilBERT Banking77 inference
│   ├── sentiment.py            # DistilBERT SST-2 + keyword escalation rules
│   ├── retriever.py            # ChromaDB retrieval using sentence embeddings
│   ├── knowledge_db.py         # ChromaDB init and similarity search
│   ├── llm_generator.py        # Ollama Llama 3.2 3B call
│   └── database.py             # Chat history SQLite (save_turn, save_feedback, stats)
│
├── models/
│   └── intent_classifier/      # Fine-tuned DistilBERT weights (generated by train script)
│
├── knowledge_base/             # .txt FAQ documents loaded into ChromaDB
│
├── seed_accounts.py            # Seeds accounts.db with 15 demo customers
├── seed_services.py            # Seeds services.db with rate cards + customer products
├── train_intent_classifier.py  # Fine-tunes DistilBERT on Banking77
├── build_index.py              # Builds ChromaDB index from knowledge_base/ files
│
├── accounts.db                 # SQLite — demo accounts & transactions (simulated)
├── services.db                 # SQLite — rate cards, FDs, loans, pawning, cards (simulated)
└── chat_history.db             # SQLite — conversation logs & feedback
```

---

## 4. Agent State

The `AgentState` TypedDict is the **shared data bag** that flows through every node in the graph. Each node reads from it and writes back to it.

```python
# src/agent_state.py

class AgentState(TypedDict):
    # ── Inputs set by run_agent() before graph starts ──────────────────
    user_message:         str                    # the customer's current message
    history:              List[Tuple[str, str]]  # [(user, bot), ...] last N turns
    session_id:           str                    # browser session UUID
    session_customer_id:  Optional[str]          # from Flask session — NEVER from user input
    pending_calculation:  Optional[dict]         # multi-turn slot state — from Flask session

    # Display name the customer chose to be called (e.g. "Alice").
    # IMPORTANT: PURELY COSMETIC — used only for warmth in LLM responses.
    # NEVER used for authentication, account lookups, or any security check.
    # Kept completely separate from session_customer_id.
    customer_display_name: Optional[str]         # from Flask session — cosmetic only

    # ── Set by individual nodes as the graph executes ──────────────────
    intent:               Optional[str]          # e.g. "card_not_working" | "fd_calculator"
    confidence:           Optional[float]        # intent classifier confidence 0–1
    sentiment:            Optional[str]          # "Positive" | "Negative" | "Neutral" | "Crisis"
    escalated:            Optional[bool]         # True → route to human agent
    category:             Optional[str]          # "fraud" | "billing" | "banking_services" | etc.
    retrieved_docs:       Optional[List[str]]    # top-k chunks from ChromaDB
    response:             Optional[str]          # final reply shown to customer
```

### Key design notes
- `session_customer_id` — the authenticated security identity. Never derived from chat text.
- `customer_display_name` — a cosmetic nickname collected conversationally. No security relevance whatsoever. These two fields must never be conflated.
- `pending_calculation` — persisted across HTTP requests via `flask.session` so multi-turn slot filling works across page-reload-safe turns.

### Why TypedDict?
LangGraph requires the state to be a typed dictionary so it can safely merge partial updates from each node. Nodes only write the fields they are responsible for — all other fields pass through unchanged.

---

## 5. Agent Graph — Full Workflow

```
                    ┌─────────────────────────────────┐
                    │         User sends message       │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │         chitchat_node            │
                    │  • first message → ask for name  │
                    │  • second message → extract name │
                    │  • "okay" / "got it" → canned   │
                    │  • "that's all" → warm close     │
                    │  • "yes" after closing → canned  │
                    │  • greeting/closing embedding    │
                    └──────────┬───────────────────────┘
                               │
               ┌───────────────┴──────────────────┐
               │ response set?                     │
              YES                                  NO
               │                                   │
              END                    ┌─────────────▼──────────────┐
          (greeting/name)            │        account_node         │
                                     │  • balance query detected?  │
                                     │  • transactions detected?   │
                                     │  • reads session_customer_id│
                                     │    from state (Flask session)│
                                     └──────────┬──────────────────┘
                                                │
                              ┌─────────────────┴──────────────────┐
                              │ response set?                       │
                             YES                                    NO
                              │                                     │
                             END                 ┌──────────────────▼──────────────┐
                       (balance/txns)            │      banking_services_node       │
                                                 │  • CALC checked before MY-records│
                                                 │  • FD / loan / pawning calc?    │
                                                 │  • transfer fee / FX rate?      │
                                                 │  • "show both" for logged-in    │
                                                 │  • my FDs / loans / cards?      │
                                                 │  • multi-turn slot filling       │
                                                 │  • dispute → escalate_node      │
                                                 └────────────┬────────────────────┘
                                                              │
                               ┌──────────────────────────────┴──────────────┐
                               │ response set?                                │
                              YES (or escalated)                              NO
                               │                                              │
                ┌──────────────┴──────────────┐              ┌───────────────▼────────────┐
               END                     escalate_node          │       intent_node          │
         (calculator /                  (dispute)             │  • DistilBERT Banking77    │
          my-records)                                         │  • 77 banking intent labels│
                                                              └───────────────┬────────────┘
                                                                              │
                                                          ┌───────────────────┴───────────────┐
                                                          │ confidence < 0.20                 │
                                                          │ AND first message?                │
                                                         YES                                  NO
                                                          │                                   │
                                                         END                  ┌───────────────▼────────────┐
                                                     (clarify)                │      sentiment_node        │
                                                                              │  • DistilBERT SST-2        │
                                                                              │  • keyword escalation rules│
                                                                              └───────────────┬────────────┘
                                                                                              │
                                                                    ┌─────────────────────────┴──────────────┐
                                                                    │ escalated?                             │
                                                                   YES                                      NO
                                                                    │                                        │
                                                                   END                      ┌───────────────▼────────────┐
                                                               (escalate)                   │      retrieve_node         │
                                                                                            │  • context-enhanced query  │
                                                                                            │  • ChromaDB category filter│
                                                                                            │  • fallback: search all    │
                                                                                            └───────────────┬────────────┘
                                                                                                            │
                                                                                            ┌───────────────▼────────────┐
                                                                                            │      generate_node         │
                                                                                            │  • PERSONA_PROMPT (tone)   │
                                                                                            │  • customer name context   │
                                                                                            │  • few-shot examples       │
                                                                                            │  • history block           │
                                                                                            │  • Llama 3.2 3B via Ollama │
                                                                                            └───────────────┬────────────┘
                                                                                                            │
                                                                                                           END
```

### Routing Summary

| From node | Condition | Next node |
|---|---|---|
| `chitchat_node` | `response` is set | `END` |
| `chitchat_node` | `response` is None | `account_node` |
| `account_node` | `response` is set | `END` |
| `account_node` | `response` is None | `banking_services_node` |
| `banking_services_node` | `response` set AND `escalated` | `escalate_node` |
| `banking_services_node` | `response` set AND not escalated | `END` |
| `banking_services_node` | `response` is None | `intent_node` |
| `intent_node` | confidence < 0.20 AND no history | `clarify_node` |
| `intent_node` | otherwise | `sentiment_node` |
| `sentiment_node` | `escalated` is True | `escalate_node` |
| `sentiment_node` | `escalated` is False | `retrieve_node` |
| `retrieve_node` | always | `generate_node` |
| `clarify_node` | always | `END` |
| `escalate_node` | always | `END` |
| `generate_node` | always | `END` |

---

## 6. Node Descriptions

### `chitchat_node`
**Purpose:** Catch short social exchanges and handle conversational UX (name collection, closings) before the ML pipeline.

The node runs **five checks in order**:

**1. Display name — ask on first message**
If `history` is empty and `customer_display_name` is None (no name collected yet), the bot asks:
> "Hi! Before we get started, what should I call you?"
This fires exactly once per session. On subsequent visits the name is already in the Flask session.

**2. Display name — extract on next reply**
If the previous bot message was the name question and no name is stored yet, the node tries to extract a name from the customer's reply using `_extract_display_name()` — a pure-regex function (no LLM). If extraction succeeds, `customer_display_name` is stored in state and persisted to Flask session. If the reply also contains a banking question, the name is stored and the message falls through to the banking pipeline.

**3. Acknowledgement check**
If the customer said "okay", "got it", "understood" etc. (≤4 words) after an informational answer, reply with "Is there anything else I can help you with?" This prevents the bot repeating the same answer on every "okay".

**4. Session-closing detection**
If the message contains a closing phrase ("that's all I need", "ok thanks", "nothing else"), reply with a warm sign-off without touching the LLM. This prevents the LLM from hallucinating a follow-up answer to a question that wasn't asked.

**5. Embedding similarity**
Encode the message with `all-MiniLM-L6-v2` and compare cosine similarity to prototype greeting/closing sentences. Score > 0.75 triggers a canned reply.

If any check fires, `state["response"]` is set and the graph ends without touching any ML model.

---

### `account_node`
**Purpose:** Answer personal banking queries directly from the database.

Scans the message for balance/transaction phrases:
- **Balance patterns:** "my balance", "check balance", "what's my balance", "how much do I have", etc.
- **Transaction patterns:** "my transactions", "recent transactions", "transaction history", "my statement", etc.

If detected, reads `session_customer_id` from `AgentState` (sourced from the Flask session — never from user input) and calls the appropriate `@tool` function. Returns a formatted string response in **LKR** without involving the LLM.

If the customer is not logged in, returns a "please verify your identity" message.

---

### `banking_services_node`
**Purpose:** Answer FD/loan/pawning/transfer/FX queries using deterministic Python math and authenticated record lookups — **without involving the LLM for any financial figure**.

**Important ordering:** Calculator checks run **before** my-records checks. This ensures messages that match both (e.g. "what is my loan EMI") go to the calculator path, which already shows personal records via the "show both" logic, rather than showing only records and letting a slot-filling reply fall through to the LLM.

This node handles three distinct modes:

**Calculator mode (no login required):**
Detects phrases matching FD, loan, EMI, pawning, transfer fee, or FX rate patterns. Extracts slots (amount, tenure, carat, loan type, currency, transfer type) via pure-regex functions in `src/slot_extraction.py`. If all slots are present, runs the deterministic calculation and formats the result. Every response unconditionally appends two disclaimers:
- *"This is an indicative estimate based on current published rates."*
- *"This is not a commitment to approve or disburse — final terms require branch verification."*

**"Show both" for logged-in users:**
When a logged-in customer asks a calculator-style question for a service that has personal records (FD, loan, pawning), the node first fetches their existing records and prepends them above the calculator result. A guest sees only the calculator result plus a note: *"Log in to also view your existing records alongside this estimate."*

**Multi-turn slot filling:**
If a required slot is missing, the node saves `{"service", "slots", "attempts", "personal_prefix", "login_note"}` to `pending_calculation` (persisted via Flask session), then asks for the missing value. On the next turn it fills the gap. After 3 failed attempts it abandons and suggests visiting a branch.

**My-records mode (login required):**
Detects phrases like "my fixed deposits", "my loans", "my cards". Reads `session_customer_id` from state (Flask session — never from user input) and calls the appropriate tool. Returns a formatted list of records with a dispute-escalation footer.

**Frustrated card detection:**
If the message contains "my card" together with frustration words like "declined", "not working", "won't work", the node skips the card-records display and falls through to the LLM, which handles it via the frustrated-customer few-shot example.

**Dispute detection:**
If the customer includes a dispute phrase ("this is wrong", "I didn't take this", "not mine", "unauthorized") in the same turn as a my-records query, the node sets `escalated=True`. The router sends the conversation to `escalate_node`.

**Information/execution boundary (permanent):**
This node ONLY informs and calculates. It never initiates any financial transaction, disbursement, record creation, or modification. All `services_db.py` functions are read-only SELECT queries.

---

### `intent_node`
**Purpose:** Classify which of 77 banking support intents the message belongs to.

Uses a fine-tuned DistilBERT model trained on the **Banking77** dataset. Returns the predicted intent label and confidence score. Examples: `card_not_working`, `balance_not_updated`, `fraud_alert`, `transfer_fee`, `pin_blocked`.

---

### `sentiment_node`
**Purpose:** Detect emotional tone and decide whether to escalate to a human.

Two-layer approach:
1. **Keyword / phrase check first** (highest priority): phrases like "I want to speak to a human", "terrible service", or crisis keywords trigger immediate escalation.
2. **DistilBERT SST-2** scores sentiment as Positive / Negative / Neutral for display in UI. The ML score alone never triggers escalation.

---

### `clarify_node`
**Purpose:** Ask the customer to rephrase when intent confidence is below 0.20 on the first message only.

---

### `escalate_node`
**Purpose:** Return a human-handoff message. The UI displays a red "ESCALATED TO HUMAN AGENT" banner.

---

### `retrieve_node`
**Purpose:** Search the ChromaDB knowledge base for relevant FAQ documents.

**Context-enhanced query:** Short follow-ups (≤8 words) with prior history → prepend last 1–2 user messages. Contact queries → override category to "account". Category filtering → fallback to all if no results.

---

### `generate_node`
**Purpose:** Build the final prompt and call Llama 3.2 3B via Ollama.

Assembles a structured prompt in this exact order:

| Layer | Content |
|---|---|
| **[1] PERSONA_PROMPT** | Tone/persona system layer (see Section 11) |
| **[2] name_context** | `"Customer's preferred name: Alice\n\n"` — or empty if no name yet |
| **[3] FEW_SHOT_EXAMPLES** | 5 hand-crafted examples covering all 4 KB categories + frustrated customer |
| **[4] history block** | Last 4 conversation turns formatted as `Customer: / Assistant:` |
| **[5] context** | Top-3 document chunks from ChromaDB |
| **[6] query** | Current customer message |
| **[7] Rules + Answer:** | Output constraints (answer only from context, numbered steps for how-to, etc.) |

The raw LLM output is cleaned (strips internal reasoning markers like "Answer:", "Step 4 -") before being stored in state.

---

## 7. Tool Definitions

All tools in `src/agent_tools.py` use the LangChain `@tool` decorator.

```
# ── Core ML pipeline tools ──────────────────────────────────────────────────
tool_classify_intent(message)          → {intent, confidence}
tool_analyse_sentiment(message)        → {label, score, escalate}
tool_search_knowledge_base(query, cat) → [doc_chunk, ...]
tool_generate_response(prompt)         → str

# ── Account tools (login required) ─────────────────────────────────────────
tool_get_account_balance(customer_id)  → str  (formatted balance in LKR)
tool_get_recent_transactions(cid)      → str  (formatted transaction list in LKR)

# ── Banking service calculators (login NOT required; return dict) ────────────
tool_calculate_fd_interest(principal, tenure_months)
    → {principal, requested_tenure, matched_tenure, annual_rate,
       interest_earned, maturity_amount, penalty_rate}
    | {error: str}

tool_calculate_loan_emi(loan_amount, tenure_months, loan_type)
    → {loan_type, annual_rate, principal, tenure_months,
       emi, total_payable, total_interest}
    | {error: str}

tool_calculate_pawning_advance(weight_grams, carat, tenure_months)
    → {carat, weight_grams, rate_per_gram, advance_amount,
       monthly_interest_rate, tenure_months, total_interest, total_payable}
    | {error: str}

tool_calculate_transfer_fee(amount, transfer_type)
    → {transfer_type, amount, fee_type, fee, total_amount}
    | {error: str}

tool_get_fx_rate(currency_code)
    → {currency_code, rate_to_lkr, updated_date}
    | {error: str}

# ── My-records tools (login required; return pre-formatted str) ─────────────
tool_get_my_fixed_deposits(customer_id)   → str  (FD list + dispute footer)
tool_get_my_loans(customer_id)            → str  (loan list + dispute footer)
tool_get_my_pawning_records(customer_id)  → str  (pawning list + dispute footer)
tool_get_my_cards(customer_id)            → str  (card list + dispute footer)
```

### Security note
`customer_id` is **never supplied by the LLM**. Nodes read it directly from `state["session_customer_id"]`, set by `run_agent()` from `flask.session`. Tools also fail-safe: empty `customer_id` → "please log in" message, no DB query.

### Why calculators return `dict` but my-records return `str`
Calculator tools return a `dict` so `banking_services_node` formats the display string in Python — guaranteeing the LLM never generates a financial figure. My-records tools return `str` because their output is pre-formatted record data (no calculation).

---

## 8. RAG Pipeline (Retrieval-Augmented Generation)

### Knowledge Base
FAQ documents stored as `.txt` files in `knowledge_base/`. Each file belongs to one of four categories: `fraud`, `billing`, `technical`, `account`.

### Embedding
`SentenceTransformer("all-MiniLM-L6-v2")` — 22M parameters, 384-dimensional embeddings. Used for ChromaDB indexing, query encoding, and chitchat prototype matching.

### Retrieval
ChromaDB cosine similarity search. Top-3 chunks ≥ 0.45 similarity returned. Category-filtered → fallback to all categories if empty.

### Generation
Chunks injected into the structured prompt (Section 6, `generate_node`). Llama 3.2 3B generates the response. Instructed to answer only from provided context.

---

## 9. Account Banking Feature

> **DEMO MODE** — All account data is completely simulated.

### Flow

```
Customer: "what's my balance?"
        │
        ▼
  account_node detects balance pattern
        │
        ▼
  reads session_customer_id from AgentState (from flask.session — trusted)
        │
        ▼
  tool_get_account_balance.invoke({"customer_id": session_customer_id})
        │
        ▼
  "Your account balance:
     Account : **** 2345 (Checking)
     Balance : LKR 5,432.50
     Last transaction : 2026-06-11"
```

### Authentication Flow

```
Browser → GET /           → not authenticated → redirect /login
Browser → GET /login      → shows Login/Signup page
User enters DEMO001 / 1234
Browser → POST /login     → Flask calls verify_pin()
                          → bcrypt.checkpw() passes
                          → session['authenticated_customer_id'] = 'DEMO001'
Browser → GET /           → authenticated → shows chat page
JS → GET /me              → returns {name, masked_account_no, balance (LKR), ...}
                          → account header bar rendered
User: "what's my balance?"
Browser → POST /chat      → session_customer_id = session.get('authenticated_customer_id')
                          → run_agent(..., session_customer_id='DEMO001')
```

### Rate Limiting / Lockout
- 5 failed PIN attempts within 15 minutes → account locked for 15 minutes
- Same generic error message for wrong customer ID or wrong PIN (prevents user enumeration)

---

## 10. Banking Services Feature

> **DEMO MODE** — All rates and customer records are simulated.

### Information / Execution Boundary

| Allowed (INFORM only) | Prohibited (never implement) |
|---|---|
| Calculate indicative FD maturity amount | Actually open or close an FD |
| Estimate loan EMI | Disburse a loan or approve an application |
| Quote pawning advance for given weight/carat | Create or redeem a pawning record |
| Show transfer fee for a given amount | Move or transfer any funds |
| Show FX rate | Execute a currency conversion |
| Display customer's existing product records | Modify any record |

### Two Modes + "Show Both"

**Calculator mode** (no login required):
```
Customer: "What's the FD interest if I deposit 500,000 for 1 year?"
→ banking_services_node detects FD pattern
→ extract_amount → 500000.0 | extract_tenure_months → 12
→ tool_calculate_fd_interest(500000.0, 12)
→ Python: interest = 500000 × 10.50% × 1 = LKR 52,500.00
→ Format in Python (LLM never sees figure):
  "Fixed Deposit Estimate
   Principal   : LKR 500,000.00
   Tenure      : 12 months @ 10.50% p.a.
   Interest    : LKR  52,500.00
   Maturity    : LKR 552,500.00
   ..."
```

**"Show both" (logged-in calculator queries):**
When a logged-in customer asks a calculator question for a service with personal records, the bot first shows their records, then the calculator estimate below a separator:
```
Your Fixed Deposits (2 records):
  1. FD-D001-01 ...  [existing records]
  2. FD-D001-02 ...

────────────────────────────────────

Fixed Deposit Estimate
  [calculator result]
```
Unauthenticated users see only the calculator result with a note to log in.

**My-records mode** (login required):
```
Customer: "Show my fixed deposits"
→ banking_services_node detects my-FD pattern
→ reads session_customer_id (Flask session)
→ tool_get_my_fixed_deposits("DEMO001") → formatted FD list
→ Response includes records + dispute escalation footer
```

### Multi-Turn Slot Filling
```
Turn 1  Customer: "I want to know my loan EMI"
        Bot: "What loan amount are you considering? (e.g. LKR 500,000)"

Turn 2  Customer: "500,000 for 3 years"
        Bot: "What type of loan? (personal / housing / vehicle / education / business)"

Turn 3  Customer: "personal"
        Bot: "Personal Loan EMI Estimate
               Monthly EMI    : LKR 18,075.44
               Total Payable  : LKR 650,715.84
               ..."
```
After 3 failed attempts the node abandons and suggests visiting a branch.

### Dispute Detection
Phrases like "this is wrong", "I didn't take this", "not mine" in the same message as a my-records query → records shown + `escalated=True` → `escalate_node`.

### Required Disclaimers
Every calculator response unconditionally appends (injected in Python, not by LLM):
> *This is an indicative estimate based on current published rates.*
> *This is not a commitment to approve or disburse — final terms require branch verification.*

---

## 11. Conversational UX Features

### 11.1 Customer Display Name

The bot collects the customer's preferred name once per session through natural conversation:

```
Turn 1  Customer: "hi"
        Bot: "Hi! Before we get started, what should I call you?"

Turn 2  Customer: "I'm Alice"
        Bot: "Nice to meet you, Alice! How can I help you today?"

Turn 3+ Customer: "what's my balance?"
        Bot: (uses "Alice" occasionally — not in every response)
```

**How it works:**
- `chitchat_node` detects first message (`history == []`, `display_name is None`) → asks for name
- `_extract_display_name()` — pure regex, no LLM. Strips name-introduction prefixes ("I'm", "my name is", "call me"), rejects banking keywords, questions, and excessively long text. Accepts 1–3 word names.
- Name stored in `state["customer_display_name"]` → persisted to `flask.session` → returned as 7th element of `run_agent()` tuple
- On page reload: name already in Flask session → passed into graph → no name-ask

**Security separation:**
`customer_display_name` and `session_customer_id` are completely independent fields. The display name is never used for database lookups or authentication. Using one in place of the other is explicitly prevented by code structure and comments.

**Name usage in `generate_node`:**
When `customer_display_name` is set, it is injected into the prompt as:
```
Customer's preferred name: Alice
```
The PERSONA_PROMPT instructs the model to use the name naturally in appropriate moments (first greeting, closing, softening bad news) — **never in every message**, **never in back-to-back responses**.

### 11.2 Tone / Persona System

`PERSONA_PROMPT` in `src/prompt_templates.py` is injected as the first layer of every `generate_node` prompt:

```
You are a warm, professional banking service assistant — the way a good,
experienced human bank teller speaks: efficient, clear, and genuinely
attentive, without being overly formal or robotic.

Guidelines:
- Use the customer's name naturally, not in every message. Never in back-to-back responses.
- Be concise. Don't pad with unnecessary apologies or filler.
- Never invent, estimate, or restate a number — those come from tool results in context.
- Accuracy comes first, warmth comes second.
- Do not make promises, guarantees, or commitments beyond what the context states.
- For frustrated customers: acknowledge briefly in one phrase, then resolve directly.
```

This layer has no effect on `banking_services_node` responses (which bypass `generate_node` entirely) — it only applies to LLM-generated answers.

### 11.3 Session Closing Detection

Messages like "ok thanks, that's all I need" are caught by `_check_acknowledgement()` via an explicit closing-phrase set (`_CLOSING_PHRASES`). This prevents the LLM from interpreting a goodbye as a question and hallucinating a follow-up financial answer.

---

## 12. Security Model

| Threat | Mitigation |
|---|---|
| PIN storage | bcrypt hash with random salt — plaintext never stored |
| Session hijacking | `SESSION_COOKIE_HTTPONLY=True` |
| Customer ID injection via chat | `session_customer_id` sourced exclusively from `flask.session`, never from chat body |
| SQL injection on `accounts.db` | All queries use parameterised `?` placeholders |
| SQL injection on `services.db` | All queries use parameterised `?` placeholders |
| User enumeration | `verify_pin()` returns identical error for wrong ID and wrong PIN |
| Brute force | Lockout after 5 failures in 15 minutes per customer ID |
| Account number exposure | `get_account_info()` always masks to last 4 digits |
| PIN/hash leakage | `get_account_info()` excludes `pin_hash` and full `account_no` |
| LLM generating financial figures | All calculator results computed in Python, formatted in `banking_services_node` — `generate_node` bypassed entirely |
| LLM hallucinating closing answers | Session-closing phrases caught by `chitchat_node` before reaching `generate_node` |
| LLM prompt injection on banking services | Calculator and my-records queries handled before `intent_node` / `generate_node` |
| Unauthenticated my-records access | Node checks `session_customer_id`; tools also fail-safe |
| Dispute escalation | Dispute phrases → `escalated=True` → `escalate_node` |
| Display name / auth identity conflation | `customer_display_name` and `session_customer_id` are separate fields; display name never used in DB queries |

---

## 13. Database Schemas

### `accounts.db`

```sql
CREATE TABLE accounts (
    customer_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    account_no    TEXT NOT NULL UNIQUE,
    balance       REAL NOT NULL DEFAULT 0.0,
    account_type  TEXT NOT NULL DEFAULT 'Checking',
    last_txn_date TEXT,
    pin_hash      TEXT NOT NULL
);

CREATE TABLE transactions (
    txn_id      TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    date        TEXT NOT NULL,
    description TEXT NOT NULL,
    amount      REAL NOT NULL,
    txn_type    TEXT NOT NULL CHECK (txn_type IN ('credit', 'debit')),
    FOREIGN KEY (customer_id) REFERENCES accounts(customer_id)
);

CREATE TABLE login_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    success     INTEGER NOT NULL DEFAULT 0
);
```

### `chat_history.db`

```sql
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    user_msg    TEXT    NOT NULL,
    bot_msg     TEXT    NOT NULL,
    intent      TEXT,
    category    TEXT,
    sentiment   TEXT,
    escalated   INTEGER DEFAULT 0,
    feedback    INTEGER DEFAULT NULL,
    timestamp   TEXT    DEFAULT (datetime('now','localtime'))
);
```

### `services.db`

```sql
CREATE TABLE fd_rates (
    tenure_months                  INTEGER PRIMARY KEY,
    annual_rate                    REAL NOT NULL,
    early_withdrawal_penalty_rate  REAL NOT NULL
);
-- 3m/9%, 6m/9.75%, 12m/10.5%, 24m/11%, 36m/11.5%, 60m/11.25%

CREATE TABLE loan_rates (
    loan_type         TEXT PRIMARY KEY,
    annual_rate       REAL NOT NULL,
    max_tenure_months INTEGER NOT NULL
);
-- personal/18%/60m, housing/12.5%/300m, vehicle/14%/84m, education/13.5%/120m, business/16%/120m

CREATE TABLE pawning_rates (
    carat                 INTEGER PRIMARY KEY,
    rate_per_gram         REAL NOT NULL,
    ltv_percent           REAL NOT NULL,
    monthly_interest_rate REAL NOT NULL
);
-- 18ct/LKR9500/2%pm, 22ct/LKR13000/2%pm, 24ct/LKR14200/2%pm

CREATE TABLE transfer_fees (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transfer_type TEXT NOT NULL,
    min_amount    REAL NOT NULL,
    max_amount    REAL,
    fee_type      TEXT NOT NULL CHECK (fee_type IN ('fixed', 'percent')),
    fee_value     REAL NOT NULL,
    min_fee       REAL,
    currency      TEXT NOT NULL DEFAULT 'LKR'
);

CREATE TABLE fx_rates (
    currency_code TEXT PRIMARY KEY,
    rate_to_lkr   REAL NOT NULL,
    updated_date  TEXT NOT NULL
);
-- USD/325, GBP/415, EUR/355, AUD/215, SGD/245, INR/3.9, SAR/86.5

CREATE TABLE fixed_deposits (
    fd_id           TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    principal       REAL NOT NULL,
    tenure_months   INTEGER NOT NULL,
    annual_rate     REAL NOT NULL,
    start_date      TEXT NOT NULL,
    maturity_date   TEXT NOT NULL,
    maturity_amount REAL NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('active','matured','withdrawn'))
);

CREATE TABLE loans (
    loan_id              TEXT PRIMARY KEY,
    customer_id          TEXT NOT NULL,
    loan_type            TEXT NOT NULL,
    principal            REAL NOT NULL,
    outstanding_balance  REAL NOT NULL,
    monthly_installment  REAL NOT NULL,
    tenure_months        INTEGER NOT NULL,
    start_date           TEXT NOT NULL
);

CREATE TABLE pawning_records (
    pawn_id          TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL,
    item_description TEXT NOT NULL,
    weight_grams     REAL NOT NULL,
    carat            INTEGER NOT NULL,
    advance_amount   REAL NOT NULL,
    interest_rate    REAL NOT NULL,
    pawn_date        TEXT NOT NULL,
    due_date         TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('active','redeemed','forfeited'))
);

CREATE TABLE cards (
    card_id         TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    card_type       TEXT NOT NULL CHECK (card_type IN ('debit','credit','prepaid')),
    masked_number   TEXT NOT NULL,
    credit_limit    REAL,
    available_limit REAL,
    status          TEXT NOT NULL CHECK (status IN ('active','blocked','expired'))
);
```

---

## 14. Flask API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/` | Yes | Chat page (redirects to `/login` if not authenticated) |
| GET | `/login` | No | Login / Signup page |
| POST | `/login` | No | Verify PIN, set session. Body: `{customer_id, pin}` |
| POST | `/logout` | No | Clears `authenticated_customer_id`, `pending_calculation`, `customer_display_name` from session |
| POST | `/signup` | No | Create demo account + auto-login. Body: `{name, customer_id, pin}` |
| GET | `/me` | Yes | Returns `{authenticated, name, masked_account_no, account_type, balance (LKR)}` |
| POST | `/chat` | Yes* | Body: `{message, history, session_id}`. Returns `{response, intent, sentiment, category, escalated, turn_id}` |
| POST | `/feedback` | No | Body: `{turn_id, rating}` |
| GET | `/stats` | No | JSON analytics summary |
| GET | `/dashboard` | No | HTML analytics dashboard |

*`/chat` works without auth but returns "please verify identity" for account queries.

### `/chat` session flow
```python
# Reads from Flask session — NEVER from the message body
session_customer_id  = session.get('authenticated_customer_id')
pending_calc         = session.get('pending_calculation')
display_name         = session.get('customer_display_name')

response, intent, sentiment, escalated, category, pending_calc, display_name = run_agent(
    message, history,
    session_id=session_id,
    session_customer_id=session_customer_id,
    pending_calculation=pending_calc,
    customer_display_name=display_name,
)

# Persist or clear state for next turn
if pending_calc:
    session['pending_calculation'] = pending_calc
else:
    session.pop('pending_calculation', None)

if display_name:
    session['customer_display_name'] = display_name
```

---

## 15. ML Models

### Intent Classifier — Fine-tuned DistilBERT
- **Base:** `distilbert-base-uncased` | **Dataset:** Banking77 (13,083 samples, 77 classes)
- **Training:** `python train_intent_classifier.py` (~10–30 min on CPU)
- **Output:** `{intent: str, confidence: float}` | **Stored:** `models/intent_classifier/`

### Sentiment Analyser — DistilBERT SST-2
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Output:** `{label, score, escalate}` — escalation driven by keyword rules, not model score

### Embedding Model — all-MiniLM-L6-v2
- 22M parameters, 384-dimensional embeddings
- Used for ChromaDB indexing, query encoding, chitchat prototype matching

### LLM — Llama 3.2 3B (upgraded from 1B)
- **Served by:** Ollama (local, no internet required at inference time)
- **Called via:** `ollama.chat(model="llama3.2:3b", ...)`
- **Role:** Final response generation from context + structured prompt
- **Why 3B:** See Section 19 for full comparison. The 1B model hallucinated financial figures, ignored context in ~6/15 test cases, and copied few-shot examples verbatim. The 3B model passed 12/15 tests correctly.

---

## 16. How to Run

### Prerequisites

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install and start Ollama (separate terminal — keep running)
# Download from https://ollama.com
ollama serve

# 4. Pull the LLM (one-time download, ~2 GB)
ollama pull llama3.2:3b

# 5. Train the intent classifier (one-time, ~10–30 minutes on CPU)
python train_intent_classifier.py

# 6. Build the ChromaDB knowledge base index (one-time)
python build_index.py

# 7. Seed the demo accounts database (one-time)
python seed_accounts.py

# 8. Seed the banking services rate cards and customer records (one-time)
python seed_services.py
```

### Start the App

```bash
# Activate venv first if not already active
venv\Scripts\activate

# Start Flask
python ui/app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Re-seed (reset all demo data)

```bash
python seed_accounts.py --force
python seed_services.py --force
```

### Verify model is running

```bash
# Check Ollama is serving and model is available
venv\Scripts\python -c "import ollama; print([m['model'] for m in ollama.list()['models']])"
# Expected output includes: ['llama3.2:3b']
```

### Useful development commands

```bash
# Syntax check all modified files
python -c "import ast; ast.parse(open('src/agent_graph.py').read()); print('OK')"
python -c "import ast; ast.parse(open('src/prompt_templates.py').read()); print('OK')"
python -c "import ast; ast.parse(open('ui/app.py').read()); print('OK')"

# Check Ollama is responding
venv\Scripts\python -c "import ollama; r=ollama.chat('llama3.2:3b',[{'role':'user','content':'hi'}]); print(r['message']['content'])"

# Inspect services database
venv\Scripts\python -c "
import sqlite3
conn = sqlite3.connect('services.db')
for row in conn.execute('SELECT * FROM fd_rates'): print(row)
conn.close()
"

# Inspect accounts database
venv\Scripts\python -c "
import sqlite3
conn = sqlite3.connect('accounts.db')
for row in conn.execute('SELECT customer_id, name, account_type FROM accounts'): print(row)
conn.close()
"
```

---

## 17. Demo Credentials

> All accounts below contain **simulated/fictional data only**.

| Customer ID | Name | Account Type | PIN |
|---|---|---|---|
| DEMO001 | Alice Johnson | Checking | 1234 |
| DEMO002 | Bob Smith | Savings | 2345 |
| DEMO003 | Carol White | Checking | 3456 |
| DEMO004 | David Brown | Premium | 4567 |
| DEMO005 | Emma Davis | Checking | 5678 |
| DEMO006 | Frank Miller | Savings | 6789 |
| DEMO007 | Grace Wilson | Checking | 7890 |
| DEMO008 | Henry Moore | Premium | 8901 |
| DEMO009 | Iris Taylor | Checking | 9012 |
| DEMO010 | Jack Anderson | Savings | 0123 |
| DEMO011 | Kate Thomas | Checking | 1111 |
| DEMO012 | Liam Jackson | Checking | 2222 |
| DEMO013 | Mia Harris | Premium | 3333 |
| DEMO014 | Noah Martin | Savings | 4444 |
| DEMO015 | Olivia Garcia | Checking | 5555 |

### Example Queries

**Account (after login)**
- "What's my balance?" / "Show my recent transactions"
- "My card was declined" / "I think someone made an unauthorised transfer"
- "How do I dispute a charge?" / "What is the support email?"

**Banking Services Calculators (no login required)**
- "What's the FD interest if I deposit LKR 500,000 for 1 year?"
- "What is the EMI for a personal loan of LKR 300,000 for 3 years?"
- "Can I get a personal loan for 5 years?" *(slot filling)*
- "How much advance will I get for 10 grams of 22ct gold for 3 months?"
- "How much does it cost to send LKR 500 overseas?"
- "What is the USD to LKR exchange rate?"

**My Records (after login)**
- "Show my fixed deposits" / "Show my loans" / "Show my cards"
- "What is my outstanding loan balance?"
- "Show my loans — I didn't take this loan" *(triggers dispute escalation)*

**Display Name Flow**
- First message: "hi" → bot asks for name
- Reply: "I'm Alice" → bot greets by name
- Subsequent: name used occasionally, not every message

---

## 18. Challenges, Debugging & Fixes

This section documents the real problems encountered during development and the exact fixes applied. It serves as a reference for understanding why the code is structured the way it is.

---

### Challenge 1 — `banking_services_node` not intercepting messages in the live app

**Symptom:** Queries like "What's the FD interest for 500,000 for 1 year?" returned LLM-generated responses with hallucinated rates (e.g., "5.99% interest") instead of the deterministic calculator output.

**Root cause:** The Flask app was not restarted after code changes. The agent graph (`agent = build_agent()`) is compiled once at module import time. Any changes to `agent_graph.py` only take effect after a full process restart — Python cannot hot-reload a compiled LangGraph.

**How to confirm:** The LLM response changes on each restart (because Ollama is non-deterministic), confirming that the process is restarting. If you see the same hallucinated rate multiple times, the process did not restart.

**Fix:**
```bash
# Stop the running process
Ctrl + C

# Restart
python ui/app.py
```

Additionally, invalidate the Flask session cookie when testing changes to session-dependent features (display name, pending_calculation) by temporarily changing the secret key:
```python
# ui/app.py
app.secret_key = "demo-secret-change-in-production-v2"  # bump suffix to invalidate sessions
```

---

### Challenge 2 — LLM hallucinating financial figures (security violation)

**Symptom:** When a user sent slot values after a my-records response (e.g., "500000, personal, 36 months"), the message fell through to `generate_node` because no `pending_calculation` was set. The LLM generated "Your monthly installment is LKR 13,956.92" — a fabricated figure.

**Root cause:** The original code checked `MY_CHECKS` before `CALC_CHECKS`. "What is my loan EMI" matched `_MY_LOAN_PATTERNS` ("my loan") → showed records → returned immediately without setting `pending_calculation`. The user's follow-up slot message had no trigger patterns and fell to the LLM.

**Why this is a security issue:** The LLM's figure (LKR 13,956.92) was wrong (correct EMI ≈ LKR 18,077 at 18% over 36 months) AND was generated without any calculation. This violates the core principle that the LLM must never generate a financial figure.

**Fix — swap CALC before MY in `banking_services_node`:**
```python
# OLD order: MY_CHECKS first, then CALC_CHECKS
# NEW order: CALC_CHECKS first, then MY_CHECKS

# CALC_CHECKS (section 2 in node)
# MY_CHECKS   (section 3 in node)
```
"What is my loan EMI" now matches `_LOAN_CALC_PATTERNS` ("loan emi") first → triggers slot filling → `pending_calculation` set → user's "500000, personal, 36 months" fills slots → Python calculator runs → correct figure returned.

The CALC path already includes the "show both" logic (shows personal records above calculator result), so no information is lost.

---

### Challenge 3 — Missing trigger patterns (fee and loan queries not intercepted)

**Symptom 1:** "How much does it cost to send LKR 500 overseas?" → LLM response "Unfortunately, we don't have information on current exchange rates."

**Root cause:** `_FEE_CALC_PATTERNS` had "how much does it cost to transfer" but not "send overseas", "send abroad", or "cost to send".

**Fix — added to `_FEE_CALC_PATTERNS`:**
```python
"send overseas", "send abroad", "overseas transfer", "international transfer",
"foreign transfer", "transfer abroad", "send money overseas",
"cost to send", "how much to send",
```

**Symptom 2:** "Can I get a personal loan for 10 years?" → LLM response "I don't have information on current loan terms."

**Root cause:** `_LOAN_CALC_PATTERNS` had "loan emi", "emi for", etc. but not natural phrasing like "can I get a loan", "personal loan for", or "apply for a loan".

**Fix — added to `_LOAN_CALC_PATTERNS`:**
```python
"can i get a loan", "can i apply for a loan", "get a personal loan",
"personal loan for", "housing loan for", "vehicle loan for",
"apply for a loan", "take a loan", "borrow money", "loan for",
```

---

### Challenge 4 — Frustrated card message showing card records instead of help

**Symptom:** "This is so frustrating, my card keeps getting declined and nobody helps" → bot showed the card records (`tool_get_my_cards`) because "my card" matched `_MY_CARD_PATTERNS`.

**Root cause:** The MY_CHECKS loop matched "my card" without checking context. A frustrated complaint about a card declining is not a request to view card details.

**Fix — frustration bypass in MY_CHECKS:**
```python
_CARD_FRUSTRATION = [
    "declined", "not working", "doesn't work", "wont work", "won't work",
    "rejected", "blocked by", "keeps declining", "keeps getting declined",
]

for patterns, tool_fn, service_key in MY_CHECKS:
    if any(p in msg for p in patterns):
        # Card + frustration phrase → fall through to LLM (frustrated-customer few-shot)
        if service_key == "cards" and any(f in msg for f in _CARD_FRUSTRATION):
            break
        # ... normal my-records handling
```
The LLM then handles the frustrated message using the frustrated-customer few-shot example in `FEW_SHOT_EXAMPLES`.

---

### Challenge 5 — Session-closing message causing LLM hallucination

**Symptom:** "Ok thanks, that's all I need" → LLM responded "LKR 500 overseas transaction would incur a fee of LKR 1,500" — completely fabricated, based on a previous question in context.

**Root cause:** `_check_acknowledgement()` had a `len(text.split()) <= 4` word limit. "ok thanks, that's all I need" has 7 words and was not caught. The message fell through to `generate_node`, which hallucinated a financial answer.

**Fix — explicit closing-phrase set with no word-count limit:**
```python
_CLOSING_PHRASES = {
    "that's all", "thats all", "that's all i need", "thats all i need",
    "that's all i needed", "ok thanks", "okay thanks", "ok thank you",
    "thanks that's all", "nothing else", "no more questions", ...
}

def _check_acknowledgement(text, history):
    ...
    # New: closing phrases — no word-count restriction
    if any(phrase in normalized for phrase in _CLOSING_PHRASES):
        return "You're welcome! Have a great day. Feel free to reach out anytime."
```

---

### Challenge 6 — LLM model too small (llama3.2:1b context failures)

**Symptom:** Multiple test cases where the 1B model ignored the provided context entirely, copied few-shot examples verbatim, or gave completely wrong answers.

**Root cause:** 1B parameter models lack capacity to simultaneously attend to a long structured prompt (persona layer + few-shot + history + context + rules) while generating a contextually correct answer.

**Specific failures:**
- T04: Answered the card-decline few-shot example instead of the loan tenure question
- T06: Gave app-reinstall steps when asked about FD maturity amount
- T08: Evaded the zero credit limit entirely, asked an unrelated question
- T10: Said "How can I assist you today?" ignoring PIN reset context
- T12: Printed the card-decline few-shot verbatim instead of showing transactions

**Fix:** Upgraded to `llama3.2:3b`. See Section 19 for the full comparison.

```python
# src/llm_generator.py
response = ollama.chat(
    model="llama3.2:3b",   # was "llama3.2:1b"
    ...
)
```

---

### Challenge 7 — Frustrated customer answered with a clarifying question

**Symptom (3B model, T02):** "This is the THIRD time calling! My card keeps getting declined!" → bot responded "What seems to be the issue?" instead of acknowledging and giving steps.

**Root cause:** No few-shot example demonstrated the acknowledge-then-resolve pattern for frustrated customers.

**Fix — added frustrated-customer example to `FEW_SHOT_EXAMPLES`:**
```
Customer: This is the third time I've called about the same problem! My card
          keeps getting declined and nobody fixes it!
Assistant: I hear you — let's get this sorted right now.
A card decline despite sufficient balance is usually one of these:
1. Daily spending limit reached — check Account > Cards > Spending Limits.
2. Merchant category blocked — some plans restrict certain merchant types.
3. Temporary security hold — unusual spending patterns trigger an auto-hold.
Call our 24/7 line at 1-800-123-4567 if you need it unblocked immediately.
```

---

### Challenge 8 — Customer name used in back-to-back responses

**Symptom (3B model, T12):** Name appeared at the end of a transaction list response even though it had already been used in the immediately preceding reply.

**Fix — tightened name-frequency rule in `PERSONA_PROMPT`:**
```
# Before
"not in every message, only when it feels natural"

# After
"not in every message, only when it feels natural. Never use it in
back-to-back responses; if you used it in the immediately preceding
reply, skip it this turn."
```

---

### Challenge 9 — UnicodeEncodeError in Windows terminal

**Symptom:** Test scripts printing Unicode characters (`→`, `─`) crashed with `UnicodeEncodeError: 'charmap' codec` on Windows (cp1252 encoding).

**Fix:** Reconfigure stdout encoding at script start:
```python
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
```
This does not affect the Flask app (which serves JSON over HTTP and is unaffected by terminal encoding).

---

### Challenge 10 — `venv/Scripts/python` segfault on import

**Symptom:** Running `venv/Scripts/python -c "from src.agent_graph import ..."` caused a segfault because PyTorch/transformers load GPU/CUDA code at import time on Windows.

**Fix:** Stub out heavy modules in test scripts that don't need them:
```python
import sys, types
for mod in ['torch', 'transformers', 'sentence_transformers', 'datasets']:
    sys.modules.setdefault(mod, types.ModuleType(mod))
```

---

## 19. LLM Model Comparison — llama3.2:1b vs llama3.2:3b

The same 15-test suite was run against both models using the identical prompt (PERSONA_PROMPT + few-shot + context + rules). Tests covered: happy customer, frustrated customer, bad news delivery, number accuracy, unwelcome fact plainness, contact details, how-to instructions, multi-turn name usage, and session closing.

### Test Results

| ID | Scenario | 1b result | 3b result |
|---|---|---|---|
| T01 | Happy – balance check | ✅ correct figure, slightly stiff phrasing | ✅ correct, clean one-liner |
| T02 | Frustrated – card declined | ❌ wrong advice (copied wrong example) | ⚠️ turned it back as question (fixed by adding few-shot) |
| T03 | Bad news – LKR 2,000 fee on LKR 500 send | ⚠️ fee correct but hallucinated template fields | ✅ clean, fee stated plainly |
| T04 | Loan tenure exceeded | ❌ answered card-decline example instead | ✅ stated 60-month limit directly |
| T05 | EMI – LKR 16,607.32 (must not round) | ❌ never stated EMI, told user to check statement | ✅ exact figure stated |
| T06 | FD maturity – LKR 1,105,000 | ❌ gave app-reinstall steps | ✅ maturity amount stated correctly |
| T07 | Card blocked – state plainly | ⚠️ mentioned blocked but added irrelevant advice | ✅ led with blocked status |
| T08 | Zero credit limit – must not soften | ❌ evaded entirely, asked unrelated question | ✅ "available limit is LKR 0.00" stated plainly |
| T09 | Contact details | ✅ correct | ✅ correct |
| T10 | How-to – PIN reset | ❌ "How can I assist you today?" | ✅ 5 correct numbered steps |
| T11 | Multi-turn turn-1: name in first reply | ✅ used name | ⚠️ skipped name (acceptable — optional) |
| T12 | Multi-turn turn-2: name NOT repeated | ❌ printed card-decline few-shot verbatim | ⚠️ used name at end (fixed by tightening rule) |
| T13 | Multi-turn closing: brief sign-off | ❌ gave unsolicited balance + PIN steps | ✅ brief warm closing |
| T14 | Confused customer – explain fee | ❌ asked to dispute a charge | ✅ explained minimum fee correctly |
| T15 | Fraud alert – phishing email | ✅ correct | ✅ correct |

### Score Summary

| Criterion | llama3.2:1b | llama3.2:3b |
|---|---|---|
| **C1** Number accuracy (4 number tests) | 2/4 | **4/4** |
| **C2** Name frequency (9 named tests) | 8/9 | 7/9 *(minor over-use, since fixed)* |
| **C3** Unwelcome facts stated plainly (4 tests) | 1/4 | **4/4** |
| **C4** Correct / natural phrasing (all 15) | 3/15 | **12/15** |

### Decision
**`llama3.2:3b` was selected.** The 1B model's failures were structural — it lacked the capacity to attend to a long structured prompt while generating contextually correct answers. The 3B model's remaining issues (T02 frustrated, T12 name repeat) were prompt engineering problems, fixed by adding a frustrated-customer few-shot example and tightening the name-frequency rule.

```bash
# One-time: pull 3b model (~2 GB)
ollama pull llama3.2:3b

# Verify
venv\Scripts\python -c "import ollama; print([m['model'] for m in ollama.list()['models']])"
```
