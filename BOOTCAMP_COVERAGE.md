# Agentic AI Bootcamp — Project Coverage Map

**Project:** Intelligent Customer Support Chatbot — AutoTrust Bank  
**Mapped against:** Agentic AI Bootcamp (Data Science Dojo × University of New Mexico)

This document maps every bootcamp module to what was concretely built in this project,
explains *why* each technology was chosen, and describes *why it matters* in production AI systems.

---

## Module 1 — Transformers & Attention Mechanism

### What we covered

| Bootcamp Topic | What we built |
|---|---|
| Embeddings & Similarity | `SentenceTransformer("all-MiniLM-L6-v2")` encodes every FAQ document and every incoming customer message into a 384-dimensional vector |
| Semantic Search | `retrieve_top_k()` in `src/retriever.py` — cosine similarity between query vector and stored document vectors |
| RAG | `rag_node` retrieves the 3 most relevant FAQ chunks and injects them into the LLM prompt before generation |

### Why we used it
Customer queries are phrased in infinite ways. A customer asking *"my salary didn't arrive"* and another asking *"my credit transfer is missing"* mean the same thing but share almost no keywords. Keyword search fails here. Sentence Transformers map both phrases to nearby points in vector space, so both retrieve the correct "credit / salary transfer delay" FAQ document.

### Why it matters
This is the foundation of every modern RAG system. Without semantic retrieval, the LLM would answer purely from training data — risking hallucinated bank policies, wrong rates, and outdated procedures. Grounding the LLM in retrieved documents is what makes the chatbot *trustworthy* for a regulated domain like banking.

---

## Module 2 — Introduction to Agentic AI

### What we covered

| Bootcamp Topic | What we built |
|---|---|
| Reasoning, Context, Autonomy | 9-node LangGraph that reasons about intent, retrieves context, and autonomously decides the response path without human intervention per message |
| Session Memory | `AgentState` carries `history` (conversation turns), `pending_calculation` (multi-turn slot state), `customer_display_name`, and `session_customer_id` across the graph |
| Long-term Memory | `chat_history.db` stores every conversation turn; `accounts.db` holds persistent customer identity; `services.db` holds rate data |
| Human-in-the-loop | `escalate_node` detects negative sentiment or explicit dispute keywords and routes to a human agent flag (`escalated=True`) |
| Safety Guardrails | `session_customer_id` is sourced exclusively from `flask.session` — the LLM can never supply or modify it. All financial figures come from deterministic DB lookups |
| Agentic Design Patterns | Sequential routing (intent → sentiment → agent_loop_node), conditional branching (route_to_node), tool use (15 `@tool` functions), ReAct loop in `agent_loop_node` (THINK → ACT → OBSERVE) |

### Why we used it
A single LLM call cannot handle the full customer support workflow. It cannot verify identity, check a real database, escalate a frustrated customer, or track what was asked two turns ago. The agentic framework separates each concern into a dedicated node, making each step independently testable and auditable.

### Why it matters
Agentic design is what separates a *demo chatbot* from a *production system*. The autonomy loop (perceive → reason → act) lets the system handle multi-step requests (e.g., "check my balance and then tell me if I qualify for a loan") without a human scripting every path.

---

## Module 3 — Mastering LangChain

### What we covered

| Bootcamp Topic | What we built |
|---|---|
| `@tool` decorator | 15 LangChain tools in `src/agent_tools.py`: intent classifier, sentiment analyser, KB search, LLM generator, account tools, and 9 banking service tools |
| Prompt Templates | `CHAIN_OF_THOUGHT_TEMPLATE`, `FEW_SHOT_EXAMPLES`, `PERSONA_PROMPT` in `src/prompt_templates.py` — structured prompts that enforce tone, reasoning style, and safety rules |
| Output Parsers | `classify_intent()` returns a structured `{"intent": str, "confidence": float}` dict; `analyse_sentiment()` returns `{"label": str, "escalate": bool}` |
| Document Loaders | `populate_from_txt_files()` reads `.txt` FAQ files from `data/`, chunks them, embeds them, and stores them in `knowledge.db` |
| Chains | The LangGraph graph *is* a chain: each node calls one or more tools, passes structured output to the next node via `AgentState`. `agent_loop_node` uses `ChatOllama.bind_tools()` to give the LLM callable tool schemas — a LangChain Core feature for structured tool-calling |

### Why we used it
LangChain's `@tool` decorator gives every function a schema that LangGraph can inspect and route. Prompt templates enforce consistent behaviour across all LLM calls — without them, the LLM's tone and safety rules would drift unpredictably. Output parsers make LLM responses machine-readable so the next node can act on them programmatically rather than parsing free text.

### Why it matters
LangChain is the connective tissue of the system. It standardises how tools are defined, how prompts are structured, and how outputs are typed — all of which are non-negotiable requirements in production where reliability and predictability matter more than raw capability.

---

## Module 4 — Vector Databases and Agentic RAG

### What we covered

| Bootcamp Topic | What we built |
|---|---|
| Embedding storage | `knowledge.db` (SQLite) stores the `embedding` blob for every FAQ chunk as a serialised numpy array |
| Similarity search | `retrieve_similar()` in `src/knowledge_db.py` loads all embeddings, computes cosine similarity, and returns chunks above a 0.45 threshold |
| Agentic RAG | `agent_loop_node` replaces the fixed retrieve→generate pair with a **ReAct loop**: the LLM is given `tool_search_knowledge_base` as a callable tool and iteratively issues search queries (up to 3 iterations) until it decides it has enough context to respond. Retrieved docs accumulate across iterations in `AgentState["retrieved_docs"]` |
| Metadata filtering | `retrieve_top_k(query, category="fraud")` filters by category inside `tool_search_knowledge_base` — the LLM can pass a `category` argument in its tool call, narrowing retrieval to relevant docs |
| Hallucination mitigation | If all searches return nothing, the loop appends "No relevant documents found." as the ToolMessage, and the LLM is forced to acknowledge the gap rather than hallucinate. A single-pass RAG fallback fires if tool-calling itself fails. |

### Why we used it
The knowledge base has 4 categories (fraud, billing, technical, account). Without metadata filtering, a question about a fraudulent transaction might retrieve a billing FAQ, confusing the LLM. Category-aware retrieval improves precision. The 0.45 cosine threshold is a hard gate — below it, no document is injected, preventing low-quality context from misleading the LLM.

### Why it matters
RAG is the industry-standard solution to LLM hallucination in closed-domain applications. A bank cannot afford responses based on the model's training data — rates change, policies change, and model knowledge has a cutoff date. RAG ensures every factual claim in the response is backed by a retrieved, auditable source document.

---

## Module 5 — Context Engineering

### What we covered

| Bootcamp Topic | What we built |
|---|---|
| LangGraph fundamentals (nodes, edges, state) | `StateGraph` with 9 nodes: `chitchat_node`, `account_node`, `banking_services_node`, `intent_node`, `sentiment_node`, `escalate_node`, `clarify_node`, and `agent_loop_node` (replaced the old `rag_node` + `generate_node` pair with a single ReAct loop node) |
| Conditional edges | `route_to_node` and `route_banking_services` — conditional routing functions that read `AgentState` fields and select the next node |
| Deterministic control flows | `banking_services_node` computes financial figures directly from the database and sets `state["response"]` — bypassing `agent_loop_node` entirely so the LLM never touches a number |
| Memory layers | `pending_calculation` persists multi-turn slot state via `flask.session`; `history` carries the full conversation; `customer_display_name` carries cosmetic identity |
| Tool integration | Each node calls exactly the tools it needs. `account_node` calls only `tool_get_account_balance` and `tool_get_recent_transactions` |
| Router agents | `chitchat_node` uses sentence similarity to short-circuit greetings/closings before any downstream node runs; `banking_services_node` intercepts service queries before the intent classifier |
| Reflection / self-critique | Prompt rules added to `CHAIN_OF_THOUGHT_TEMPLATE`: "Do not ask a question you have already asked in the conversation history" — enforces awareness of prior turns |

### Why we used it
Context engineering is what keeps the LLM *within bounds* in a high-stakes domain. The deterministic bypass for financial figures (EMI, FD maturity, FX rates) is a deliberate architectural decision: the LLM is good at natural language, not arithmetic. Letting it compute a loan EMI would risk a wrong answer delivered with complete confidence. Doing the maths in Python and handing the LLM only the formatted result is the correct engineering choice.

### Why it matters
In banking, a wrong number is not a UX problem — it is a liability. Context engineering is the discipline that draws the boundary between what the LLM is allowed to do (phrase a response naturally) and what it must never do (produce a financial figure). This boundary is the most important design decision in the entire system.

---

## Module 6 — Agentic Design Patterns

### What we covered

| Pattern | What we built |
|---|---|
| **Tool Use** | 15 `@tool` functions — intent classification, sentiment analysis, KB retrieval, LLM generation, 4 account tools, 5 calculator tools, 4 "my records" tools. Each node selects and calls the right tool for its step |
| **Planning** | Multi-turn slot filling in `banking_services_node` — when slots are missing (e.g., loan amount for an EMI calculation), the node stores `pending_calculation = {"service": "loan", "slots": {...}, "attempts": 1}` and asks the customer for the missing value. On the next turn, it resumes from where it left off |
| **Reflection** | Two prompt rules that enforce history-awareness: never repeat a question already asked; treat information provided in earlier turns as known. This is a structured reflection loop operating at the prompt level |
| **Multi-agent Collaboration** | 9 specialised nodes, each with a single responsibility. `chitchat_node` handles greetings. `account_node` handles authenticated account queries. `banking_services_node` handles service calculators and records. `agent_loop_node` handles open-ended FAQ queries via a ReAct loop (THINK → ACT → OBSERVE). `escalate_node` handles human handoff |
| **Opt-in UX Monitoring** | A second independent system (`tracker.js` + `struggle_detector.py`) runs a rule-based struggle detection loop on product pages — detect → tip → dismiss — with no ML involvement |

### Why we used it
Each pattern solves a different class of problem. Tool use prevents the LLM from hallucinating data. Planning enables multi-step workflows that span multiple HTTP requests. Reflection prevents the model from being annoyingly repetitive with frustrated customers. Multi-agent design ensures each node can be tested, replaced, or extended without touching the rest of the graph.

### Why it matters
These patterns are what distinguish a *toy demo* from a *reliable system*. A single LLM call with a long prompt cannot plan across turns, cannot use external tools safely, and cannot self-correct when it repeats itself. The patterns impose structure that makes the system predictable and auditable — both essential properties for a regulated industry.

---

## Module 7 — Agentic AI Protocols (Partial)

### What we covered

| Bootcamp Topic | What we built |
|---|---|
| Internal tool calling | Tools are registered in a `TOOL_MAP` dict and called by name from graph nodes — a lightweight internal protocol for agent-to-tool communication |
| Structured message format | Every `/chat` request carries `{message, history, session_id}`; every response carries `{response, intent, sentiment, category, escalated, turn_id}` — a fixed, typed contract between the browser and the agent |

### What was NOT covered
MCP (Model Context Protocol), A2A, and ACP were not implemented. The system uses a single-agent architecture with internal tool routing rather than a multi-agent protocol.

### Why it matters (for future work)
MCP would allow external systems (a CRM, a core banking API, a compliance tool) to expose their capabilities as tools the agent can discover and call dynamically. A2A would enable a "loan specialist agent" to be called by the main support agent without being hard-coded into the graph. These are the natural next steps when the system grows beyond a single domain.

---

## Module 8 — Model Context Protocol

### What we covered
This module was not directly implemented. The project uses a fixed tool registry rather than a dynamic MCP server. However, the architecture is MCP-*ready*: all tools are already defined with schema (via `@tool`) and can be exposed as MCP tools with minimal changes.

---

## Module 9 — Evaluation of Agents

### What we covered

| Bootcamp Topic | What we built |
|---|---|
| Human feedback collection | Thumbs up / thumbs down buttons on every bot response — stored in `chat_history.db` via `save_feedback()` |
| Metrics tracking | Every turn stores `intent`, `sentiment`, `category`, `escalated`, `rating` — the raw data for all evaluation metrics |
| Analytics dashboard | `/dashboard` renders intent distribution (bar chart), escalation rate, thumbs up/down counts, and the 10 most negatively rated responses |
| Model comparison | Tested `llama3.2:1b` vs `llama3.2:3b` across 15 query types — documented in `AGENTIC_WORKFLOW.md` Section 19 |
| Observability | Intent and sentiment are displayed live on the chat UI (meta bar); escalation is visually flagged with a red banner |
| UX struggle detection | `struggle_detector.py` evaluates interaction events from `ui_events.db` using pure rule-based logic (2+ empty-blur or submit-fail events on the same field → tip) |

### Why we used it
Evaluation closes the loop. Without it, there is no way to know if the model is improving or regressing, which intents are failing, or which customers are being escalated unnecessarily. The feedback system surfaces this data without requiring offline batch evaluation.

### Why it matters
In production, a model that performs well on day 1 may degrade over time as customer language evolves, new products are added, or the knowledge base becomes stale. Continuous evaluation via the dashboard and feedback data makes degradation visible before it affects customers at scale.

---

## Module 10 — Final Project: Multi-Agent LLM Application

### What we built — full system summary

This project implements the **"Conversational Workflow Orchestration"** track from the bootcamp, extended with **"Knowledge-Enhanced Agent"** capabilities.

| Final Project Requirement | Implementation |
|---|---|
| Multi-turn assistant | LangGraph graph processes every message through a 9-node pipeline; conversation `history` is passed on every turn |
| Specialised agent coordination | 9 nodes, each with a single responsibility, connected by conditional routing edges |
| Search and APIs for grounding | `agent_loop_node` autonomously retrieves from a 4-category FAQ knowledge base via a THINK→ACT→OBSERVE loop (up to 3 search iterations); `account_node` reads live account data from `accounts.db`; `banking_services_node` reads from `services.db` |
| Fact-checking / no hallucination | Financial figures (EMI, FD maturity, FX rates, transfer fees) are computed in Python and injected into the state — LLM never produces a number |
| External tool use | 15 LangChain `@tool` functions covering classification, retrieval, generation, account queries, and 9 banking service operations |
| Human-in-the-loop | `escalate_node` flags the conversation for human takeover on negative sentiment or dispute keywords |
| Production-ready deployment | Flask server, SQLite persistence, session management, login/logout, analytics dashboard, opt-in UX monitoring |

### Technology stack summary

| Technology | Role | Why chosen |
|---|---|---|
| **LangGraph** | Orchestration framework | Stateful, conditional graph execution — nodes can set state that changes downstream routing |
| **LangChain `@tool`** | Tool definition | Provides schema, type safety, and a standard interface for all agent tools |
| **Sentence Transformers (`all-MiniLM-L6-v2`)** | Embedding model | Lightweight (80 MB), runs fully offline, strong semantic similarity performance for short texts |
| **Ollama (`llama3.2:3b`)** | LLM generation + tool calling | Runs locally — no API cost, no data leaving the machine, suitable for a banking demo |
| **`langchain_ollama.ChatOllama`** | ReAct agent loop | `ChatOllama.bind_tools([tool_search_knowledge_base])` enables the LLM to emit structured tool calls that the `agent_loop_node` executes iteratively |
| **SQLite** | Persistence | Zero-config, file-based, sufficient for a demo with 15 customers; `accounts.db`, `services.db`, `chat_history.db`, `knowledge.db`, `ui_events.db` |
| **Flask** | Web server | Lightweight Python server with session management; integrates cleanly with the Python-native AI stack |
| **Jinja2** | Template engine | Comes with Flask; enables a clean separation between Python logic and HTML |
| **Python `re` (regex)** | Slot extraction | Deterministic, fast, no model required — extracts amounts, tenures, loan types from freeform text |

---

## Coverage Summary

| Bootcamp Module | Coverage |
|---|---|
| Transformers & Attention | ✅ Full — embeddings, semantic search, RAG |
| Introduction to Agentic AI | ✅ Full — reasoning, memory, autonomy, safety, escalation |
| Mastering LangChain | ✅ Full — tools, prompt templates, output parsers, document loaders, chains |
| Vector Databases & Agentic RAG | ✅ Full — embedding storage, similarity search, metadata filtering, hallucination mitigation, ReAct agentic loop with iterative retrieval |
| Context Engineering | ✅ Full — LangGraph nodes/edges/state, deterministic control flow, memory layers, router agents, reflection |
| Agentic Design Patterns | ✅ Full — tool use, planning (slot filling), reflection (prompt rules), multi-agent specialisation, ReAct loop (`agent_loop_node`) |
| Agentic AI Protocols | ⚠️ Partial — internal tool registry only; MCP/A2A/ACP not implemented |
| Model Context Protocol | ❌ Not implemented — architecture is MCP-ready for future extension |
| Evaluation of Agents | ✅ Full — human feedback, metrics dashboard, model comparison, UX struggle detection |
| Final Project | ✅ Complete — production-ready multi-agent banking support system |
