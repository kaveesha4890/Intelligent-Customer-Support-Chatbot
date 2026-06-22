import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langgraph.graph import StateGraph, END
from sentence_transformers import util

from src.agent_state import AgentState
from src.agent_tools import (
    tool_classify_intent,
    tool_analyse_sentiment,
    tool_search_knowledge_base,
    tool_generate_response,
    tool_get_account_balance,
    tool_get_recent_transactions,
)
from src.prompt_templates import CHAIN_OF_THOUGHT_TEMPLATE, FEW_SHOT_EXAMPLES
import src.retriever as _retriever
from src.retriever import init_retriever

# ── constants ────────────────────────────────────────────────────────────────
CHITCHAT_THRESHOLD   = 0.75
CONFIDENCE_THRESHOLD = 0.20
CATEGORY_THRESHOLD   = 0.50

_GREETING_PROTOTYPES = ["Hello", "Hi there", "Good morning", "Hey", "Greetings", "Hi", "Good evening"]
_CLOSING_PROTOTYPES  = ["Thank you", "Thanks", "Goodbye", "Bye", "That's all", "Cheers", "Thanks a lot"]

_greeting_vecs = None
_closing_vecs  = None

# Short affirmatives after a bot closing question ("yes I have another question")
_CONTINUATION_WORDS = {"yes", "yeah", "yep", "sure", "please", "yup", "of course"}
# Phrases in the bot's last message that signal it was a closing/question
_BOT_CLOSING_PHRASES = [
    "is there anything else", "anything else i can", "you're welcome",
    "have a great day", "glad i could help", "can i help you with",
]

# Short acknowledgements after an informational answer ("okay", "got it", "I see")
_ACKNOWLEDGEMENT_WORDS = {
    "okay", "ok", "got it", "alright", "i see", "understood",
    "noted", "fine", "oh okay", "oh ok", "i got it", "makes sense",
}

# Short contact-related queries that should always search the account/contact doc
_CONTACT_KEYWORDS = {"email", "contact", "reach", "phone", "number", "helpdesk", "support line", "message"}

# Personal account query patterns — detected before intent classification
_BALANCE_PATTERNS = [
    "my balance", "check balance", "account balance", "how much do i have",
    "what's my balance", "what is my balance", "my funds", "available balance",
    "how much money",
]
_TXNS_PATTERNS = [
    "my transactions", "recent transactions", "transaction history",
    "my statement", "show my transactions", "recent activity",
    "last transactions", "my spending", "show transactions",
]


def _detect_account_query(text: str):
    """Return 'balance' or 'transactions' if the message is a personal account query."""
    t = text.lower()
    if any(p in t for p in _BALANCE_PATTERNS):
        return "balance"
    if any(p in t for p in _TXNS_PATTERNS):
        return "transactions"
    return None

_CATEGORY_RULES = [
    (["fraud", "scam", "phishing", "compromised", "unauthori", "stolen",
      "takeover", "skimm", "hack"], "fraud"),
    (["charge", "fee", "bill", "payment", "transfer", "refund", "top_up", "balance",
      "statement", "exchange", "transaction", "overdraft", "proration", "autopay",
      "subscription", "cancel", "dispute"], "billing"),
    (["card", "pin", "app", "otp", "login", "browser", "biometric", "notification",
      "session", "update", "technical", "declined", "contactless", "virtual", "atm"], "technical"),
    (["account", "password", "personal", "email", "verify", "identity",
      "limit", "close", "open", "joint", "inactive", "statement", "two_factor",
      "minimum", "lost", "passcode"], "account"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_chitchat_vecs():
    global _greeting_vecs, _closing_vecs
    if _greeting_vecs is None:
        init_retriever()
        _greeting_vecs = _retriever.embed_model.encode(_GREETING_PROTOTYPES, convert_to_tensor=True)
        _closing_vecs  = _retriever.embed_model.encode(_CLOSING_PROTOTYPES,  convert_to_tensor=True)


def _check_chitchat(text: str):
    _load_chitchat_vecs()
    vec = _retriever.embed_model.encode(text, convert_to_tensor=True)
    if util.cos_sim(vec, _greeting_vecs).max().item() > CHITCHAT_THRESHOLD:
        return "Hello! I'm your AI customer support assistant. How can I help you today?"
    if util.cos_sim(vec, _closing_vecs).max().item() > CHITCHAT_THRESHOLD:
        return "You're welcome! Is there anything else I can help you with?"
    return None


def _check_continuation(text: str, history: list):
    """Detect short affirmatives ('yes', 'sure') after a bot closing question."""
    if not history:
        return None
    last_bot = history[-1][1].lower()
    bot_was_closing = any(phrase in last_bot for phrase in _BOT_CLOSING_PHRASES)
    user_is_affirming = text.strip().lower().rstrip("?!. ") in _CONTINUATION_WORDS
    if bot_was_closing and user_is_affirming:
        return "Of course! What would you like help with?"
    return None


def _check_acknowledgement(text: str, history: list):
    """Detect short acknowledgements ('okay', 'got it') after an informational answer.
    Without this the bot repeats the same answer on every 'okay'."""
    if not history:
        return None
    normalized = text.strip().lower().rstrip("?!. ")
    if normalized in _ACKNOWLEDGEMENT_WORDS and len(text.split()) <= 4:
        return "Is there anything else I can help you with?"
    return None


def _intent_to_category(intent: str, confidence: float):
    if confidence < CATEGORY_THRESHOLD:
        return None
    for keywords, category in _CATEGORY_RULES:
        if any(k in intent.lower() for k in keywords):
            return category
    return None


def _clean_response(text: str) -> str:
    for marker in ["Answer:", "Response:", "Step 4", "Step 4 -"]:
        if marker in text:
            text = text.split(marker, 1)[1]
    lines = text.strip().split("\n")
    if lines and lines[0].rstrip().endswith(":"):
        lines = lines[1:]
    return "\n".join(lines).strip()

# ── nodes ────────────────────────────────────────────────────────────────────

def chitchat_node(state: AgentState) -> AgentState:
    """Detect greetings/closings via embedding similarity, and continuation affirmatives
    after a closing bot message — both skip the full ML pipeline."""
    history = state.get("history") or []

    # 1. "okay" / "got it" after any informational answer → ask what else they need
    reply = _check_acknowledgement(state["user_message"], history)
    # 2. "yes" / "sure" after a bot closing question → ask what they'd like help with
    if not reply:
        reply = _check_continuation(state["user_message"], history)
    # 3. Greeting / closing embedding check
    if not reply:
        reply = _check_chitchat(state["user_message"])

    if reply:
        state["response"]  = reply
        state["intent"]    = "greeting"
        state["sentiment"] = "Neutral"
        state["escalated"] = False
        state["category"]  = "all"
    return state


def account_node(state: AgentState) -> AgentState:
    """Handle personal account queries (balance / transactions).

    session_customer_id comes from AgentState, set by run_agent() from flask.session.
    The LLM never controls this value — it flows from the trusted server-side session.
    """
    qtype = _detect_account_query(state["user_message"])
    if not qtype:
        return state   # nothing to do — routing will send to intent_node

    # Read customer_id from state (sourced from Flask session, never user input)
    customer_id = state.get("session_customer_id") or ""

    if qtype == "balance":
        response = tool_get_account_balance.invoke({"customer_id": customer_id})
        intent   = "account_balance"
    else:
        response = tool_get_recent_transactions.invoke({"customer_id": customer_id})
        intent   = "account_transactions"

    state["response"]  = response
    state["intent"]    = intent
    state["sentiment"] = "Neutral"
    state["escalated"] = False
    state["category"]  = "personal_banking"
    return state


def intent_node(state: AgentState) -> AgentState:
    """Classify the banking intent using fine-tuned DistilBERT (Banking77)."""
    result = tool_classify_intent.invoke({"message": state["user_message"]})
    state["intent"]     = result["intent"]
    state["confidence"] = result["confidence"]
    return state


def sentiment_node(state: AgentState) -> AgentState:
    """Detect emotional tone and set escalation flag."""
    result = tool_analyse_sentiment.invoke({"message": state["user_message"]})
    state["sentiment"] = result["label"]
    state["escalated"] = result["escalate"]
    return state


def clarify_node(state: AgentState) -> AgentState:
    """Ask the customer to rephrase when intent confidence is too low on the first message."""
    state["response"]  = "I'm not sure I understood that. Could you rephrase or give me more details?"
    state["intent"]    = "unclear"
    state["escalated"] = False
    state["category"]  = "all"
    return state


def escalate_node(state: AgentState) -> AgentState:
    """Return a human-handoff message for distressed customers."""
    state["response"] = (
        "I understand this is frustrating, and I'm escalating your request "
        "to a human support agent right now. Someone will be with you shortly."
    )
    state["category"] = "all"
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Search ChromaDB for relevant FAQ documents using a context-enhanced query."""
    history = state.get("history") or []
    msg = state["user_message"]

    msg_words = set(msg.lower().split())
    is_contact_query = bool(msg_words & _CONTACT_KEYWORDS) and len(msg.split()) <= 8

    if is_contact_query:
        # Use only the current message so "email" / "phone" cleanly matches
        # the contact_support doc without card/billing context diluting it.
        retrieval_query = msg
        category = "account"
    else:
        # Context-enhanced query: for short follow-ups include the last TWO user turns
        if len(history) >= 2 and len(msg.split()) <= 8:
            retrieval_query = f"{history[-2][0]} {history[-1][0]} {msg}"
        elif history:
            retrieval_query = f"{history[-1][0]} {msg}"
        else:
            retrieval_query = msg
        category = _intent_to_category(state["intent"], state["confidence"])

    state["category"] = category or "all"

    docs = tool_search_knowledge_base.invoke({
        "query": retrieval_query,
        "category": category or "",
    })

    # Fallback: if category filter returned nothing, search all categories
    if not docs and category:
        docs = tool_search_knowledge_base.invoke({
            "query": retrieval_query,
            "category": "",
        })

    state["retrieved_docs"] = docs
    return state


def generate_node(state: AgentState) -> AgentState:
    """Build the few-shot + history + context prompt and call the local LLM."""
    history = state.get("history") or []
    history_block = ""
    if history:
        lines = [f"Customer: {u}\nAssistant: {b}" for u, b in history[-4:]]
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    docs    = state.get("retrieved_docs") or []
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(
        few_shot=FEW_SHOT_EXAMPLES,
        history=history_block,
        context=context,
        query=state["user_message"],
    )

    raw      = tool_generate_response.invoke({"prompt": prompt})
    state["response"] = _clean_response(raw)
    return state

# ── conditional edges ─────────────────────────────────────────────────────────

def route_chitchat(state: AgentState) -> str:
    """After chitchat check: if a canned reply was set, go to END; otherwise check account query."""
    return END if state.get("response") else "account_node"


def route_account(state: AgentState) -> str:
    """After account check: if handled, go to END; otherwise classify intent."""
    return END if state.get("response") else "intent_node"


def route_intent(state: AgentState) -> str:
    """After intent classification: clarify on low confidence (first message only)."""
    history = state.get("history") or []
    if state["confidence"] < CONFIDENCE_THRESHOLD and not history:
        return "clarify_node"
    return "sentiment_node"


def route_sentiment(state: AgentState) -> str:
    """After sentiment analysis: escalate or continue to retrieval."""
    return "escalate_node" if state["escalated"] else "retrieve_node"

# ── graph assembly ────────────────────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("chitchat_node",  chitchat_node)
    graph.add_node("account_node",   account_node)
    graph.add_node("intent_node",    intent_node)
    graph.add_node("sentiment_node", sentiment_node)
    graph.add_node("clarify_node",   clarify_node)
    graph.add_node("escalate_node",  escalate_node)
    graph.add_node("retrieve_node",  retrieve_node)
    graph.add_node("generate_node",  generate_node)

    graph.set_entry_point("chitchat_node")

    graph.add_conditional_edges("chitchat_node", route_chitchat,
                                {"__end__": END, "account_node": "account_node"})
    graph.add_conditional_edges("account_node", route_account,
                                {"__end__": END, "intent_node": "intent_node"})
    graph.add_conditional_edges("intent_node", route_intent,
                                {"clarify_node": "clarify_node", "sentiment_node": "sentiment_node"})
    graph.add_conditional_edges("sentiment_node", route_sentiment,
                                {"escalate_node": "escalate_node", "retrieve_node": "retrieve_node"})

    graph.add_edge("clarify_node",  END)
    graph.add_edge("escalate_node", END)
    graph.add_edge("retrieve_node", "generate_node")
    graph.add_edge("generate_node", END)

    return graph.compile()


# Compiled once at import time — restart app to pick up any graph changes
agent = build_agent()


def run_agent(
    user_message: str,
    history: list,
    session_id: str = "default",
    session_customer_id: str | None = None,
) -> tuple:
    """Run the agent and return (response, intent, sentiment, escalated, category).

    session_customer_id: authenticated customer from flask.session — never from user input.
    """
    result = agent.invoke({
        "user_message":       user_message,
        "history":            history,
        "session_id":         session_id,
        "session_customer_id": session_customer_id,   # trusted — from Flask session
        "intent":             None,
        "confidence":         None,
        "sentiment":          None,
        "escalated":          None,
        "category":           None,
        "retrieved_docs":     None,
        "response":           None,
    })
    return (
        result["response"],
        result["intent"]    or "unclear",
        result["sentiment"] or "Neutral",
        result["escalated"] or False,
        result["category"]  or "all",
    )
