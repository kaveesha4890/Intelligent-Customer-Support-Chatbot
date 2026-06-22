from sentence_transformers import util
from src.intent_classifier import classify_intent
from src.sentiment import analyse_sentiment
from src.retriever import retrieve_top_k, init_retriever
import src.retriever as _retriever
from src.prompt_templates import CHAIN_OF_THOUGHT_TEMPLATE, FEW_SHOT_EXAMPLES
from src.llm_generator import generate_response
from src.account_tool import get_account_balance, get_recent_transactions

INTENT_CONFIDENCE_THRESHOLD = 0.20
CATEGORY_ROUTE_THRESHOLD    = 0.50

# ── category routing rules ────────────────────────────────────────────────────

_CATEGORY_RULES = [
    (["fraud", "scam", "phishing", "compromised", "unauthori", "stolen", "takeover", "skimm", "hack"], "fraud"),
    (["charge", "fee", "bill", "payment", "transfer", "refund", "top_up", "balance",
      "statement", "exchange", "transaction", "overdraft", "proration", "autopay",
      "subscription", "cancel", "dispute"], "billing"),
    (["card", "pin", "app", "otp", "login", "browser", "biometric", "notification",
      "session", "update", "technical", "declined", "contactless", "virtual", "atm"], "technical"),
    (["account", "password", "personal", "email", "verify", "identity",
      "limit", "close", "open", "joint", "inactive", "statement", "two_factor",
      "minimum", "lost", "passcode"], "account"),
]


def _intent_to_category(intent: str, confidence: float) -> str | None:
    if confidence < CATEGORY_ROUTE_THRESHOLD:
        return None
    for keywords, category in _CATEGORY_RULES:
        if any(k in intent.lower() for k in keywords):
            return category
    return None


# ── chitchat / continuation detection ────────────────────────────────────────

CHITCHAT_SIMILARITY_THRESHOLD = 0.75

_GREETING_PROTOTYPES = ["Hello", "Hi there", "Good morning", "Hey", "Greetings", "Hi", "Good evening"]
_CLOSING_PROTOTYPES  = ["Thank you", "Thanks", "Goodbye", "Bye", "That's all", "Cheers", "Thanks a lot"]

_greeting_vecs = None
_closing_vecs  = None

# "okay" / "got it" after an informational answer → ask what else they need
_ACKNOWLEDGEMENT_WORDS = {
    "okay", "ok", "got it", "alright", "i see", "understood",
    "noted", "fine", "oh okay", "oh ok", "i got it", "makes sense",
}
# "yes" / "sure" after a bot closing question → ask what they'd like help with
_CONTINUATION_WORDS = {"yes", "yeah", "yep", "sure", "please", "yup", "of course"}
_BOT_CLOSING_PHRASES = [
    "is there anything else", "anything else i can", "you're welcome",
    "have a great day", "glad i could help", "can i help you with",
]

# Short contact queries → force account category so support doc is found
_CONTACT_KEYWORDS = {"email", "contact", "reach", "phone", "number", "helpdesk", "message"}


def _get_chitchat_vecs():
    global _greeting_vecs, _closing_vecs
    if _greeting_vecs is None:
        init_retriever()
        _greeting_vecs = _retriever.embed_model.encode(_GREETING_PROTOTYPES, convert_to_tensor=True)
        _closing_vecs  = _retriever.embed_model.encode(_CLOSING_PROTOTYPES,  convert_to_tensor=True)


def _check_chitchat(text: str) -> str | None:
    _get_chitchat_vecs()
    vec = _retriever.embed_model.encode(text, convert_to_tensor=True)
    if util.cos_sim(vec, _greeting_vecs).max().item() > CHITCHAT_SIMILARITY_THRESHOLD:
        return "Hello! I'm your AI customer support assistant. How can I help you today?"
    if util.cos_sim(vec, _closing_vecs).max().item() > CHITCHAT_SIMILARITY_THRESHOLD:
        return "You're welcome! Is there anything else I can help you with?"
    return None


def _check_acknowledgement(text: str, history: list) -> str | None:
    """'okay' / 'got it' after an answer → ask what else they need."""
    if not history:
        return None
    if text.strip().lower().rstrip("?!. ") in _ACKNOWLEDGEMENT_WORDS and len(text.split()) <= 4:
        return "Is there anything else I can help you with?"
    return None


def _check_continuation(text: str, history: list) -> str | None:
    """'yes' / 'sure' after a bot closing question → ask what they'd like help with."""
    if not history:
        return None
    last_bot = history[-1][1].lower()
    if any(p in last_bot for p in _BOT_CLOSING_PHRASES):
        if text.strip().lower().rstrip("?!. ") in _CONTINUATION_WORDS:
            return "Of course! What would you like help with?"
    return None


# ── account query detection ───────────────────────────────────────────────────

# Phrases that unambiguously mean "show me my balance / transactions"
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


def _detect_account_query(text: str) -> str | None:
    """Return 'balance' or 'transactions' if the message is clearly a personal
    account query, otherwise None."""
    t = text.lower()
    if any(p in t for p in _BALANCE_PATTERNS):
        return "balance"
    if any(p in t for p in _TXNS_PATTERNS):
        return "transactions"
    return None


def _handle_account_query(query_type: str, session_customer_id: str | None) -> str:
    """Route to the appropriate account function.
    session_customer_id comes from Flask session — never from user input."""
    if query_type == "balance":
        return get_account_balance(session_customer_id or "")
    if query_type == "transactions":
        return get_recent_transactions(session_customer_id or "")
    return "I can check your balance or recent transactions. Which would you like?"


# ── response cleaner ──────────────────────────────────────────────────────────

def _clean_response(response: str) -> str:
    for marker in ["Answer:", "Response:", "Step 4", "Step 4 -"]:
        if marker in response:
            response = response.split(marker, 1)[1]
    lines = response.strip().split("\n")
    if lines and lines[0].rstrip().endswith(":"):
        lines = lines[1:]
    return "\n".join(lines).strip()


# ── main pipeline ─────────────────────────────────────────────────────────────

def chat(user_message: str, history: list, session_customer_id: str | None = None):
    """Fixed RAG pipeline.

    session_customer_id: authenticated customer ID from Flask session.
    MUST be sourced from flask.session by the caller — never from user input.
    """

    # 1. Short acknowledgements ("okay", "got it") → "Is there anything else?"
    ack = _check_acknowledgement(user_message, history)
    if ack:
        return ack, "greeting", "Neutral", False, "all"

    # 2. Affirmative after bot closing question ("yes") → "What would you like help with?"
    cont = _check_continuation(user_message, history)
    if cont:
        return cont, "greeting", "Neutral", False, "all"

    # 3. Greeting / closing embedding check
    quick_reply = _check_chitchat(user_message)
    if quick_reply:
        return quick_reply, "greeting", "Neutral", False, "all"

    # 4. Personal account queries (balance / transactions) — no LLM needed
    account_qtype = _detect_account_query(user_message)
    if account_qtype:
        response = _handle_account_query(account_qtype, session_customer_id)
        intent   = "account_balance" if account_qtype == "balance" else "account_transactions"
        return response, intent, "Neutral", False, "personal_banking"

    # 5. Intent classification
    intent_result    = classify_intent(user_message)
    sentiment_result = analyse_sentiment(user_message)

    # 6. Low-confidence on first message → ask to rephrase
    if intent_result["confidence"] < INTENT_CONFIDENCE_THRESHOLD and not history:
        return (
            "I'm not sure I understood that. Could you rephrase or give me more details?",
            "unclear",
            sentiment_result["label"],
            False,
            "all",
        )

    # 7. Context-enhanced ChromaDB retrieval
    msg_words = set(user_message.lower().split())
    is_contact = bool(msg_words & _CONTACT_KEYWORDS) and len(user_message.split()) <= 8
    if is_contact:
        retrieval_query = user_message
        category = "account"
    else:
        if len(history) >= 2 and len(user_message.split()) <= 8:
            retrieval_query = f"{history[-2][0]} {history[-1][0]} {user_message}"
        elif history:
            retrieval_query = f"{history[-1][0]} {user_message}"
        else:
            retrieval_query = user_message
        category = _intent_to_category(intent_result["intent"], intent_result["confidence"])

    docs = retrieve_top_k(retrieval_query, k=3, category=category)
    if not docs and category:
        docs = retrieve_top_k(retrieval_query, k=3)
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    # 8. History block (last 4 turns)
    history_block = ""
    if history:
        lines = [f"Customer: {u}\nAssistant: {b}" for u, b in history[-4:]]
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    # 9. Escalation or LLM generation
    if sentiment_result["escalate"]:
        response = ("I understand this is frustrating, and I'm escalating your request "
                    "to a human support agent right now. Someone will be with you shortly.")
    else:
        prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(
            few_shot=FEW_SHOT_EXAMPLES,
            history=history_block,
            context=context,
            query=user_message,
        )
        response = _clean_response(generate_response(prompt))

    return response, intent_result["intent"], sentiment_result["label"], sentiment_result["escalate"], category or "all"
