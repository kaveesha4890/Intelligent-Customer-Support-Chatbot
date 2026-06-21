from sentence_transformers import util
from src.intent_classifier import classify_intent
from src.sentiment import analyse_sentiment
from src.retriever import retrieve_top_k, init_retriever
import src.retriever as _retriever
from src.prompt_templates import CHAIN_OF_THOUGHT_TEMPLATE, FEW_SHOT_EXAMPLES
from src.llm_generator import generate_response

INTENT_CONFIDENCE_THRESHOLD = 0.20
CATEGORY_ROUTE_THRESHOLD   = 0.50   # only route to a category when fairly confident

# Keywords in Banking77 intent names → our ChromaDB category
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
    """Map a Banking77 intent name to a ChromaDB category for filtered search."""
    if confidence < CATEGORY_ROUTE_THRESHOLD:
        return None   # not confident enough — search all categories
    intent_lower = intent.lower()
    for keywords, category in _CATEGORY_RULES:
        if any(k in intent_lower for k in keywords):
            return category
    return None
CHITCHAT_SIMILARITY_THRESHOLD = 0.75

# Prototype sentences that represent each chitchat category
_GREETING_PROTOTYPES = [
    "Hello", "Hi there", "Good morning", "Hey", "Greetings", "Hi", "Good evening",
]
_CLOSING_PROTOTYPES = [
    "Thank you", "Thanks", "Goodbye", "Bye", "That's all", "Cheers", "Thanks a lot",
]

# Embeddings are computed once after the retriever model is loaded
_greeting_vecs = None
_closing_vecs  = None


def _get_chitchat_vecs():
    """Lazily compute prototype embeddings using the shared MiniLM model."""
    global _greeting_vecs, _closing_vecs
    if _greeting_vecs is None:
        init_retriever()   # sets _retriever.embed_model
        _greeting_vecs = _retriever.embed_model.encode(_GREETING_PROTOTYPES, convert_to_tensor=True)
        _closing_vecs  = _retriever.embed_model.encode(_CLOSING_PROTOTYPES,  convert_to_tensor=True)


def _check_chitchat(text: str) -> str | None:
    """Return a canned reply if the message is a greeting or closing, else None."""
    _get_chitchat_vecs()
    vec = _retriever.embed_model.encode(text, convert_to_tensor=True)

    greeting_score = util.cos_sim(vec, _greeting_vecs).max().item()
    closing_score  = util.cos_sim(vec, _closing_vecs).max().item()

    if greeting_score > CHITCHAT_SIMILARITY_THRESHOLD:
        return "Hello! I'm your AI customer support assistant. How can I help you today?"
    if closing_score > CHITCHAT_SIMILARITY_THRESHOLD:
        return "You're welcome! Is there anything else I can help you with?"
    return None


def _clean_response(response: str) -> str:
    """Strip any leaked reasoning or labels from small model outputs."""
    for marker in ["Answer:", "Response:", "Step 4", "Step 4 -"]:
        if marker in response:
            response = response.split(marker, 1)[1]
    lines = response.strip().split("\n")
    if lines and lines[0].rstrip().endswith(":"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def chat(user_message: str, history: list):
    # 1. Embedding-based chitchat detection (greetings / closings)
    quick_reply = _check_chitchat(user_message)
    if quick_reply:
        return quick_reply, "greeting", "Neutral", False, "all"

    intent_result = classify_intent(user_message)
    sentiment_result = analyse_sentiment(user_message)

    # 2. Low-confidence intent → ask for clarification, but only on the first message.
    # Follow-up messages (history exists) are passed to the LLM which already has context.
    if intent_result["confidence"] < INTENT_CONFIDENCE_THRESHOLD and not history:
        return (
            "I'm not sure I understood that. Could you rephrase or give me more details?",
            "unclear",
            sentiment_result["label"],
            False,
            "all",
        )

    # Build an enriched query for ChromaDB by prepending the last user message.
    # This ensures "What is the email?" searches as "login problem What is the email?"
    # instead of matching phishing docs that mention "email".
    if history:
        last_user_msg = history[-1][0]
        retrieval_query = f"{last_user_msg} {user_message}"
    else:
        retrieval_query = user_message

    category = _intent_to_category(intent_result["intent"], intent_result["confidence"])
    docs = retrieve_top_k(retrieval_query, k=3, category=category)
    # Fallback: if category filter returned nothing, search all categories
    if not docs and category:
        docs = retrieve_top_k(retrieval_query, k=3)
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    # Build conversation history block (last 4 turns to keep prompt short)
    history_block = ""
    if history:
        recent = history[-4:]
        lines = []
        for user_turn, bot_turn in recent:
            lines.append(f"Customer: {user_turn}")
            lines.append(f"Assistant: {bot_turn}")
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

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
        response = generate_response(prompt)
        response = _clean_response(response)

    return response, intent_result["intent"], sentiment_result["label"], sentiment_result["escalate"], category or "all"
