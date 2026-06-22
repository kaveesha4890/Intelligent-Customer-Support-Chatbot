from langchain_core.tools import tool
from src.intent_classifier import classify_intent
from src.sentiment import analyse_sentiment
from src.retriever import retrieve_top_k
from src.llm_generator import generate_response


@tool
def tool_classify_intent(message: str) -> dict:
    """Classify the banking support intent from a customer message.
    Returns a dict with 'intent' (str) and 'confidence' (float)."""
    return classify_intent(message)


@tool
def tool_analyse_sentiment(message: str) -> dict:
    """Detect the emotional tone of a customer message and whether escalation is needed.
    Returns a dict with 'label' (str) and 'escalate' (bool)."""
    return analyse_sentiment(message)


@tool
def tool_search_knowledge_base(query: str, category: str = "") -> list:
    """Search the customer support FAQ knowledge base for relevant documents.
    Use category to narrow the search: 'fraud', 'billing', 'technical', 'account', or '' for all."""
    return retrieve_top_k(query, k=3, category=category if category else None)


@tool
def tool_generate_response(prompt: str) -> str:
    """Generate a customer support response using the local Llama LLM."""
    return generate_response(prompt)
