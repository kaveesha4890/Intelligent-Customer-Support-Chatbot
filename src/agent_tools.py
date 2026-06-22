from langchain_core.tools import tool
from src.intent_classifier import classify_intent
from src.sentiment import analyse_sentiment
from src.retriever import retrieve_top_k
from src.llm_generator import generate_response
from src.accounts_db import get_account_info, get_transactions


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


@tool
def tool_get_account_balance(customer_id: str) -> str:
    """Return the account balance for an authenticated customer.

    SECURITY: customer_id must come from AgentState['session_customer_id'],
    which is sourced from the Flask session by run_agent(). The graph node
    passes it directly — the LLM never supplies this value.
    """
    if not customer_id:
        return (
            "To check your balance, please verify your identity first.\n"
            "Click the 'Verify Identity' button or visit /login."
        )
    info = get_account_info(customer_id)
    if info is None:
        return "Account information is unavailable. Please contact support."
    return (
        f"Your account balance:\n"
        f"  Account : {info['masked_account_no']} ({info['account_type']})\n"
        f"  Balance : ${info['balance']:,.2f}\n"
        f"  Last transaction : {info['last_txn_date'] or 'N/A'}"
    )


@tool
def tool_get_recent_transactions(customer_id: str) -> str:
    """Return the 5 most recent transactions for an authenticated customer.

    SECURITY: customer_id must come from AgentState['session_customer_id'],
    which is sourced from the Flask session by run_agent(). The graph node
    passes it directly — the LLM never supplies this value.
    """
    if not customer_id:
        return (
            "To view your transactions, please verify your identity first.\n"
            "Click the 'Verify Identity' button or visit /login."
        )
    txns = get_transactions(customer_id, limit=5)
    if not txns:
        return "No recent transactions found on your account."
    lines = [f"Your {len(txns)} most recent transactions:\n"]
    for t in txns:
        sign = "+" if t["txn_type"] == "credit" else "-"
        lines.append(f"  {t['date']}  {t['description'][:32]:<32}  {sign}${t['amount']:,.2f}")
    return "\n".join(lines)
