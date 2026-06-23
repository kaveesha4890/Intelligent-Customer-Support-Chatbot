from typing import TypedDict, Optional, List, Tuple


class AgentState(TypedDict):
    # ── inputs ─────────────────────────────────────────────
    user_message: str
    history: List[Tuple[str, str]]
    session_id: str

    # Authenticated customer ID from Flask session.
    # Set by run_agent() from flask.session — NEVER from user chat input.
    session_customer_id: Optional[str]

    # Pending multi-turn slot collection for banking service calculators.
    # Persisted across HTTP requests via flask.session — injected by run_agent().
    # Format: {"service": str, "slots": dict, "attempts": int} | None
    pending_calculation: Optional[dict]

    # Display name the customer chose to be called (e.g. "Alice").
    # IMPORTANT: This is PURELY COSMETIC — used only for warmth in LLM responses.
    # It is NOT the authenticated identity and must NEVER be used to look up
    # account data or bypass any security check. Keep completely independent
    # from session_customer_id, which is the verified security-relevant identity.
    customer_display_name: Optional[str]

    # The Flask request.host_url (e.g. "http://127.0.0.1:5000/") passed in by
    # the /chat route so banking_services_node can build deterministic product
    # page URLs via get_site_url(). Never generated or modified by any LLM node.
    host_url: Optional[str]

    # ── set by individual nodes ────────────────────────────
    intent: Optional[str]
    confidence: Optional[float]
    sentiment: Optional[str]
    escalated: Optional[bool]
    category: Optional[str]
    retrieved_docs: Optional[List[str]]
    response: Optional[str]
