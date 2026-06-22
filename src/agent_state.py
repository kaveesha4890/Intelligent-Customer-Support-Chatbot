from typing import TypedDict, Optional, List, Tuple


class AgentState(TypedDict):
    # ── inputs ─────────────────────────────────────────────
    user_message: str
    history: List[Tuple[str, str]]
    session_id: str

    # ── set by individual nodes ────────────────────────────
    intent: Optional[str]
    confidence: Optional[float]
    sentiment: Optional[str]
    escalated: Optional[bool]
    category: Optional[str]
    retrieved_docs: Optional[List[str]]
    response: Optional[str]
