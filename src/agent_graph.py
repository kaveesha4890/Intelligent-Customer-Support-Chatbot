import os
import re
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
    tool_calculate_fd_interest,
    tool_calculate_loan_emi,
    tool_calculate_pawning_advance,
    tool_calculate_transfer_fee,
    tool_get_fx_rate,
    tool_get_my_fixed_deposits,
    tool_get_my_loans,
    tool_get_my_pawning_records,
    tool_get_my_cards,
)
from src.slot_extraction import (
    extract_amount,
    extract_weight_grams,
    extract_tenure_months,
    extract_loan_type,
    extract_carat,
    extract_transfer_type,
    extract_currency_code,
)
from src.prompt_templates import CHAIN_OF_THOUGHT_TEMPLATE, FEW_SHOT_EXAMPLES, PERSONA_PROMPT
from src.site_routes import get_site_url
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

# Closing phrases that end a session — handled regardless of message length
_CLOSING_PHRASES = {
    "that's all", "thats all", "that's all i need", "thats all i need",
    "that's all i needed", "thats all i needed", "all i needed",
    "ok thanks", "okay thanks", "ok thank you", "okay thank you",
    "thanks that's all", "thanks thats all", "thank you that's all",
    "that will be all", "nothing else", "no more questions",
}

# Short contact-related queries that should always search the account/contact doc
_CONTACT_KEYWORDS = {"email", "contact", "reach", "phone", "number", "helpdesk", "support line", "message"}

# ── Banking service patterns ──────────────────────────────────────────────────

# Calculator triggers (hypothetical / rate-query phrasing — no login needed)
_FD_CALC_PATTERNS   = [
    "if i deposit", "fd interest", "fixed deposit interest", "how much will i get",
    "fd calculator", "deposit calculator", "earn if i put", "return on deposit",
    "fd rate", "fixed deposit rate", "interest on fd", "interest on fixed deposit",
]
_LOAN_CALC_PATTERNS = [
    "loan emi", "how much would i pay", "monthly payment", "emi for",
    "loan calculator", "emi calculator", "how much can i borrow",
    "loan interest rate", "monthly instalment", "monthly installment",
    "calculate emi", "calculate loan",
    "can i get a loan", "can i apply for a loan", "get a personal loan",
    "personal loan for", "housing loan for", "vehicle loan for",
    "apply for a loan", "take a loan", "borrow money", "loan for",
]
_PAWN_CALC_PATTERNS = [
    "pawning advance", "gold advance", "how much for my gold", "pawn my gold",
    "gold loan", "advance for gold", "pawning rate", "gold pawning",
    "how much can i get for gold", "gold advance rate",
]
_FEE_CALC_PATTERNS  = [
    "transfer fee", "how much to transfer", "fee for sending", "transfer charge",
    "slips fee", "ceft fee", "what is the fee", "fee for transfer",
    "transaction fee", "how much does it cost to transfer",
    "send overseas", "send abroad", "overseas transfer", "international transfer",
    "foreign transfer", "transfer abroad", "send money overseas",
    "cost to send", "how much to send",
]
_FX_CALC_PATTERNS   = [
    "exchange rate", "usd to lkr", "lkr to usd", "gbp to lkr", "eur to lkr",
    "forex", "foreign currency rate", "how much in lkr", "currency rate",
    "convert usd", "convert gbp", "convert eur", "convert dollars",
    "dollar rate", "pound rate", "euro rate",
]

# My Records triggers (personal / account phrasing — login required)
_MY_FD_PATTERNS     = ["my fixed deposit", "my fd", "my deposits", "my fd records"]
_MY_LOAN_PATTERNS   = ["my loan", "my loans", "loan balance", "my outstanding",
                        "my loan balance", "my loan records"]
_MY_PAWN_PATTERNS   = ["my pawning", "my pawn record", "my pawn", "my gold pawn",
                        "my pawning record"]
_MY_CARD_PATTERNS   = ["my card", "my cards", "my credit card", "card limit",
                        "available limit", "card balance", "my debit card"]

# Dispute phrases — if detected in the same message as a records query, escalate
_DISPUTE_PHRASES    = [
    "didn't take", "did not take", "didn't open", "did not open",
    "not mine", "not my loan", "not my card", "i didn't", "i did not",
    "incorrect balance", "wrong balance", "dispute", "unauthorized",
    "never applied", "never took", "don't recognize", "do not recognize",
    "this is wrong", "this is incorrect", "i never",
]

# ── Banking service helpers ───────────────────────────────────────────────────

def _extract_all_slots(service: str, msg: str) -> dict:
    """Extract all required slots for a service from the message text."""
    if service == "fd":
        return {
            "principal":     extract_amount(msg),
            "tenure_months": extract_tenure_months(msg),
        }
    if service == "loan":
        return {
            "loan_amount":   extract_amount(msg),
            "tenure_months": extract_tenure_months(msg),
            "loan_type":     extract_loan_type(msg),
        }
    if service == "pawning":
        return {
            "weight_grams":  extract_weight_grams(msg),
            "carat":         extract_carat(msg),
            "tenure_months": extract_tenure_months(msg),
        }
    if service == "transfer":
        return {
            "amount":        extract_amount(msg),
            "transfer_type": extract_transfer_type(msg),
        }
    if service == "fx":
        return {
            "currency_code": extract_currency_code(msg),
        }
    return {}


def _fill_missing_slots(service: str, msg: str, slots: dict) -> None:
    """Try to fill only the None slots in-place from the given message."""
    fresh = _extract_all_slots(service, msg)
    for k, v in fresh.items():
        if slots.get(k) is None and v is not None:
            slots[k] = v


def _run_calculation(service: str, slots: dict) -> dict:
    """Invoke the appropriate calculator tool and return its result dict."""
    if service == "fd":
        return tool_calculate_fd_interest.invoke({
            "principal":     slots["principal"],
            "tenure_months": slots["tenure_months"],
        })
    if service == "loan":
        return tool_calculate_loan_emi.invoke({
            "loan_amount":   slots["loan_amount"],
            "tenure_months": slots["tenure_months"],
            "loan_type":     slots["loan_type"],
        })
    if service == "pawning":
        return tool_calculate_pawning_advance.invoke({
            "weight_grams":  slots["weight_grams"],
            "carat":         slots["carat"],
            "tenure_months": slots["tenure_months"],
        })
    if service == "transfer":
        return tool_calculate_transfer_fee.invoke({
            "amount":        slots["amount"],
            "transfer_type": slots["transfer_type"],
        })
    if service == "fx":
        return tool_get_fx_rate.invoke({
            "currency_code": slots["currency_code"],
        })
    return {"error": "Unknown service."}


_TRANSFER_TYPE_LABELS = {
    "local_slips":  "Local SLIPS",
    "local_ceft":   "Local CEFT",
    "foreign_wire": "Foreign Wire Transfer",
}

_SHARED_DISCLAIMER = (
    "\n\n*This is an indicative estimate based on current published rates.*\n"
    "*This is not a commitment to approve or disburse — "
    "final terms require branch verification.*"
)

# ── URL injection helpers ─────────────────────────────────────────────────────
# Maps the internal service name (and loan_type slot) to a SITE_ROUTES key.
# The LLM never sees these URLs — they are appended in Python before the
# response is returned, using the same bypass principle as rates and balances.

# For my-records queries (service_key is the MY_CHECKS tuple label)
_MY_RECORDS_URL_KEY: dict[str, str] = {
    "fixed_deposits":  "fd_calculator",
    "pawning_records": "pawning_info",
    "cards":           "card_services",
    # "loans": intentionally omitted — no generic loans page; type unknown at this point
}


def _append_service_url(service: str, slots: dict, host_url: str) -> str:
    """Return a formatted URL line for the response, or '' if not applicable.

    Called only after a successful calculation — never in error or slot-question paths.
    The URL is produced deterministically from SITE_ROUTES; the LLM never generates it.
    """
    if not host_url:
        return ""
    if service == "fd":
        key = "fd_calculator"
    elif service == "loan":
        loan_type = (slots.get("loan_type") or "").lower()
        key = f"loan_{loan_type}"  # e.g. loan_personal, loan_housing — returns None if type missing
    elif service == "pawning":
        key = "pawning_info"
    elif service in ("transfer", "fx"):
        key = "money_transfer"
    else:
        return ""
    url = get_site_url(key, host_url)
    return f"\n\nView current rates and apply here: {url}" if url else ""


def _format_calc_result(service: str, result: dict) -> str:
    """Format a calculation result dict into a human-readable response string.

    Appends the required estimate + no-guarantee disclaimers unconditionally.
    All numeric values are computed by Python tools — the LLM never generates figures.
    """
    if service == "fd":
        r = result
        tenure_note = ""
        if r["requested_tenure"] != r["matched_tenure"]:
            tenure_note = f" (nearest available: {r['matched_tenure']} months)"
        text = (
            f"Fixed Deposit Estimate\n"
            f"{'=' * 36}\n"
            f"Principal      : LKR {r['principal']:>14,.2f}\n"
            f"Tenure         : {r['requested_tenure']} months{tenure_note}\n"
            f"Interest Rate  : {r['annual_rate']:.2f}% p.a.\n"
            f"Interest Earned: LKR {r['interest_earned']:>14,.2f}\n"
            f"Maturity Amount: LKR {r['maturity_amount']:>14,.2f}\n"
            f"\nNote: Early withdrawal attracts a {r['penalty_rate']:.2f}% penalty. "
            f"Contact a branch for details."
        )
        return text + _SHARED_DISCLAIMER

    if service == "loan":
        r = result
        text = (
            f"Loan EMI Estimate\n"
            f"{'=' * 36}\n"
            f"Loan Type      : {r['loan_type'].capitalize()} Loan\n"
            f"Principal      : LKR {r['principal']:>14,.2f}\n"
            f"Tenure         : {r['tenure_months']} months\n"
            f"Interest Rate  : {r['annual_rate']:.2f}% p.a. (reducing balance)\n"
            f"Monthly EMI    : LKR {r['emi']:>14,.2f}\n"
            f"Total Payable  : LKR {r['total_payable']:>14,.2f}\n"
            f"Total Interest : LKR {r['total_interest']:>14,.2f}\n"
            f"\nFinal EMI is subject to income verification, credit assessment, "
            f"and applicable processing charges."
        )
        return text + _SHARED_DISCLAIMER

    if service == "pawning":
        r = result
        text = (
            f"Pawning Advance Estimate\n"
            f"{'=' * 36}\n"
            f"Gold           : {r['carat']}ct, {r['weight_grams']:.1f}g\n"
            f"Advance Amount : LKR {r['advance_amount']:>14,.2f}\n"
            f"Tenure         : {r['tenure_months']} months\n"
            f"Monthly Interest: {r['monthly_interest_rate']:.2f}% per month\n"
            f"Total Interest : LKR {r['total_interest']:>14,.2f}\n"
            f"Total Payable  : LKR {r['total_payable']:>14,.2f}\n"
            f"\nAdvance is based on the bank's assessed rate, not the live gold "
            f"market price. Actual advance is subject to physical appraisal at branch."
        )
        return text + _SHARED_DISCLAIMER

    if service == "transfer":
        r = result
        label = _TRANSFER_TYPE_LABELS.get(r["transfer_type"], r["transfer_type"])
        fee_note = f"{r['fee_type'].capitalize()} fee" if r["fee_type"] == "fixed" \
                   else f"Percentage fee (min LKR 2,000.00)"
        text = (
            f"Transfer Fee Estimate\n"
            f"{'=' * 36}\n"
            f"Transfer Type  : {label}\n"
            f"Amount         : LKR {r['amount']:>14,.2f}\n"
            f"Fee            : LKR {r['fee']:>14,.2f}  ({fee_note})\n"
            f"Total          : LKR {r['total_amount']:>14,.2f}"
        )
        return text + _SHARED_DISCLAIMER

    if service == "fx":
        r = result
        code = r["currency_code"]
        rate = r["rate_to_lkr"]
        reverse = 1000 / rate  # how much foreign currency per LKR 1000
        text = (
            f"Exchange Rate\n"
            f"{'=' * 36}\n"
            f"1 {code}      = LKR {rate:,.2f}\n"
            f"LKR 1,000    = {code} {reverse:,.4f}\n"
            f"\nRates are indicative and updated periodically. "
            f"Actual rates at time of transaction may differ."
        )
        return text + _SHARED_DISCLAIMER

    return "Calculation complete."


_SLOT_QUESTIONS = {
    "fd": {
        "principal":     "How much would you like to deposit? (e.g. LKR 500,000)",
        "tenure_months": "For how long? (e.g. 12 months, 2 years)",
    },
    "loan": {
        "loan_amount":   "How much would you like to borrow? (in LKR)",
        "tenure_months": "For how long? (e.g. 36 months, 3 years)",
        "loan_type":     "What type of loan?\n(personal / housing / vehicle / education / business)",
    },
    "pawning": {
        "weight_grams":  "What is the weight of your gold item? (in grams, e.g. 15g)",
        "carat":         "What is the gold purity? (18ct, 22ct, or 24ct)",
        "tenure_months": "For how long? (in months, e.g. 6 months)",
    },
    "transfer": {
        "amount":        "How much would you like to transfer? (in LKR)",
        "transfer_type": "What type of transfer?\n(Local SLIPS / Local CEFT / Foreign Wire)",
    },
    "fx": {
        "currency_code": "Which foreign currency? (e.g. USD, GBP, EUR, AUD, SGD, INR, SAR)",
    },
}


def _build_slot_question(service: str, missing: list) -> str:
    """Ask for all missing slots in a single natural question."""
    q = _SLOT_QUESTIONS.get(service, {})
    if len(missing) == 1:
        return q.get(missing[0], "Could you provide more details?")
    items = "\n".join(f"• {q[k]}" for k in missing if k in q)
    return f"To calculate, I need a few more details:\n{items}"


# ── Personal account query patterns — detected before intent classification ───
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

# ── Display name extraction ───────────────────────────────────────────────────
# Purely cosmetic — result is stored in customer_display_name, NEVER in
# session_customer_id and NEVER used for account lookups or authentication.

_NAME_PREFIX_RE = re.compile(
    r"^(?:i'?m|i am|my name is|call me|it'?s|name'?s|they call me|you can call me)\s+",
    re.IGNORECASE,
)

# Words that indicate the user is asking a question rather than giving a name
_NON_NAME_WORDS = {
    # banking keywords
    "balance", "account", "transfer", "loan", "card", "deposit", "interest",
    "emi", "rate", "fee", "gold", "pawning", "what", "how", "can", "could",
    "would", "show", "tell", "check", "is", "are", "help",
    # greetings / short non-name words that pass the length check
    "hi", "hey", "hello", "ok", "okay", "yes", "no", "nope", "sure",
    "nothing", "none", "skip", "here", "there", "just", "fine", "good",
}


def _extract_display_name(text: str) -> str | None:
    """Extract a customer's preferred display name from a short free-text response.

    Uses deterministic regex only — no LLM. Returns None if the text looks like
    a banking question or a refusal rather than a name.
    """
    text = text.strip().rstrip(".!?,")
    # Remove common "my name is …" prefixes
    text = _NAME_PREFIX_RE.sub("", text)
    words = text.split()

    if not words:
        return None
    first = words[0].lower()
    # Reject very short/long first words, question marks, and banking keywords
    if len(first) < 2 or len(first) > 30 or "?" in text:
        return None
    if first in _NON_NAME_WORDS:
        return None
    # Accept up to 3 words (handles "Mary Jane", "de Silva", etc.)
    if len(words) > 4:
        return None
    return " ".join(w.capitalize() for w in words[:3])


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """Detect short acknowledgements ('okay', 'got it') and session-closing phrases
    ('that's all I need', 'ok thanks') after an informational answer."""
    if not history:
        return None
    normalized = text.strip().lower().rstrip("?!. ")
    if normalized in _ACKNOWLEDGEMENT_WORDS and len(text.split()) <= 4:
        return "Is there anything else I can help you with?"
    # Closing phrases — no word-count restriction
    if any(phrase in normalized for phrase in _CLOSING_PHRASES):
        return "You're welcome! Have a great day. Feel free to reach out anytime."
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
    after a closing bot message — both skip the full ML pipeline.

    Also handles one-time display name collection on the very first message of a
    new session. customer_display_name is PURELY COSMETIC and is kept completely
    separate from session_customer_id (the authenticated security identity).
    """
    history = state.get("history") or []
    display_name = state.get("customer_display_name")

    # ── Display name: ask on first message ───────────────────────────────────
    # Only when there is no prior history and no name has been collected yet.
    if not history and display_name is None:
        state["response"]  = "Hi! Before we get started, what should I call you?"
        state["intent"]    = "greeting"
        state["sentiment"] = "Neutral"
        state["escalated"] = False
        state["category"]  = "all"
        return state

    # ── Display name: extract from the very next reply ────────────────────────
    if (len(history) == 1
            and "what should i call you" in history[0][1].lower()
            and display_name is None):
        name = _extract_display_name(state["user_message"])
        if name:
            state["customer_display_name"] = name

        # Only give a greeting reply if the message doesn't contain a banking query.
        # If it does (e.g. "I'm Alice, what's my balance?"), fall through so the
        # banking pipeline handles the real question — name is already stored.
        msg_lower = state["user_message"].lower()
        has_banking = any(p in msg_lower for p in (
            _BALANCE_PATTERNS + _TXNS_PATTERNS +
            _FD_CALC_PATTERNS + _LOAN_CALC_PATTERNS + _PAWN_CALC_PATTERNS +
            _FEE_CALC_PATTERNS + _FX_CALC_PATTERNS +
            _MY_FD_PATTERNS + _MY_LOAN_PATTERNS + _MY_PAWN_PATTERNS + _MY_CARD_PATTERNS
        ))
        if not has_banking:
            greeting = f"Nice to meet you, {name}! " if name else ""
            state["response"]  = greeting + "How can I help you today?"
            state["intent"]    = "greeting"
            state["sentiment"] = "Neutral"
            state["escalated"] = False
            state["category"]  = "all"
            return state
        # has_banking → fall through with name already set in state

    # ── Normal chitchat detection ─────────────────────────────────────────────
    # 1. "okay" / "got it" after any informational answer
    reply = _check_acknowledgement(state["user_message"], history)
    # 2. "yes" / "sure" after a bot closing question
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


def banking_services_node(state: AgentState) -> AgentState:
    """Handle FD, loan, pawning, card, and transfer queries.

    Calculator mode: no login needed — rate card lookups + deterministic Python math.
    My-records mode: login required — customer_id sourced only from session_customer_id.

    All numeric figures are computed in Python and injected as formatted strings.
    generate_node is never reached for any of these queries.

    INFORMATION BOUNDARY: This node informs and calculates only.
    It never executes any financial transaction, disbursement, or record modification.
    """
    msg     = state["user_message"].lower()
    pending = state.get("pending_calculation")

    # ── 1. Resume pending multi-turn slot collection ──────────────────────────
    if pending:
        service  = pending["service"]
        slots    = dict(pending["slots"])
        attempts = pending.get("attempts", 1)

        # If the user is clearly asking for a completely different service,
        # clear pending and fall through to normal detection below.
        all_trigger_pats = (
            _MY_FD_PATTERNS + _MY_LOAN_PATTERNS + _MY_PAWN_PATTERNS + _MY_CARD_PATTERNS
            + _FD_CALC_PATTERNS + _LOAN_CALC_PATTERNS + _PAWN_CALC_PATTERNS
            + _FEE_CALC_PATTERNS + _FX_CALC_PATTERNS
        )
        changed_topic = any(p in msg for p in all_trigger_pats)

        if not changed_topic:
            _fill_missing_slots(service, msg, slots)
            missing = [k for k, v in slots.items() if v is None]

            # Recover personal_prefix / login_note carried from the first turn
            personal_prefix = pending.get("personal_prefix", "")
            login_note      = pending.get("login_note", "")

            if not missing:
                result = _run_calculation(service, slots)
                if "error" in result:
                    state["response"] = personal_prefix + result["error"]
                else:
                    url_suffix = _append_service_url(
                        service, slots, state.get("host_url") or ""
                    )
                    state["response"] = (
                        personal_prefix
                        + _format_calc_result(service, result)
                        + login_note
                        + url_suffix
                    )
                state["intent"]              = f"{service}_calculator"
                state["sentiment"]           = "Neutral"
                state["escalated"]           = False
                state["category"]            = "banking_services"
                state["pending_calculation"] = None
                return state

            # Still missing — ask again or give up
            attempts += 1
            if attempts > 2:
                state["response"] = (
                    "I wasn't able to collect all the details I need. "
                    "Please visit a branch or call us — our team will be happy to help."
                )
                state["intent"]              = f"{service}_calculator"
                state["sentiment"]           = "Neutral"
                state["escalated"]           = False
                state["category"]            = "banking_services"
                state["pending_calculation"] = None
                return state

            state["response"]            = _build_slot_question(service, missing)
            state["pending_calculation"] = {
                "service":         service,
                "slots":           slots,
                "attempts":        attempts,
                "personal_prefix": personal_prefix,
                "login_note":      login_note,
            }
            state["intent"]             = f"{service}_calculator"
            state["sentiment"]          = "Neutral"
            state["escalated"]          = False
            state["category"]           = "banking_services"
            return state

    # ── 2. Calculator queries ─────────────────────────────────────────────────
    # Checked BEFORE my-records so that messages matching both (e.g. "what is my
    # loan emi") go to the calculator path, which already shows personal records
    # via the "show both" logic — avoiding the LLM ever generating a financial figure.
    # For services that also have personal records, show existing records first
    # when the customer is logged in — they may be asking about their own account.
    _RECORDS_FOR_SERVICE = {
        "fd":      (tool_get_my_fixed_deposits,  "fixed deposits"),
        "loan":    (tool_get_my_loans,           "loans"),
        "pawning": (tool_get_my_pawning_records, "pawning records"),
    }
    # Transfer fees and FX rates have no personal record equivalent.

    CALC_CHECKS = [
        (_FD_CALC_PATTERNS,   "fd"),
        (_LOAN_CALC_PATTERNS, "loan"),
        (_PAWN_CALC_PATTERNS, "pawning"),
        (_FEE_CALC_PATTERNS,  "transfer"),
        (_FX_CALC_PATTERNS,   "fx"),
    ]
    for patterns, service in CALC_CHECKS:
        if any(p in msg for p in patterns):
            cid = state.get("session_customer_id") or ""

            # ── Personal records prefix ──────────────────────────────────────
            # Logged-in customers asking a calculator-style question may actually
            # want to see their existing account, not just a generic rate quote.
            # Show their records (if any) above the calculator result so both
            # the "check my account" and "explore new scenarios" intents are met.
            personal_prefix = ""
            login_note      = ""
            records_info    = _RECORDS_FOR_SERVICE.get(service)
            if records_info:
                records_tool, records_label = records_info
                if cid:
                    records_str = records_tool.invoke({"customer_id": cid})
                    # Only prepend when actual records exist (skip the "no records" message)
                    if not records_str.startswith("No "):
                        personal_prefix = records_str + "\n\n" + "─" * 36 + "\n\n"
                else:
                    # Unauthenticated — suggest login after the calculator result
                    login_note = (
                        f"\n\n*Log in to also view your existing {records_label} "
                        f"alongside this estimate.*"
                    )

            slots   = _extract_all_slots(service, msg)
            missing = [k for k, v in slots.items() if v is None]

            if not missing:
                result = _run_calculation(service, slots)
                if "error" in result:
                    state["response"] = personal_prefix + result["error"]
                else:
                    url_suffix = _append_service_url(
                        service, slots, state.get("host_url") or ""
                    )
                    state["response"] = (
                        personal_prefix
                        + _format_calc_result(service, result)
                        + login_note
                        + url_suffix
                    )
                state["intent"]              = f"{service}_calculator"
                state["sentiment"]           = "Neutral"
                state["escalated"]           = False
                state["category"]            = "banking_services"
                state["pending_calculation"] = None
                return state

            # Missing slots — ask and set pending state
            # (personal_prefix is not shown yet; will be combined when calc completes)
            state["response"]            = _build_slot_question(service, missing)
            state["pending_calculation"] = {
                "service":         service,
                "slots":           slots,
                "attempts":        1,
                "personal_prefix": personal_prefix,   # carry forward for final response
                "login_note":      login_note,
            }
            state["intent"]    = f"{service}_calculator"
            state["sentiment"] = "Neutral"
            state["escalated"] = False
            state["category"]  = "banking_services"
            return state

    # ── 3. My-records queries (login required) ────────────────────────────────
    # Checked AFTER calculators so that ambiguous messages like "what is my loan emi"
    # go to the calculator (with "show both"), not just plain record display.
    # Exception: card frustration phrases ("declined", "not working") bypass this
    # block and fall through to the LLM so the frustrated-customer few-shot fires.
    _CARD_FRUSTRATION = [
        "declined", "not working", "doesn't work", "wont work", "won't work",
        "rejected", "blocked by", "keeps declining", "keeps getting declined",
    ]

    MY_CHECKS = [
        (_MY_FD_PATTERNS,   tool_get_my_fixed_deposits,  "fixed_deposits"),
        (_MY_LOAN_PATTERNS, tool_get_my_loans,           "loans"),
        (_MY_PAWN_PATTERNS, tool_get_my_pawning_records, "pawning_records"),
        (_MY_CARD_PATTERNS, tool_get_my_cards,           "cards"),
    ]
    for patterns, tool_fn, service_key in MY_CHECKS:
        if any(p in msg for p in patterns):
            # Card + frustration phrase → give immediate troubleshooting steps directly.
            # Never ask the customer to explain the problem they just described.
            if service_key == "cards" and any(f in msg for f in _CARD_FRUSTRATION):
                display_name = state.get("customer_display_name")
                name_part    = f"{display_name}, " if display_name else ""
                state["response"] = (
                    f"{name_part}I hear you — let's get this sorted right now.\n\n"
                    "A card decline despite sufficient balance is usually one of these:\n"
                    "1. Daily spending limit reached — check Account > Cards > Spending Limits.\n"
                    "2. Merchant category blocked — some card plans restrict certain merchant types.\n"
                    "3. Temporary security hold — unusual spending patterns trigger an auto-hold.\n\n"
                    "Call our 24/7 line at 1-800-123-4567 if you need it unblocked immediately."
                )
                state["intent"]    = "card_not_working"
                state["sentiment"] = "Negative"
                state["escalated"] = False
                state["category"]  = "technical"
                return state

            cid = state.get("session_customer_id") or ""
            if not cid:
                label = service_key.replace("_", " ")
                state["response"]  = (
                    f"To view your {label}, please log in first. "
                    f"Use the 'Verify Identity' button or visit /login."
                )
                state["intent"]    = f"my_{service_key}"
                state["sentiment"] = "Neutral"
                state["escalated"] = False
                state["category"]  = "personal_banking"
                state["pending_calculation"] = None
                return state

            result = tool_fn.invoke({"customer_id": cid})

            # Detect dispute phrases in the same message
            is_dispute = any(p in msg for p in _DISPUTE_PHRASES)
            if is_dispute:
                result += (
                    "\n\n[!] I've noted that you're disputing this information. "
                    "This has been flagged for immediate review by a human specialist. "
                    "Please do not take any action — someone will contact you shortly."
                )

            # Append product page link (deterministic — never LLM-generated)
            _rec_key = _MY_RECORDS_URL_KEY.get(service_key)
            if _rec_key:
                _rec_url = get_site_url(_rec_key, state.get("host_url") or "")
                if _rec_url:
                    result += f"\n\nView product details and rates here: {_rec_url}"

            state["response"]            = result
            state["intent"]              = f"my_{service_key}"
            state["sentiment"]           = "Neutral"
            state["escalated"]           = is_dispute
            state["category"]            = "personal_banking"
            state["pending_calculation"] = None
            return state

    # ── 4. No service matched — fall through to intent_node ──────────────────
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
    """Build the persona + few-shot + history + context prompt and call the local LLM."""
    history = state.get("history") or []
    history_block = ""
    if history:
        lines = [f"Customer: {u}\nAssistant: {b}" for u, b in history[-4:]]
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    docs    = state.get("retrieved_docs") or []
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    # Inject customer's preferred name if available (purely cosmetic — see AgentState)
    display_name = state.get("customer_display_name")
    name_context = f"Customer's preferred name: {display_name}\n\n" if display_name else ""

    prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(
        persona=PERSONA_PROMPT,
        name_context=name_context,
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
    """After account check: if handled, go to END; otherwise check banking services."""
    return END if state.get("response") else "banking_services_node"


def route_banking_services(state: AgentState) -> str:
    """After banking services check: if response set, END (or escalate_node on dispute)."""
    if state.get("response"):
        return "escalate_node" if state.get("escalated") else END
    return "intent_node"


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

    graph.add_node("chitchat_node",         chitchat_node)
    graph.add_node("account_node",          account_node)
    graph.add_node("banking_services_node", banking_services_node)
    graph.add_node("intent_node",           intent_node)
    graph.add_node("sentiment_node",        sentiment_node)
    graph.add_node("clarify_node",          clarify_node)
    graph.add_node("escalate_node",         escalate_node)
    graph.add_node("retrieve_node",         retrieve_node)
    graph.add_node("generate_node",         generate_node)

    graph.set_entry_point("chitchat_node")

    graph.add_conditional_edges("chitchat_node", route_chitchat,
                                {"__end__": END, "account_node": "account_node"})
    graph.add_conditional_edges("account_node", route_account,
                                {"__end__": END, "banking_services_node": "banking_services_node"})
    graph.add_conditional_edges("banking_services_node", route_banking_services,
                                {"__end__": END, "escalate_node": "escalate_node",
                                 "intent_node": "intent_node"})
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
    pending_calculation: dict | None = None,
    customer_display_name: str | None = None,
    host_url: str = "",
) -> tuple:
    """Run the agent and return a 7-tuple:
      (response, intent, sentiment, escalated, category, pending_calculation, customer_display_name)

    session_customer_id:   authenticated customer from flask.session — never from user input.
    pending_calculation:   multi-turn slot state — from flask.session, returned so /chat can persist it.
    customer_display_name: purely cosmetic preferred name — from flask.session, returned so /chat can persist it.
                           NEVER used for authentication or account lookups.
    """
    result = agent.invoke({
        "user_message":          user_message,
        "history":               history,
        "session_id":            session_id,
        "session_customer_id":   session_customer_id,      # trusted — from Flask session
        "pending_calculation":   pending_calculation,      # trusted — from Flask session
        "customer_display_name": customer_display_name,   # cosmetic — from Flask session
        "host_url":              host_url,                # for deterministic URL injection only
        "intent":                None,
        "confidence":            None,
        "sentiment":             None,
        "escalated":             None,
        "category":              None,
        "retrieved_docs":        None,
        "response":              None,
    })
    return (
        result["response"],
        result["intent"]    or "unclear",
        result["sentiment"] or "Neutral",
        result["escalated"] or False,
        result["category"]  or "all",
        result.get("pending_calculation"),      # None = no pending; dict = still collecting slots
        result.get("customer_display_name"),    # None until customer provides it
    )
