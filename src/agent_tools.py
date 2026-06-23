from langchain_core.tools import tool
from src.intent_classifier import classify_intent
from src.sentiment import analyse_sentiment
from src.retriever import retrieve_top_k
from src.llm_generator import generate_response
from src.accounts_db import get_account_info, get_transactions
from src.services_db import (
    get_fd_rate, get_loan_rate, get_pawning_rate, get_transfer_fee, get_fx_rate,
    get_my_fixed_deposits, get_my_loans, get_my_pawning_records, get_my_cards,
)


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
        f"  Balance : LKR {info['balance']:,.2f}\n"
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
        lines.append(f"  {t['date']}  {t['description'][:32]:<32}  {sign}LKR {t['amount']:,.2f}")
    return "\n".join(lines)


# ── Banking service calculators (return dict; node formats display string) ────
# SECURITY: These tools perform calculations only. They never execute any
# financial transaction, disbursement, or record modification.

@tool
def tool_calculate_fd_interest(principal: float, tenure_months: int) -> dict:
    """Calculate indicative Fixed Deposit maturity amount.

    Looks up the nearest tenure bracket from the rate card and uses simple interest.
    Returns a dict with calculation details, or {'error': str} if no rate found.
    The LLM must never generate or guess any figure in these results.
    """
    rate_info = get_fd_rate(tenure_months)
    if rate_info is None:
        return {"error": "No FD rate data available. Please contact a branch."}

    matched_tenure = rate_info["tenure_months"]
    annual_rate    = rate_info["annual_rate"]
    interest       = principal * (annual_rate / 100) * (matched_tenure / 12)
    maturity       = principal + interest

    return {
        "principal":       round(principal, 2),
        "requested_tenure": tenure_months,
        "matched_tenure":  matched_tenure,
        "annual_rate":     annual_rate,
        "interest_earned": round(interest, 2),
        "maturity_amount": round(maturity, 2),
        "penalty_rate":    rate_info["early_withdrawal_penalty_rate"],
    }


@tool
def tool_calculate_loan_emi(loan_amount: float, tenure_months: int, loan_type: str) -> dict:
    """Calculate indicative loan EMI using reducing-balance (annuity) formula.

    Returns a dict with EMI and totals, or {'error': str} if loan type is invalid
    or tenure exceeds the maximum for that loan type.
    The LLM must never generate or guess any figure in these results.
    """
    rate_info = get_loan_rate(loan_type.lower())
    if rate_info is None:
        return {
            "error": (
                f"No rate available for '{loan_type}' loans. "
                "Valid types: personal, housing, vehicle, education, business."
            )
        }

    max_tenure  = rate_info["max_tenure_months"]
    annual_rate = rate_info["annual_rate"]

    if tenure_months > max_tenure:
        return {
            "error": (
                f"Maximum tenure for {loan_type} loans is {max_tenure} months "
                f"({max_tenure // 12} years)."
            )
        }

    r   = annual_rate / 12 / 100
    n   = tenure_months
    emi = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_payable  = emi * n
    total_interest = total_payable - loan_amount

    return {
        "loan_type":       loan_type.lower(),
        "annual_rate":     annual_rate,
        "principal":       round(loan_amount, 2),
        "tenure_months":   tenure_months,
        "emi":             round(emi, 2),
        "total_payable":   round(total_payable, 2),
        "total_interest":  round(total_interest, 2),
    }


@tool
def tool_calculate_pawning_advance(weight_grams: float, carat: int, tenure_months: int) -> dict:
    """Calculate indicative pawning advance and interest (simple interest).

    rate_per_gram is the bank's advance rate, not the live gold market price.
    interest = advance * monthly_rate * tenure_months (simple interest).
    Returns a dict with advance details, or {'error': str} if carat is invalid.
    The LLM must never generate or guess any figure in these results.
    """
    rate_info = get_pawning_rate(carat)
    if rate_info is None:
        return {
            "error": (
                f"No rate available for {carat}ct gold. "
                "Valid carats: 18, 22, 24."
            )
        }

    advance          = weight_grams * rate_info["rate_per_gram"]
    monthly_rate_pct = rate_info["monthly_interest_rate"]
    total_interest   = advance * (monthly_rate_pct / 100) * tenure_months
    total_payable    = advance + total_interest

    return {
        "carat":                carat,
        "weight_grams":         weight_grams,
        "rate_per_gram":        rate_info["rate_per_gram"],
        "advance_amount":       round(advance, 2),
        "monthly_interest_rate": monthly_rate_pct,
        "tenure_months":        tenure_months,
        "total_interest":       round(total_interest, 2),
        "total_payable":        round(total_payable, 2),
    }


@tool
def tool_calculate_transfer_fee(amount: float, transfer_type: str) -> dict:
    """Calculate indicative transfer fee for a given amount and transfer type.

    transfer_type must be one of: 'local_slips', 'local_ceft', 'foreign_wire'.
    For 'percent' fee types a minimum fee may apply (e.g. foreign wire).
    Returns a dict with fee breakdown, or {'error': str} if no fee schedule found.
    The LLM must never generate or guess any figure in these results.
    """
    fee_info = get_transfer_fee(transfer_type.lower(), amount)
    if fee_info is None:
        return {
            "error": (
                f"No fee schedule found for '{transfer_type}' at LKR {amount:,.2f}. "
                "Valid types: local_slips, local_ceft, foreign_wire."
            )
        }

    if fee_info["fee_type"] == "fixed":
        fee = fee_info["fee_value"]
    else:  # percent
        fee = amount * fee_info["fee_value"] / 100
        min_fee = fee_info.get("min_fee")
        if min_fee and fee < min_fee:
            fee = min_fee

    return {
        "transfer_type": transfer_type.lower(),
        "amount":        round(amount, 2),
        "fee_type":      fee_info["fee_type"],
        "fee":           round(fee, 2),
        "total_amount":  round(amount + fee, 2),
    }


@tool
def tool_get_fx_rate(currency_code: str) -> dict:
    """Return the indicative exchange rate to LKR for a given currency code.

    currency_code must be one of: USD, GBP, EUR, AUD, SGD, INR, SAR.
    Returns a dict with rate details, or {'error': str} if code not found.
    Rates are simulated demo data — not live market rates.
    """
    rate_info = get_fx_rate(currency_code.upper())
    if rate_info is None:
        return {
            "error": (
                f"No rate found for '{currency_code.upper()}'. "
                "Available currencies: USD, GBP, EUR, AUD, SGD, INR, SAR."
            )
        }
    return dict(rate_info)


# ── My records tools (return str; customer_id MUST come from session only) ────
# SECURITY: customer_id is sourced exclusively from AgentState['session_customer_id'],
# which flows from flask.session via run_agent(). The LLM never supplies this value.
# These tools display records only — they never modify any data.

_DISPUTE_FOOTER = (
    "\n---\n"
    "If any information above appears incorrect, please contact us immediately "
    "and we will escalate your concern to a specialist."
)


@tool
def tool_get_my_fixed_deposits(customer_id: str) -> str:
    """Return all Fixed Deposit records for an authenticated customer (display only).

    SECURITY: customer_id must come from AgentState['session_customer_id'].
    The LLM never supplies this value.
    """
    if not customer_id:
        return "To view your fixed deposits, please verify your identity first by logging in."

    records = get_my_fixed_deposits(customer_id)
    if not records:
        return "No fixed deposit records found on your account."

    lines = [f"Your Fixed Deposits ({len(records)} record{'s' if len(records)>1 else ''}):", ""]
    for i, r in enumerate(records, 1):
        lines.append(f"{i}. {r['fd_id']}")
        lines.append(f"   Principal    : LKR {r['principal']:>14,.2f}  |  {r['tenure_months']} months @ {r['annual_rate']:.2f}% p.a.")
        lines.append(f"   Period       : {r['start_date']}  →  {r['maturity_date']}")
        lines.append(f"   Maturity Amt : LKR {r['maturity_amount']:>14,.2f}")
        lines.append(f"   Status       : {r['status'].capitalize()}")
        lines.append("")

    lines.append(_DISPUTE_FOOTER)
    return "\n".join(lines)


@tool
def tool_get_my_loans(customer_id: str) -> str:
    """Return all loan records for an authenticated customer (display only).

    SECURITY: customer_id must come from AgentState['session_customer_id'].
    The LLM never supplies this value.
    """
    if not customer_id:
        return "To view your loans, please verify your identity first by logging in."

    records = get_my_loans(customer_id)
    if not records:
        return "No active loan records found on your account."

    lines = [f"Your Loans ({len(records)} record{'s' if len(records)>1 else ''}):", ""]
    for i, r in enumerate(records, 1):
        lines.append(f"{i}. {r['loan_id']}  —  {r['loan_type'].capitalize()} Loan")
        lines.append(f"   Original Principal : LKR {r['principal']:>14,.2f}")
        lines.append(f"   Outstanding Balance: LKR {r['outstanding_balance']:>14,.2f}")
        lines.append(f"   Monthly Installment: LKR {r['monthly_installment']:>14,.2f}")
        lines.append(f"   Tenure             : {r['tenure_months']} months  |  Since: {r['start_date']}")
        lines.append("")

    lines.append(_DISPUTE_FOOTER)
    return "\n".join(lines)


@tool
def tool_get_my_pawning_records(customer_id: str) -> str:
    """Return all pawning records for an authenticated customer (display only).

    SECURITY: customer_id must come from AgentState['session_customer_id'].
    The LLM never supplies this value.
    """
    if not customer_id:
        return "To view your pawning records, please verify your identity first by logging in."

    records = get_my_pawning_records(customer_id)
    if not records:
        return "No pawning records found on your account."

    lines = [f"Your Pawning Records ({len(records)} record{'s' if len(records)>1 else ''}):", ""]
    for i, r in enumerate(records, 1):
        lines.append(f"{i}. {r['pawn_id']}")
        lines.append(f"   Item         : {r['item_description']}")
        lines.append(f"   Gold         : {r['carat']}ct, {r['weight_grams']:.1f}g")
        lines.append(f"   Advance      : LKR {r['advance_amount']:>14,.2f}")
        lines.append(f"   Interest Rate: {r['interest_rate']:.2f}% per month")
        lines.append(f"   Period       : {r['pawn_date']}  →  {r['due_date']}")
        lines.append(f"   Status       : {r['status'].capitalize()}")
        lines.append("")

    lines.append(_DISPUTE_FOOTER)
    return "\n".join(lines)


@tool
def tool_get_my_cards(customer_id: str) -> str:
    """Return all card records for an authenticated customer (display only).

    SECURITY: customer_id must come from AgentState['session_customer_id'].
    The LLM never supplies this value.
    """
    if not customer_id:
        return "To view your cards, please verify your identity first by logging in."

    records = get_my_cards(customer_id)
    if not records:
        return "No card records found on your account."

    lines = [f"Your Cards ({len(records)} record{'s' if len(records)>1 else ''}):", ""]
    for i, r in enumerate(records, 1):
        lines.append(f"{i}. {r['card_type'].capitalize()} Card  —  {r['masked_number']}")
        if r["card_type"] == "credit":
            lines.append(f"   Credit Limit   : LKR {r['credit_limit']:>12,.2f}")
            lines.append(f"   Available Limit: LKR {r['available_limit']:>12,.2f}")
        lines.append(f"   Status         : {r['status'].capitalize()}")
        lines.append("")

    lines.append(_DISPUTE_FOOTER)
    return "\n".join(lines)
