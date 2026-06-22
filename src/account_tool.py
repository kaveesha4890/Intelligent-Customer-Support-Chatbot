"""
Account helper functions for the fixed-pipeline chatbot.

SECURITY INVARIANT
==================
Both functions receive `customer_id` as a parameter.  That value MUST always
originate from the server-side Flask session (flask.session['authenticated_customer_id']),
extracted by ui/app.py and passed into chatbot_pipeline.chat() as
`session_customer_id`.

It must NEVER be read from the user's chat message.  chatbot_pipeline.py
calls these functions only after detecting a balance/transaction query, and
it always passes session_customer_id — never anything parsed from user input.

If session_customer_id is empty/None the functions return a "please verify"
message — they fail CLOSED, never leaking internals or raising exceptions.

DEMO NOTE: Operates on simulated/dummy data in accounts.db (see seed_accounts.py).
"""

from src.accounts_db import get_account_info, get_transactions


def get_account_balance(customer_id: str) -> str:
    """Return a formatted balance string for an authenticated customer.

    SECURITY: customer_id must come from Flask session, never from user chat input.
    """
    if not customer_id:
        return (
            "To check your balance, please verify your identity first.\n"
            "Click the 'Verify Identity' button above or visit /login."
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


def get_recent_transactions(customer_id: str, limit: int = 5) -> str:
    """Return formatted recent transactions for an authenticated customer.

    SECURITY: customer_id must come from Flask session, never from user chat input.
    """
    if not customer_id:
        return (
            "To view your transactions, please verify your identity first.\n"
            "Click the 'Verify Identity' button above or visit /login."
        )

    txns = get_transactions(customer_id, limit=limit)
    if not txns:
        return "No recent transactions found on your account."

    lines = [f"Your {len(txns)} most recent transactions:\n"]
    for t in txns:
        sign = "+" if t["txn_type"] == "credit" else "-"
        desc = t["description"][:32]
        lines.append(f"  {t['date']}  {desc:<32}  {sign}${t['amount']:,.2f}")

    return "\n".join(lines)
