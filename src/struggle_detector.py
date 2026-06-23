"""
Struggle detector — deterministic rule-based only, no ML or LLM.

Rule: if the same field_name appears in 2+ blur_empty events
      OR 2+ submit_fail events for this session+page, return a tip.

Tip messages are a static lookup table keyed by (page, field_name).
One line per entry — fully auditable.
"""

from collections import Counter
from typing import Optional

from src.ui_events_db import get_recent_events

# ── Tip lookup table ──────────────────────────────────────────────────────────
# Key: (page_key, field_name)  →  Value: short, actionable hint shown to customer
_TIPS: dict[tuple[str, str], str] = {
    # Personal Loan
    ("loans_personal",  "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("loans_personal",  "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("loans_personal",  "loan_amount"):   "Enter the amount you'd like to borrow in LKR, e.g. 500000.",
    ("loans_personal",  "tenure_months"): "Enter repayment period in months (max 60). E.g. 36 for 3 years.",
    # Housing Loan
    ("loans_housing",   "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("loans_housing",   "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("loans_housing",   "loan_amount"):   "Enter the housing loan amount in LKR, e.g. 5000000.",
    ("loans_housing",   "tenure_months"): "Enter repayment period in months (max 300). E.g. 120 for 10 years.",
    # Vehicle Loan
    ("loans_vehicle",   "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("loans_vehicle",   "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("loans_vehicle",   "loan_amount"):   "Enter the vehicle loan amount in LKR, e.g. 1500000.",
    ("loans_vehicle",   "tenure_months"): "Enter repayment period in months (max 84). E.g. 60 for 5 years.",
    # Education Loan
    ("loans_education", "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("loans_education", "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("loans_education", "loan_amount"):   "Enter the education loan amount in LKR, e.g. 300000.",
    ("loans_education", "tenure_months"): "Enter repayment period in months (max 120). E.g. 48 for 4 years.",
    # Business Loan
    ("loans_business",  "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("loans_business",  "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("loans_business",  "loan_amount"):   "Enter the business loan amount in LKR, e.g. 2000000.",
    ("loans_business",  "tenure_months"): "Enter repayment period in months (max 120). E.g. 60 for 5 years.",
    # Fixed Deposits
    ("fd",              "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("fd",              "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("fd",              "principal"):     "Enter the deposit amount in LKR (minimum LKR 10,000), e.g. 100000.",
    ("fd",              "tenure_months"): "Enter deposit period in months. Options: 3, 6, 12, 24, 36, or 60.",
    # Gold Pawning
    ("pawning",         "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("pawning",         "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("pawning",         "weight_grams"):  "Enter the weight of your gold item in grams, e.g. 10.5.",
    ("pawning",         "carat"):         "Select the gold purity: 18ct, 22ct, or 24ct.",
    # Cards
    ("cards",           "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("cards",           "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("cards",           "card_type"):     "Choose Debit (linked to your account), Credit, or Prepaid.",
    # Transfers
    ("transfers",       "full_name"):     "Enter your full legal name exactly as it appears on your NIC.",
    ("transfers",       "phone"):         "Enter your mobile number with area code, e.g. 0771234567.",
    ("transfers",       "amount"):        "Enter the transfer amount in LKR, e.g. 50000.",
    ("transfers",       "transfer_type"): "Choose Local SLIPS, Local CEFT, or Foreign Wire Transfer.",
}


def detect_struggle(session_id: str, page: str) -> Optional[dict]:
    """Check recent interaction events for this session+page for struggle patterns.

    Returns a tip dict {"field": ..., "message": ...} if a pattern is found,
    else None.  Rule-based only — no model or LLM judgment involved.
    """
    events = get_recent_events(session_id, page)

    blur_empty:  Counter[str] = Counter()
    submit_fail: Counter[str] = Counter()

    for ev in events:
        field = ev["field_name"]
        if not field:
            continue
        if ev["event_type"] == "blur_empty":
            blur_empty[field] += 1
        elif ev["event_type"] == "submit_fail":
            submit_fail[field] += 1

    # Check blur_empty first, then submit_fail (OR rule — either triggers independently)
    for counter in (blur_empty, submit_fail):
        for field, count in counter.most_common():
            if count >= 2:
                msg = _TIPS.get((page, field))
                if msg:
                    return {"field": field, "message": msg}

    return None
