"""
Deterministic regex-based slot extractors for banking service calculations.

All functions return None when the slot cannot be reliably extracted —
callers must treat None as "slot still missing" and ask the user.

No LLM is used here. No defaults are assumed. If a value is ambiguous, None is returned.
"""

import re
from typing import Optional

# ── Amount (LKR) ─────────────────────────────────────────────────────────────

def extract_amount(text: str) -> Optional[float]:
    """Extract a monetary amount from text.

    Handles: 500000 | 500,000 | 500k | 500K | 5 lakh | 5 lakhs | 5.5 lakhs
    """
    t = text.lower()

    # "N lakh(s)" — highest priority (ambiguous with bare N if text has both)
    m = re.search(r'(\d+(?:\.\d+)?)\s*lakh', t)
    if m:
        return float(m.group(1)) * 100_000

    # "Nk" or "NK"
    m = re.search(r'(\d+(?:\.\d+)?)\s*k\b', t)
    if m:
        return float(m.group(1)) * 1_000

    # Bare number with optional commas: 500,000 or 500000
    m = re.search(r'\b(\d{1,3}(?:,\d{3})+|\d{4,})\b', t)
    if m:
        return float(m.group(1).replace(",", ""))

    # Small 3-digit number (e.g. "500 LKR" or just "500")
    m = re.search(r'\b(\d{3})\b', t)
    if m:
        return float(m.group(1))

    return None


# ── Weight (grams) ────────────────────────────────────────────────────────────

def extract_weight_grams(text: str) -> Optional[float]:
    """Extract a weight in grams: '15g', '15 grams', '15.5 gram'."""
    t = text.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:grams?|g)\b', t)
    return float(m.group(1)) if m else None


# ── Tenure (months) ───────────────────────────────────────────────────────────

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1,  # "a year"
}

def extract_tenure_months(text: str) -> Optional[int]:
    """Extract tenure in months.

    Handles: '6 months', '2 years', 'one year', 'six months', 'half a year',
             '1.5 years', '18 months', 'a year'
    """
    t = text.lower()

    # "half a year" or "half year"
    if re.search(r'half\s+(?:a\s+)?year', t):
        return 6

    # "N.N years" or "N years"
    m = re.search(r'(\d+(?:\.\d+)?)\s*years?', t)
    if m:
        return round(float(m.group(1)) * 12)

    # "word year(s)"
    pattern = "(" + "|".join(_WORD_TO_NUM.keys()) + r")\s+years?"
    m = re.search(pattern, t)
    if m:
        return _WORD_TO_NUM[m.group(1)] * 12

    # "N months"
    m = re.search(r'(\d+)\s*months?', t)
    if m:
        return int(m.group(1))

    # "word months"
    pattern = "(" + "|".join(_WORD_TO_NUM.keys()) + r")\s+months?"
    m = re.search(pattern, t)
    if m:
        return _WORD_TO_NUM[m.group(1)]

    return None


# ── Loan type ─────────────────────────────────────────────────────────────────

_LOAN_KEYWORDS = {
    "personal":  ["personal"],
    "housing":   ["housing", "home loan", "home-loan", "mortgage", "house loan"],
    "vehicle":   ["vehicle", "car loan", "car-loan", "auto loan", "bike loan", "motor"],
    "education": ["education", "student loan", "study loan", "educational"],
    "business":  ["business", "sme", "commercial"],
}

def extract_loan_type(text: str) -> Optional[str]:
    """Return one of: 'personal' | 'housing' | 'vehicle' | 'education' | 'business' | None."""
    t = text.lower()
    for loan_type, keywords in _LOAN_KEYWORDS.items():
        if any(k in t for k in keywords):
            return loan_type
    return None


# ── Gold carat ────────────────────────────────────────────────────────────────

_VALID_CARATS = {18, 22, 24}

def extract_carat(text: str) -> Optional[int]:
    """Return one of: 18 | 22 | 24 | None.

    Accepts: '22ct', '22 carat', '22 karat', '22kt', '22k gold', '24ct pure gold'
    Rejects numbers that don't correspond to valid gold carats.
    """
    t = text.lower()

    # "pure gold" → 24ct
    if "pure gold" in t or "24 pure" in t:
        return 24

    # N followed by carat/karat/ct/kt variant
    m = re.search(r'\b(\d{2})\s*(?:carat|karat|ct|kt)\b', t)
    if m:
        c = int(m.group(1))
        return c if c in _VALID_CARATS else None

    # Bare "18k gold" / "22k gold" (k must precede 'gold' or be word-final near 'gold')
    m = re.search(r'\b(\d{2})\s*k\b(?:\s*gold)?', t)
    if m:
        c = int(m.group(1))
        return c if c in _VALID_CARATS else None

    # Last resort: bare valid carat number when 'gold' is present
    if "gold" in t:
        for c in [24, 22, 18]:
            if re.search(rf'\b{c}\b', t):
                return c

    return None


# ── Transfer type ─────────────────────────────────────────────────────────────

def extract_transfer_type(text: str) -> Optional[str]:
    """Return one of: 'local_slips' | 'local_ceft' | 'foreign_wire' | None."""
    t = text.lower()
    if any(k in t for k in ["foreign", "international", "overseas", "wire transfer",
                             "swift", "abroad", "foreign wire"]):
        return "foreign_wire"
    if "ceft" in t:
        return "local_ceft"
    if any(k in t for k in ["slips", "local transfer", "local bank transfer"]):
        return "local_slips"
    return None


# ── Currency code ─────────────────────────────────────────────────────────────

_CURRENCY_MAP: dict[str, str] = {
    "usd": "USD", "dollar": "USD", "us dollar": "USD", "american dollar": "USD", "dollars": "USD",
    "gbp": "GBP", "pound": "GBP", "british pound": "GBP", "sterling": "GBP", "pounds": "GBP",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "aud": "AUD", "australian dollar": "AUD", "aussie dollar": "AUD", "australian dollars": "AUD",
    "sgd": "SGD", "singapore dollar": "SGD", "sing dollar": "SGD",
    "inr": "INR", "indian rupee": "INR", "rupee": "INR", "indian rupees": "INR",
    "sar": "SAR", "saudi riyal": "SAR", "riyal": "SAR", "saudi": "SAR",
}

def extract_currency_code(text: str) -> Optional[str]:
    """Return an uppercase 3-letter ISO currency code or None.

    Handles explicit codes (USD, GBP) and common currency names (dollar, pound, euro).
    """
    # Try explicit 3-letter code first (case-insensitive)
    m = re.search(r'\b(USD|GBP|EUR|AUD|SGD|INR|SAR)\b', text.upper())
    if m:
        return m.group(1)

    # Try common currency names (longest match first to avoid 'US' matching 'US dollar' partially)
    t = text.lower()
    for key in sorted(_CURRENCY_MAP, key=len, reverse=True):
        if key in t:
            return _CURRENCY_MAP[key]

    return None
