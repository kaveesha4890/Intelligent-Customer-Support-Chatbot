"""
AutoTrust Bank — Site Route Registry
=====================================
Deterministic mapping from service/intent identifiers to real URL paths.

IMPORTANT: The chatbot must ALWAYS use get_site_url() to produce links.
It must NEVER invent, guess, or alter these URLs. The LLM never sees this
module — URLs are injected into the formatted response string in Python,
using the same bypass principle as interest rates and account balances.
"""

from typing import Optional

SITE_ROUTES: dict[str, str] = {
    "loan_personal":  "/loans/personal",
    "loan_housing":   "/loans/housing",
    "loan_vehicle":   "/loans/vehicle",
    "loan_education": "/loans/education",
    "loan_business":  "/loans/business",
    "fd_calculator":  "/deposits/fixed-deposits",
    "pawning_info":   "/services/pawning",
    "card_services":  "/cards",
    "money_transfer": "/transfers",
}


def get_site_url(service_key: str, base_url: str) -> Optional[str]:
    """Return the full URL for a service key, or None if not found.

    Args:
        service_key: One of the keys defined in SITE_ROUTES above.
        base_url:    The request's host URL, e.g. request.host_url from Flask
                     (includes scheme + host + trailing slash).

    Returns:
        A full absolute URL string, or None if service_key is not registered.
        Returning None is the correct no-op — callers must check for None before
        inserting a link, so a missing key never produces a broken URL.
    """
    path = SITE_ROUTES.get(service_key)
    if not path or not base_url:
        return None
    return f"{base_url.rstrip('/')}{path}"
