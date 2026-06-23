#!/usr/bin/env python3
"""
DEMO DATA SEEDER — BANKING SERVICES — FOR DEMONSTRATION / ACADEMIC PURPOSES ONLY
=================================================================================
Creates SIMULATED / DUMMY rate cards and customer banking product records.
ALL DATA IS COMPLETELY FICTIONAL.

Run:          python seed_services.py
Reset+reseed: python seed_services.py --force
"""

import sys
import os
import math
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.services_db import init_services_db, _DB_PATH, _conn

# ── Rate cards ────────────────────────────────────────────────────────────────

FD_RATES = [
    # (tenure_months, annual_rate_pct, early_withdrawal_penalty_pct)
    (3,   9.00, 2.00),
    (6,   9.75, 2.00),
    (12, 10.50, 1.50),
    (24, 11.00, 1.00),
    (36, 11.50, 1.00),
    (60, 11.25, 1.00),
]

LOAN_RATES = [
    # (loan_type, annual_rate_pct, max_tenure_months)
    ("personal",  18.00,  60),
    ("housing",   12.50, 300),
    ("vehicle",   14.00,  84),
    ("education", 13.50, 120),
    ("business",  16.00, 120),
]

PAWNING_RATES = [
    # (carat, rate_per_gram_lkr, ltv_pct, monthly_interest_rate_pct)
    # rate_per_gram = bank advance rate in LKR/g (NOT live market price)
    # e.g. 22ct market ~LKR 17,500/g x 75% LTV → advance ~13,000/g
    (18,  9_500.00, 72.0, 2.00),
    (22, 13_000.00, 75.0, 2.00),
    (24, 14_200.00, 78.0, 1.75),
]

TRANSFER_FEES = [
    # (transfer_type, min_amount, max_amount_or_None, fee_type, fee_value, min_fee_or_None)
    ("local_slips",   0,       100_000, "fixed",    50.00, None),
    ("local_slips",   100_001, 500_000, "fixed",   100.00, None),
    ("local_slips",   500_001, None,    "fixed",   200.00, None),
    ("local_ceft",    0,        50_000, "fixed",    25.00, None),
    ("local_ceft",    50_001,   None,   "fixed",    50.00, None),
    ("foreign_wire",  0,        None,   "percent",   0.25, 2000.00),
]

FX_RATES = [
    # (currency_code, rate_to_lkr, updated_date)
    # Approximate rates as of 2025 — clearly marked as simulated
    ("USD", 325.00, "2025-01-01"),
    ("GBP", 415.00, "2025-01-01"),
    ("EUR", 355.00, "2025-01-01"),
    ("AUD", 215.00, "2025-01-01"),
    ("SGD", 245.00, "2025-01-01"),
    ("INR",   3.90, "2025-01-01"),
    ("SAR",  86.50, "2025-01-01"),
]

# ── Calculation helpers ────────────────────────────────────────────────────────

def _emi(principal: float, annual_rate: float, months: int) -> float:
    """Reducing-balance EMI formula."""
    r = annual_rate / 12 / 100
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

def _fd_maturity(principal: float, annual_rate: float, months: int) -> float:
    """Simple interest FD maturity amount."""
    return principal * (1 + annual_rate / 100 * months / 12)

def _outstanding(principal: float, annual_rate: float, months: int, paid: int) -> float:
    """Outstanding loan balance after 'paid' installments."""
    r = annual_rate / 12 / 100
    emi = _emi(principal, annual_rate, months)
    remaining = months - paid
    if remaining <= 0:
        return 0.0
    return (emi / r) * (1 - (1 + r) ** (-remaining))

def _date(base: str, add_months: int) -> str:
    """Add calendar months to a YYYY-MM-DD date string."""
    dt = datetime.strptime(base, "%Y-%m-%d")
    m = dt.month - 1 + add_months
    year = dt.year + m // 12
    month = m % 12 + 1
    day = min(dt.day, [31,28,31,30,31,30,31,31,30,31,30,31][month-1])
    return f"{year:04d}-{month:02d}-{day:02d}"

def _months_between(start: str, end: str = "2026-06-22") -> int:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    return max(0, (e.year - s.year) * 12 + (e.month - s.month))

# ── Customer product data ─────────────────────────────────────────────────────

# Fixed Deposits: (fd_id, customer_id, principal, tenure_months, annual_rate, start_date, status)
# maturity_date and maturity_amount are computed
_FD_DATA = [
    # DEMO001 — Alice Johnson — Checking
    ("FD-D001-01", "DEMO001", 500_000.00, 12, 10.50, "2025-06-01", "matured"),
    ("FD-D001-02", "DEMO001", 250_000.00, 24, 11.00, "2025-12-01", "active"),
    # DEMO002 — Bob Smith — Savings
    ("FD-D002-01", "DEMO002", 1_000_000.00, 36, 11.50, "2024-09-01", "active"),
    # DEMO003 — Carol White — Checking (small balance)
    ("FD-D003-01", "DEMO003", 100_000.00,  6,  9.75, "2026-01-01", "active"),
    # DEMO004 — David Brown — Premium
    ("FD-D004-01", "DEMO004", 2_000_000.00, 60, 11.25, "2023-07-01", "active"),
    ("FD-D004-02", "DEMO004",   500_000.00, 12, 10.50, "2025-01-01", "matured"),
    ("FD-D004-03", "DEMO004",   750_000.00, 24, 11.00, "2025-10-01", "active"),
    # DEMO005 — Emma Davis — Checking
    ("FD-D005-01", "DEMO005", 150_000.00, 12, 10.50, "2026-01-01", "active"),
    # DEMO006 — Frank Miller — Savings
    ("FD-D006-01", "DEMO006", 400_000.00, 24, 11.00, "2025-06-01", "active"),
    ("FD-D006-02", "DEMO006", 200_000.00, 12, 10.50, "2025-03-01", "matured"),
    # DEMO008 — Henry Moore — Premium (DEMO007 has no FD)
    ("FD-D008-01", "DEMO008", 3_000_000.00, 60, 11.25, "2022-08-01", "active"),
    ("FD-D008-02", "DEMO008",   800_000.00, 36, 11.50, "2024-03-01", "active"),
    # DEMO009 — Iris Taylor — Checking
    ("FD-D009-01", "DEMO009", 200_000.00, 12, 10.50, "2026-03-01", "active"),
    # DEMO010 — Jack Anderson — Savings
    ("FD-D010-01", "DEMO010", 600_000.00, 24, 11.00, "2025-05-01", "active"),
    ("FD-D010-02", "DEMO010", 250_000.00, 12, 10.50, "2025-08-01", "active"),
    # DEMO011 — Kate Thomas — Checking
    ("FD-D011-01", "DEMO011", 180_000.00, 12, 10.50, "2026-02-01", "active"),
    # DEMO013 — Mia Harris — Premium (DEMO012 has no FD)
    ("FD-D013-01", "DEMO013", 1_500_000.00, 36, 11.50, "2024-06-01", "active"),
    ("FD-D013-02", "DEMO013",   500_000.00, 24, 11.00, "2025-08-01", "active"),
    ("FD-D013-03", "DEMO013",   300_000.00, 12, 10.50, "2025-12-01", "active"),
    # DEMO014 — Noah Martin — Savings
    ("FD-D014-01", "DEMO014", 350_000.00, 12, 10.50, "2026-01-01", "active"),
    # DEMO015 — Olivia Garcia — Checking
    ("FD-D015-01", "DEMO015",  80_000.00,  6,  9.75, "2026-04-01", "active"),
]

# Loans: (loan_id, customer_id, loan_type, principal, annual_rate, tenure_months, start_date)
# outstanding_balance and monthly_installment are computed
_LOAN_DATA = [
    # DEMO001 — Alice
    ("LN-D001-01", "DEMO001", "personal",  200_000.00, 18.00, 36, "2025-01-15"),
    # DEMO004 — David — vehicle loan
    ("LN-D004-01", "DEMO004", "vehicle",  3_500_000.00, 14.00, 84, "2024-03-01"),
    # DEMO008 — Henry — housing loan
    ("LN-D008-01", "DEMO008", "housing", 15_000_000.00, 12.50, 240, "2022-06-01"),
    # DEMO009 — Iris — education loan
    ("LN-D009-01", "DEMO009", "education", 500_000.00, 13.50, 60, "2024-09-01"),
    # DEMO011 — Kate — personal loan
    ("LN-D011-01", "DEMO011", "personal",  150_000.00, 18.00, 24, "2025-06-01"),
    # DEMO013 — Mia — business loan
    ("LN-D013-01", "DEMO013", "business", 2_000_000.00, 16.00, 120, "2023-10-01"),
]

# Pawning: (pawn_id, customer_id, item_description, weight_grams, carat, monthly_interest_rate, pawn_date, tenure_months, status)
# advance_amount and due_date are computed
_PAWN_DATA = [
    # DEMO002 — Bob
    ("PW-D002-01", "DEMO002", "22ct gold chain necklace", 18.0, 22, 2.00, "2026-03-01", 6, "active"),
    # DEMO005 — Emma
    ("PW-D005-01", "DEMO005", "18ct gold earrings set",    8.5, 18, 2.00, "2026-04-01", 4, "active"),
    # DEMO007 — Grace (no FD, no loan — only pawning)
    ("PW-D007-01", "DEMO007", "22ct gold bangle",         12.0, 22, 2.00, "2026-01-15", 6, "active"),
    # DEMO012 — Liam (no FD, no loan — only pawning)
    ("PW-D012-01", "DEMO012", "24ct gold coin (5g)",       5.0, 24, 1.75, "2025-11-01", 6, "redeemed"),
    # DEMO015 — Olivia
    ("PW-D015-01", "DEMO015", "22ct gold ring",            4.5, 22, 2.00, "2026-02-01", 4, "active"),
]

# Cards: (card_id, customer_id, card_type, last_4, credit_limit_or_None, available_limit_or_None)
_CARD_DATA = [
    # DEMO001 — Alice
    ("CD-D001-01", "DEMO001", "debit",  "4501", None,           None),
    # DEMO002 — Bob
    ("CD-D002-01", "DEMO002", "debit",  "7823", None,           None),
    ("CD-D002-02", "DEMO002", "credit", "9102", 200_000.00, 142_000.00),
    # DEMO003 — Carol
    ("CD-D003-01", "DEMO003", "debit",  "3344", None,           None),
    # DEMO004 — David — Premium, 2 cards
    ("CD-D004-01", "DEMO004", "debit",  "6612", None,           None),
    ("CD-D004-02", "DEMO004", "credit", "8831", 500_000.00, 380_000.00),
    # DEMO005 — Emma
    ("CD-D005-01", "DEMO005", "debit",  "2277", None,           None),
    # DEMO006 — Frank
    ("CD-D006-01", "DEMO006", "debit",  "5590", None,           None),
    ("CD-D006-02", "DEMO006", "credit", "1123", 150_000.00,  98_000.00),
    # DEMO007 — Grace
    ("CD-D007-01", "DEMO007", "debit",  "0041", None,           None),
    # DEMO008 — Henry — Premium, 2 cards
    ("CD-D008-01", "DEMO008", "debit",  "7714", None,           None),
    ("CD-D008-02", "DEMO008", "credit", "3389", 750_000.00, 612_000.00),
    # DEMO009 — Iris
    ("CD-D009-01", "DEMO009", "debit",  "8862", None,           None),
    # DEMO010 — Jack
    ("CD-D010-01", "DEMO010", "debit",  "4499", None,           None),
    ("CD-D010-02", "DEMO010", "credit", "7756", 250_000.00, 189_000.00),
    # DEMO011 — Kate
    ("CD-D011-01", "DEMO011", "debit",  "3312", None,           None),
    # DEMO012 — Liam
    ("CD-D012-01", "DEMO012", "debit",  "0083", None,           None),
    # DEMO013 — Mia — Premium, 2 cards
    ("CD-D013-01", "DEMO013", "debit",  "9941", None,           None),
    ("CD-D013-02", "DEMO013", "credit", "6607", 400_000.00, 355_000.00),
    # DEMO014 — Noah
    ("CD-D014-01", "DEMO014", "debit",  "1155", None,           None),
    ("CD-D014-02", "DEMO014", "credit", "4428", 120_000.00,  88_000.00),
    # DEMO015 — Olivia
    ("CD-D015-01", "DEMO015", "debit",  "7799", None,           None),
]


# ── Seeding function ──────────────────────────────────────────────────────────

def seed(force: bool = False):
    init_services_db()

    with _conn() as c:
        existing = c.execute("SELECT COUNT(*) FROM fd_rates").fetchone()[0]

    if existing > 0 and not force:
        print(f"services.db already has data. Use --force to re-seed.")
        return

    if force:
        with _conn() as c:
            for tbl in ["cards", "pawning_records", "loans", "fixed_deposits",
                        "fx_rates", "transfer_fees", "pawning_rates", "loan_rates", "fd_rates"]:
                c.execute(f"DELETE FROM {tbl}")
            c.commit()
        print("Existing data cleared.")

    print("\nSeeding rate cards...")
    with _conn() as c:
        c.executemany("INSERT INTO fd_rates VALUES (?,?,?)", FD_RATES)

        c.executemany("INSERT INTO loan_rates VALUES (?,?,?)", LOAN_RATES)

        for carat, rpg, ltv, mir in PAWNING_RATES:
            c.execute("INSERT INTO pawning_rates VALUES (?,?,?,?)", (carat, rpg, ltv, mir))

        for tf in TRANSFER_FEES:
            ttype, mn, mx, ftype, fval, minfee = tf
            c.execute(
                "INSERT INTO transfer_fees (transfer_type,min_amount,max_amount,fee_type,fee_value,min_fee) "
                "VALUES (?,?,?,?,?,?)",
                (ttype, mn, mx, ftype, fval, minfee),
            )

        c.executemany("INSERT INTO fx_rates VALUES (?,?,?)", FX_RATES)
        c.commit()

    print("Seeding fixed deposits...")
    rows_fd = 0
    with _conn() as c:
        for fd_id, cid, principal, tenure, rate, start, status in _FD_DATA:
            mat_date   = _date(start, tenure)
            mat_amount = round(_fd_maturity(principal, rate, tenure), 2)
            c.execute(
                "INSERT INTO fixed_deposits VALUES (?,?,?,?,?,?,?,?,?)",
                (fd_id, cid, principal, tenure, rate, start, mat_date, mat_amount, status),
            )
            rows_fd += 1
        c.commit()

    print("Seeding loans...")
    rows_ln = 0
    with _conn() as c:
        for loan_id, cid, ltype, principal, rate, tenure, start in _LOAN_DATA:
            emi_val  = round(_emi(principal, rate, tenure), 2)
            paid     = _months_between(start)
            paid     = min(paid, tenure)
            outstand = round(_outstanding(principal, rate, tenure, paid), 2)
            c.execute(
                "INSERT INTO loans VALUES (?,?,?,?,?,?,?,?)",
                (loan_id, cid, ltype, principal, outstand, emi_val, tenure, start),
            )
            rows_ln += 1
        c.commit()

    print("Seeding pawning records...")
    rows_pw = 0
    with _conn() as c:
        for pawn_id, cid, desc, weight, carat, mir, pawn_date, tenure, status in _PAWN_DATA:
            # Look up advance rate per gram for this carat
            rate_row = c.execute(
                "SELECT rate_per_gram FROM pawning_rates WHERE carat = ?", (carat,)
            ).fetchone()
            rate_pg  = rate_row[0] if rate_row else 0.0
            advance  = round(weight * rate_pg, 2)
            due_date = _date(pawn_date, tenure)
            c.execute(
                "INSERT INTO pawning_records VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pawn_id, cid, desc, weight, carat, advance, mir, pawn_date, due_date, status),
            )
            rows_pw += 1
        c.commit()

    print("Seeding cards...")
    rows_cd = 0
    with _conn() as c:
        for card_id, cid, ctype, last4, climit, alimit in _CARD_DATA:
            masked = f"**** **** **** {last4}"
            c.execute(
                "INSERT INTO cards VALUES (?,?,?,?,?,?,?)",
                (card_id, cid, ctype, masked, climit, alimit, "active"),
            )
            rows_cd += 1
        c.commit()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SERVICES DATABASE SEEDED  (simulated data only)")
    print("=" * 65)
    print(f"  FD Rates         : {len(FD_RATES)}")
    print(f"  Loan Rates       : {len(LOAN_RATES)}")
    print(f"  Pawning Rates    : {len(PAWNING_RATES)}")
    print(f"  Transfer Fee rows: {len(TRANSFER_FEES)}")
    print(f"  FX Rates         : {len(FX_RATES)}")
    print("-" * 65)
    print(f"  Fixed Deposits   : {rows_fd} records")
    print(f"  Loans            : {rows_ln} records")
    print(f"  Pawning Records  : {rows_pw} records")
    print(f"  Cards            : {rows_cd} records")
    print("=" * 65)
    print(f"  Database: {_DB_PATH}")
    print()


if __name__ == "__main__":
    seed("--force" in sys.argv)
