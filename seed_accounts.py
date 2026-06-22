#!/usr/bin/env python3
"""
DEMO DATA SEEDER -- FOR DEMONSTRATION / ACADEMIC PURPOSES ONLY
==============================================================
Creates SIMULATED / DUMMY bank account and transaction data.
ALL DATA IS COMPLETELY FICTIONAL.

Run:          python seed_accounts.py
Reset+reseed: python seed_accounts.py --force
"""

import sys
import os
import sqlite3
import bcrypt
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.accounts_db import init_accounts_db, _DB_PATH, _conn

DEMO_CUSTOMERS = [
    ("DEMO001", "Alice Johnson",   "DEMO10012345", 5_432.50,  "Checking", "1234"),
    ("DEMO002", "Bob Smith",       "DEMO20023456", 12_750.00, "Savings",  "2345"),
    ("DEMO003", "Carol White",     "DEMO30034567", 892.75,    "Checking", "3456"),
    ("DEMO004", "David Brown",     "DEMO40045678", 28_000.00, "Premium",  "4567"),
    ("DEMO005", "Emma Davis",      "DEMO50056789", 1_234.56,  "Checking", "5678"),
    ("DEMO006", "Frank Miller",    "DEMO60067890", 6_500.00,  "Savings",  "6789"),
    ("DEMO007", "Grace Wilson",    "DEMO70078901", 350.25,    "Checking", "7890"),
    ("DEMO008", "Henry Moore",     "DEMO80089012", 45_000.00, "Premium",  "8901"),
    ("DEMO009", "Iris Taylor",     "DEMO90090123", 2_100.00,  "Checking", "9012"),
    ("DEMO010", "Jack Anderson",   "DEMO10001234", 8_900.75,  "Savings",  "0123"),
    ("DEMO011", "Kate Thomas",     "DEMO11011234", 3_300.00,  "Checking", "1111"),
    ("DEMO012", "Liam Jackson",    "DEMO12021234", 780.50,    "Checking", "2222"),
    ("DEMO013", "Mia Harris",      "DEMO13031234", 15_600.00, "Premium",  "3333"),
    ("DEMO014", "Noah Martin",     "DEMO14041234", 4_200.25,  "Savings",  "4444"),
    ("DEMO015", "Olivia Garcia",   "DEMO15051234", 990.00,    "Checking", "5555"),
]

CREDIT_TEMPLATES = [
    ("Salary Deposit",         lambda: round(random.uniform(2000, 5000), 2)),
    ("Bank Transfer Received", lambda: round(random.uniform(50, 800), 2)),
    ("Refund - Online Store",  lambda: round(random.uniform(10, 200), 2)),
    ("Interest Payment",       lambda: round(random.uniform(1, 50), 2)),
    ("Cashback Reward",        lambda: round(random.uniform(2, 30), 2)),
]
DEBIT_TEMPLATES = [
    ("Grocery Store",          lambda: round(random.uniform(20, 150), 2)),
    ("Electric Bill",          lambda: round(random.uniform(60, 180), 2)),
    ("Internet Subscription",  lambda: round(random.uniform(15, 50), 2)),
    ("ATM Withdrawal",         lambda: round(random.choices([50, 100, 200], k=1)[0], 2)),
    ("Online Shopping",        lambda: round(random.uniform(25, 300), 2)),
    ("Restaurant",             lambda: round(random.uniform(15, 80), 2)),
    ("Fuel Station",           lambda: round(random.uniform(30, 90), 2)),
    ("Mobile Recharge",        lambda: round(random.uniform(10, 40), 2)),
    ("Streaming Service",      lambda: round(random.uniform(8, 20), 2)),
    ("Gym Membership",         lambda: round(random.uniform(20, 60), 2)),
]


def _make_transactions(customer_id, n=5):
    txns = []
    today = datetime.now()
    for i in range(n):
        date = (today - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
        if random.random() < 0.3:
            desc, amt_fn = random.choice(CREDIT_TEMPLATES)
            txn_type = "credit"
        else:
            desc, amt_fn = random.choice(DEBIT_TEMPLATES)
            txn_type = "debit"
        txns.append({
            "txn_id":      f"{customer_id}-TXN-{i+1:03d}",
            "customer_id": customer_id,
            "date":        date,
            "description": desc,
            "amount":      amt_fn(),
            "txn_type":    txn_type,
        })
    txns.sort(key=lambda t: t["date"], reverse=True)
    return txns


def seed(force=False):
    init_accounts_db()
    with _conn() as c:
        existing = c.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]

    if existing > 0 and not force:
        print(f"accounts.db already has {existing} accounts. Use --force to re-seed.")
        return

    if force:
        with _conn() as c:
            c.execute("DELETE FROM transactions")
            c.execute("DELETE FROM accounts")
            c.execute("DELETE FROM login_attempts")
            c.commit()
        print("Existing data cleared.")

    print("\nSeeding demo accounts...\n")
    rows_a = rows_t = 0

    for cid, name, account_no, balance, acc_type, demo_pin in DEMO_CUSTOMERS:
        pin_hash = bcrypt.hashpw(demo_pin.encode(), bcrypt.gensalt()).decode()
        txns     = _make_transactions(cid)
        last_txn = txns[0]["date"] if txns else None
        with _conn() as c:
            c.execute(
                "INSERT INTO accounts "
                "(customer_id,name,account_no,balance,account_type,last_txn_date,pin_hash) "
                "VALUES (?,?,?,?,?,?,?)",
                (cid, name, account_no, balance, acc_type, last_txn, pin_hash),
            )
            for t in txns:
                c.execute(
                    "INSERT INTO transactions "
                    "(txn_id,customer_id,date,description,amount,txn_type) "
                    "VALUES (?,?,?,?,?,?)",
                    (t["txn_id"],t["customer_id"],t["date"],t["description"],t["amount"],t["txn_type"]),
                )
            c.commit()
        rows_a += 1
        rows_t += len(txns)

    print("=" * 60)
    print("  DEMO CREDENTIALS  (simulated data only)")
    print("=" * 60)
    print(f"  {'Customer ID':<12} {'Name':<20} {'Type':<10} PIN")
    print("-" * 60)
    for cid, name, _, _, acc_type, pin in DEMO_CUSTOMERS:
        print(f"  {cid:<12} {name:<20} {acc_type:<10} {pin}")
    print("=" * 60)
    print(f"\n  Accounts : {rows_a}   Transactions : {rows_t}")
    print(f"  Database : {_DB_PATH}\n")


if __name__ == "__main__":
    seed("--force" in sys.argv)
