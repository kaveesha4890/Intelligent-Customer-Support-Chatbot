import sqlite3
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_history.db")


def _connect():
    return sqlite3.connect(_DB_PATH)


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                user_msg    TEXT    NOT NULL,
                bot_msg     TEXT    NOT NULL,
                intent      TEXT,
                category    TEXT,
                sentiment   TEXT,
                escalated   INTEGER DEFAULT 0,
                feedback    INTEGER DEFAULT NULL,  -- 1=thumbs up, 0=thumbs down
                timestamp   TEXT    DEFAULT (datetime('now','localtime'))
            )
        """)
        # Add columns to existing DB if upgrading
        for col, definition in [("category", "TEXT"), ("feedback", "INTEGER DEFAULT NULL")]:
            try:
                conn.execute(f"ALTER TABLE conversations ADD COLUMN {col} {definition}")
            except Exception:
                pass
        conn.commit()


def save_turn(session_id: str, user_msg: str, bot_msg: str,
              intent: str, sentiment: str, escalated: bool, category: str = None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO conversations
               (session_id, user_msg, bot_msg, intent, category, sentiment, escalated)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, user_msg, bot_msg, intent, category, sentiment, int(escalated)),
        )
        conn.commit()
        return cur.lastrowid


def save_feedback(turn_id: int, rating: int):
    """rating: 1 = thumbs up, 0 = thumbs down"""
    with _connect() as conn:
        conn.execute("UPDATE conversations SET feedback = ? WHERE id = ?", (rating, turn_id))
        conn.commit()


def get_all_stats() -> dict:
    with _connect() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        escalated = conn.execute("SELECT COUNT(*) FROM conversations WHERE escalated = 1").fetchone()[0]
        thumbs_up   = conn.execute("SELECT COUNT(*) FROM conversations WHERE feedback = 1").fetchone()[0]
        thumbs_down = conn.execute("SELECT COUNT(*) FROM conversations WHERE feedback = 0").fetchone()[0]
        top_intents = conn.execute("""
            SELECT intent, COUNT(*) as cnt
            FROM conversations
            GROUP BY intent
            ORDER BY cnt DESC
            LIMIT 10
        """).fetchall()
        bad_responses = conn.execute("""
            SELECT user_msg, bot_msg, intent, timestamp
            FROM conversations
            WHERE feedback = 0
            ORDER BY timestamp DESC
            LIMIT 10
        """).fetchall()
    return {
        "total_messages": total,
        "escalated":      escalated,
        "thumbs_up":      thumbs_up,
        "thumbs_down":    thumbs_down,
        "top_intents":    [{"intent": r[0], "count": r[1]} for r in top_intents],
        "bad_responses":  [{"user_msg": r[0], "bot_msg": r[1], "intent": r[2], "timestamp": r[3]}
                           for r in bad_responses],
    }
