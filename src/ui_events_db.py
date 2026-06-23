"""
UI Events database — opt-in customer monitoring feature.
Separate file (ui_events.db) from chat_history.db so tracking data can be
wiped or inspected independently without touching conversation history.

PRIVACY INVARIANT: the interaction_events table has NO value column, by design.
Field names are stored; field values are never captured, stored, or transmitted.
"""

import os
import sqlite3

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ui_events.db"
)


def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_ui_events_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS interaction_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                page       TEXT    NOT NULL,
                event_type TEXT    NOT NULL,
                field_name TEXT,
                timestamp  TEXT    DEFAULT (datetime('now','localtime'))
            )
        """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_ie_sess_page "
            "ON interaction_events(session_id, page)"
        )
        c.commit()


def save_event(session_id: str, page: str, event_type: str,
               field_name: str | None) -> None:
    """Store one interaction event. field_name may be None for submit_success."""
    with _conn() as c:
        c.execute(
            "INSERT INTO interaction_events (session_id, page, event_type, field_name) "
            "VALUES (?, ?, ?, ?)",
            (session_id, page, event_type, field_name),
        )
        c.commit()


def get_recent_events(session_id: str, page: str) -> list[dict]:
    """Return all events for this session+page, oldest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT event_type, field_name FROM interaction_events "
            "WHERE session_id = ? AND page = ? ORDER BY id",
            (session_id, page),
        ).fetchall()
    return [dict(r) for r in rows]
