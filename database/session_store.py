"""Short-lived connections make SQLite safe across Streamlit reruns/threads."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class SessionNotFound(ValueError):
    pass


class SessionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL, citations TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _require(conn, session_id):
        if not conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
            raise SessionNotFound("This chat session no longer exists. Create or select another chat.")

    def create_session(self, title="New conversation") -> str:
        session_id, now = str(uuid4()), self._now()
        with self._connect() as conn:
            conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (session_id, title[:80], now, now))
        return session_id

    def list_sessions(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC, id")]

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        if limit is not None and limit < 0:
            raise ValueError("History limit cannot be negative.")
        with self._connect() as conn:
            self._require(conn, session_id)
            if limit is None:
                rows = conn.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()[::-1]
        return [dict(row, citations=json.loads(row["citations"])) for row in rows]

    def save_turn(self, session_id: str, question: str, answer: str, citations: list[dict]):
        """Commit the user/assistant pair together; failed inference leaves no half turn."""
        now = self._now()
        with self._connect() as conn:
            self._require(conn, session_id)
            first_turn = not conn.execute("SELECT 1 FROM messages WHERE session_id=? LIMIT 1", (session_id,)).fetchone()
            conn.executemany(
                "INSERT INTO messages(session_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?)",
                [(session_id, "user", question, "[]", now),
                 (session_id, "assistant", answer, json.dumps(citations, ensure_ascii=False), now)],
            )
            if first_turn:
                conn.execute("UPDATE sessions SET title=? WHERE id=?", (" ".join(question.split())[:60], session_id))
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            self._require(conn, session_id)
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
