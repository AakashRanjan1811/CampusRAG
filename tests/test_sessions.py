import sqlite3

import pytest

from database.session_store import SessionNotFound, SessionStore


def test_creation_isolation_and_persistence(settings):
    store = SessionStore(settings.sqlite_path)
    a, b = store.create_session(), store.create_session()
    assert a != b
    citations = [{"filename": "Policy.pdf", "page": 2, "snippet": "75% attendance", "chunk_id": "c1", "document_id": "d1"}]
    store.save_turn(a, "Attendance?", "75% [1]", citations)
    restarted = SessionStore(settings.sqlite_path)
    assert len(restarted.list_sessions()) == 2
    assert restarted.get_messages(b) == []
    assert [m["role"] for m in restarted.get_messages(a)] == ["user", "assistant"]
    assert restarted.get_messages(a)[1]["citations"] == citations
    assert restarted.get_messages(a, limit=1)[0]["role"] == "assistant"
    assert restarted.get_messages(a, limit=0) == []
    assert next(s for s in restarted.list_sessions() if s["id"] == a)["title"] == "Attendance?"


def test_deletion_cascades_and_missing_session(settings):
    store = SessionStore(settings.sqlite_path)
    session = store.create_session()
    store.save_turn(session, "Question", "Answer", [])
    store.delete_session(session)
    assert not store.list_sessions()
    with sqlite3.connect(settings.sqlite_path) as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    for operation in [lambda: store.get_messages(session), lambda: store.save_turn(session, "Q", "A", []), lambda: store.delete_session(session)]:
        with pytest.raises(SessionNotFound):
            operation()


def test_sql_parameters_and_bounded_recent_window(settings):
    store = SessionStore(settings.sqlite_path)
    session = store.create_session("' ; DROP TABLE sessions; --")
    for i in range(10):
        store.save_turn(session, f"Q{i}", f"A{i}", [])
    assert len(store.list_sessions()) == 1
    assert [m["content"] for m in store.get_messages(session, limit=4)] == ["Q8", "A8", "Q9", "A9"]
