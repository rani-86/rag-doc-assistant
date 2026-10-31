
from collections import deque
from threading import Lock

from app.config import MAX_HISTORY_TURNS

# In-memory session store: session_id -> last N turns.
# Lost on restart; a real deployment would use Redis or a database.
_sessions: dict[str, deque] = {}
_lock = Lock()


def get_history(session_id: str | None) -> list[dict]:
    if not session_id:
        return []

    with _lock:
        return list(_sessions.get(session_id, []))


def add_turn(session_id: str | None, question: str, answer: str) -> None:
    if not session_id:
        return

    with _lock:
        history = _sessions.setdefault(
            session_id, deque(maxlen=MAX_HISTORY_TURNS)
        )
        history.append({"question": question, "answer": answer})


def clear_history(session_id: str) -> bool:
    with _lock:
        return _sessions.pop(session_id, None) is not None
