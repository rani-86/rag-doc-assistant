
import json
from datetime import datetime, timezone
from threading import Lock

from app.config import FEEDBACK_FILE

_lock = Lock()


def save_feedback(record: dict) -> dict:
    """Append one feedback record to a JSONL file."""

    entry = {
        **record,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    with _lock, FEEDBACK_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return entry
