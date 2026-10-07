"""Privacy-first logging. Submitted text is never logged, printed or stored.

`log_event` accepts ONLY whitelisted metadata keys; anything else (e.g. `text`) is dropped and counted, so a future
careless call site cannot leak user text into logs.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid

SAFE_KEYS = frozenset({"request_id", "timestamp", "method", "route", "status", "latency_ms", "model", "model_version",
                       "n_items", "language_tag", "risk_label", "event", "error_type"})

logger = logging.getLogger("mhrisk")
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def log_event(**fields) -> dict:
    """Emit one JSON log line containing only whitelisted metadata. Returns what was logged."""
    safe = {k: v for k, v in fields.items() if k in SAFE_KEYS}
    safe.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z")
    dropped = sorted(set(fields) - SAFE_KEYS)
    if dropped:
        safe["dropped_fields"] = len(dropped)  # count only; names/values are never emitted
    logger.info(json.dumps(safe, ensure_ascii=False))
    return safe
