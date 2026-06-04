"""In-memory session store.

Replaces ``st.session_state`` for the FastAPI backend. One record per browser
session (keyed by a UUID cookie). The record shape mirrors ``problem_state``
exactly so ``agent/tools/state_bridge`` works unchanged.

The store is deliberately a narrow interface so a real backing store (Redis,
Postgres) can be dropped in later without touching the API or the React
client. For v1 everything lives in-process.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


SESSION_TTL_SECONDS = 60 * 60 * 12  # 12 h of inactivity → collected


def _make_network_manager():
    """Return a NetworkManager, or None if its module fails to import.

    Isolated so `_fresh_record()` can't blow up test collection if osmnx /
    shapely are unavailable in the import graph.
    """
    try:
        from utils.network_manager import NetworkManager
        return NetworkManager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SessionStore: NetworkManager unavailable (%s)", exc)
        return None


def _fresh_record() -> Dict[str, Any]:
    """Return a freshly initialised session record.

    Keys mirror ``app.initialize_session_state()`` — if Streamlit relies on
    a key, the React backend must expose it under the same name so tools
    using ``state_bridge`` continue to work unchanged.
    """
    return {
        # Problem state — identical shape to st.session_state.problem_state.
        "problem_state": {
            "problem_type": None,
            "parameters": {},
            "constraints": {},
            "data": {},                 # dataset_name -> GeoDataFrame
            "solution": None,
            "solution_history": [],
            "aoi": None,
            "aoi_confirmed": False,
        },
        # Chat history (list of {role, content, tool_calls?}).
        "messages": [],
        # Raster uploads (kept separate from vector data, as in app.py).
        "raster_data": {},
        # Activity log events (list[ActivityEvent dict]) — kept here instead
        # of in the activity_log ring buffer so each session owns its own.
        "_activity_log": [],
        # Road-network prefetch status.
        "_network_status": None,         # None | "fetching" | "ready" | "failed"
        "_network_status_error": None,
        "_network_status_stats": None,
        # User settings.
        "generated_sites_count": 100,
        "generated_sites_seed": None,
        # Eagerly-initialised singletons scoped to this session. NetworkManager
        # must be the same instance across the prefetch thread (launched from
        # aoi.py) and the solve-time fetch (chat.py → optimize_tools). Lazy
        # creation at first use produced two instances and a double-fetch
        # race against Overpass. Instantiation is cheap.
        "_network_manager": _make_network_manager(),
        "_data_fetcher": None,
        "_data_processor": None,
        "_map_visualizer": None,
        "_pydeck_visualizer": None,
        "_export_handler": None,
        # Per-session ADK agent runner. Created on first chat turn.
        "_soca_agent": None,
        # Bookkeeping.
        "created_at": time.time(),
        "last_access": time.time(),
    }


class SessionStore:
    """Thread-safe, in-memory map of ``session_id`` → session record."""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self) -> tuple[str, Dict[str, Any]]:
        """Create a new session and return ``(session_id, record)``."""
        session_id = secrets.token_urlsafe(24)
        with self._lock:
            record = _fresh_record()
            self._records[session_id] = record
        logger.info("SessionStore: created session %s…", session_id[:8])
        return session_id, record

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the session record or ``None`` if it does not exist / expired."""
        if not session_id:
            return None
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            if time.time() - record["last_access"] > SESSION_TTL_SECONDS:
                self._records.pop(session_id, None)
                logger.info("SessionStore: session %s… expired", session_id[:8])
                return None
            record["last_access"] = time.time()
            return record

    def get_or_create(
        self, session_id: Optional[str]
    ) -> tuple[str, Dict[str, Any]]:
        """Return an existing record or a new one. The returned id is the
        canonical one (either the input id, or a newly-minted one)."""
        if session_id:
            record = self.get(session_id)
            if record is not None:
                return session_id, record
        return self.create()

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._records.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._records.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------------------
    # GC (called opportunistically from a background task)
    # ------------------------------------------------------------------

    def sweep(self) -> int:
        """Drop records that have been idle beyond ``SESSION_TTL_SECONDS``."""
        now = time.time()
        dropped = 0
        with self._lock:
            for sid in list(self._records.keys()):
                rec = self._records[sid]
                if now - rec["last_access"] > SESSION_TTL_SECONDS:
                    del self._records[sid]
                    dropped += 1
        if dropped:
            logger.info("SessionStore: swept %d expired sessions", dropped)
        return dropped


# Module-level singleton — one process, one store. Tests can construct
# additional instances directly.
_default_store: Optional[SessionStore] = None


def get_default_store() -> SessionStore:
    global _default_store
    if _default_store is None:
        _default_store = SessionStore()
    return _default_store
