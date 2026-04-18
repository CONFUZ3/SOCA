"""
Activity log — structured event bus for user-visible fetch transparency.

Modules like DataFetcher and geocoder emit events via log_event(); the AOI
selector and main app render them with render_log() so users can see exactly
which open-data source served their boundary / population / POI request.

Design notes:
  - Events live in st.session_state["_activity_log"] as a ring buffer (cap 50).
  - Stateless fallback: if there is no Streamlit script context (e.g. pytest,
    background worker), events fall through to module-level _FALLBACK so tests
    can still assert on them.
  - Each event carries enough info to reconstruct the decision path:
        stage   : "geocode.suggest" | "boundary.fetch" | "population.fetch" | …
        status  : "try" | "ok" | "fail" | "info"
        source  : "Photon" | "Overpass" | "Overture" | "GADM" | …
        detail  : human-readable one-liner
        duration_ms, timestamp, extra
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_MAX_EVENTS = 50
_STATE_KEY = "_activity_log"
_FALLBACK: list["ActivityEvent"] = []

# status → (glyph, rank) — rank used for "auto-expand on first error"
_STATUS_GLYPHS = {
    "try": ("…", 0),
    "info": ("•", 0),
    "ok": ("✓", 0),
    "fail": ("✗", 1),
}


@dataclass
class ActivityEvent:
    stage: str
    status: str
    detail: str = ""
    source: Optional[str] = None
    duration_ms: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        glyph, _ = _STATUS_GLYPHS.get(self.status, ("•", 0))
        src = f" {self.source}" if self.source else ""
        dur = f"  ({self.duration_ms:.0f} ms)" if self.duration_ms is not None else ""
        return f"{glyph} {self.stage:<20}{src:<14} {self.detail}{dur}"


def _bucket() -> list[ActivityEvent]:
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return _FALLBACK
        return st.session_state.setdefault(_STATE_KEY, [])
    except Exception:
        return _FALLBACK


def log_event(
    stage: str,
    status: str,
    detail: str = "",
    *,
    source: Optional[str] = None,
    duration_ms: Optional[float] = None,
    **extra: Any,
) -> ActivityEvent:
    """Append an event to the activity log and mirror to the Python logger."""
    evt = ActivityEvent(
        stage=stage,
        status=status,
        detail=detail,
        source=source,
        duration_ms=duration_ms,
        extra=extra or {},
    )
    buf = _bucket()
    buf.append(evt)
    if len(buf) > _MAX_EVENTS:
        del buf[: len(buf) - _MAX_EVENTS]

    level = logging.WARNING if status == "fail" else logging.INFO
    logger.log(level, "%s", evt.format())
    return evt


class timed:
    """Context manager: emit a 'try' event on enter, 'ok'/'fail' on exit.

    Usage:
        with timed("boundary.fetch", source="Overpass", detail="R175905") as t:
            ...
            t.detail = f"polygon, {n} vertices"   # updates on success
    """

    def __init__(self, stage: str, *, source: Optional[str] = None, detail: str = ""):
        self.stage = stage
        self.source = source
        self.detail = detail
        self._start: float = 0.0

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        log_event(self.stage, "try", self.detail, source=self.source)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dur_ms = (time.perf_counter() - self._start) * 1000.0
        if exc is None:
            log_event(self.stage, "ok", self.detail, source=self.source, duration_ms=dur_ms)
        else:
            msg = f"{type(exc).__name__}: {exc}" if not self.detail else f"{self.detail}  ·  {type(exc).__name__}: {exc}"
            log_event(self.stage, "fail", msg, source=self.source, duration_ms=dur_ms)
        return False  # never suppress


def get_events() -> list[ActivityEvent]:
    return list(_bucket())


def clear_events() -> None:
    _bucket().clear()


def has_errors(events: Optional[Iterable[ActivityEvent]] = None) -> bool:
    events = events if events is not None else _bucket()
    return any(e.status == "fail" for e in events)


def render_log(*, expanded: Optional[bool] = None, max_rows: int = 30) -> None:
    """Render the activity log in Streamlit. Auto-expands on first error."""
    import streamlit as st

    events = _bucket()
    if not events:
        return

    errored = any(e.status == "fail" for e in events)
    if expanded is None:
        expanded = errored

    label = f"Activity log · {len(events)} events" + ("  ⚠ contains errors" if errored else "")
    with st.expander(label, expanded=expanded):
        rows = events[-max_rows:]
        st.code("\n".join(e.format() for e in rows), language="text")
