"""FastAPI dependencies shared across routers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from fastapi import Cookie, HTTPException, Request, Response, status

from backend.services.event_bus import EventBus, get_default_bus
from backend.services.session_store import SessionStore, get_default_store

logger = logging.getLogger(__name__)

SESSION_COOKIE = "soca_session"
COOKIE_MAX_AGE = 60 * 60 * 12  # 12h, matches SESSION_TTL_SECONDS


def _cookie_secure() -> bool:
    """Return True in production (HTTPS). Disabled in dev for localhost."""
    return os.environ.get("SOCA_COOKIE_SECURE", "0") == "1"


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )


def get_store() -> SessionStore:
    return get_default_store()


def get_bus() -> EventBus:
    return get_default_bus()


def resolve_session(
    request: Request,
    response: Response,
    soca_session: Optional[str] = Cookie(default=None),
) -> Tuple[str, Dict[str, Any]]:
    """Ensure the request has an active session. Create one if missing.

    Attaches the cookie to the response so the browser persists it, and
    returns ``(session_id, record)``.
    """
    store = get_store()
    session_id, record = store.get_or_create(soca_session)
    # Always refresh the cookie so the client holds an up-to-date TTL.
    set_session_cookie(response, session_id)
    return session_id, record


def require_session(
    request: Request,
    soca_session: Optional[str] = Cookie(default=None),
) -> Tuple[str, Dict[str, Any]]:
    """Fail if the client has no active session. Useful for routes that
    must not silently create one (e.g. event stream)."""
    if not soca_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No session cookie. Call POST /api/session first.",
        )
    store = get_store()
    record = store.get(soca_session)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Call POST /api/session to start a new one.",
        )
    return soca_session, record
