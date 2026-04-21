"""SOCA FastAPI entrypoint — ``uvicorn backend.main:app``."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import aoi, chat, data, events, network, problems, session
from backend.services.event_bus import get_default_bus
from backend.services.session_store import get_default_store

load_dotenv()

# Force UTF-8 stderr so activity-log glyphs (✓ … • ✗) don't break Windows.
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=os.environ.get("SOCA_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("soca.backend")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm singletons early so the first request isn't slower than the next.
    get_default_store()
    get_default_bus()
    logger.info("SOCA backend started")
    yield
    logger.info("SOCA backend stopped")


app = FastAPI(
    title="SOCA Backend",
    version="0.1.0",
    description=(
        "REST + SSE API for the Spatial Optimization Conversational Agent. "
        "Thin wrapper over the existing Python agent/solvers/utils modules."
    ),
    lifespan=lifespan,
)


_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "SOCA_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "soca-backend"})


app.include_router(session.router)
app.include_router(problems.router)
app.include_router(aoi.router)
app.include_router(network.router)
app.include_router(events.router)
app.include_router(chat.router)
app.include_router(data.router)
