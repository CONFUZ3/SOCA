"""Smoke tests for backend REST endpoints.

Uses FastAPI's TestClient. The ADK-powered chat stream is covered only
indirectly here (unit-testing that it surfaces a clear error without a key);
integration tests for streaming live behind `pytest -m live`.
"""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "service": "soca-backend"}


def test_create_session_sets_cookie_and_returns_snapshot(client):
    resp = client.post("/api/session")
    assert resp.status_code == 200
    cookies = resp.cookies
    assert "soca_session" in cookies
    body = resp.json()
    for key in (
        "aoi",
        "aoi_confirmed",
        "problem_type",
        "datasets",
        "messages",
        "network",
        "settings",
    ):
        assert key in body


def test_dataset_summary_reports_total_population():
    import geopandas as gpd
    from shapely.geometry import Point

    from backend.api.session import _dataset_summary

    gdf = gpd.GeoDataFrame(
        {"population": [100, 150, 250]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )

    summary = _dataset_summary("demand_test", gdf)

    assert summary["numeric_preview"]["population"] == 500 / 3
    assert summary["numeric_summary"][0] == {
        "key": "population",
        "label": "total population",
        "value": 500.0,
        "stat": "total",
    }


def test_session_is_idempotent(client):
    first = client.post("/api/session")
    assert first.status_code == 200
    sid = first.cookies.get("soca_session")
    second = client.get("/api/session")
    assert second.status_code == 200
    # Cookie still present on the client, same session.
    assert client.cookies.get("soca_session") == sid


def test_list_problems(client):
    resp = client.get("/api/problems")
    assert resp.status_code == 200
    data = resp.json()
    assert "problems" in data
    assert len(data["problems"]) >= 3
    short_names = {p["short_name"] for p in data["problems"]}
    assert {"p-median", "p-center", "mclp", "lscp"}.issubset(short_names)


def test_network_status_empty_session(client):
    client.post("/api/session")
    resp = client.get("/api/network/status")
    assert resp.status_code == 200
    assert resp.json() == {"status": None, "error": None, "stats": None}


def test_network_refresh_without_aoi_is_400(client):
    client.post("/api/session")
    resp = client.post("/api/network/refresh")
    assert resp.status_code == 400


def test_events_stream_requires_session(client):
    """No cookie ⇒ 401 to avoid silently creating a session on GET."""
    fresh = TestClient(client.app)
    # stream=True would block; we just assert the initial response code.
    with fresh.stream("GET", "/api/events/stream") as resp:
        assert resp.status_code == 401


def test_list_datasets_empty(client):
    client.post("/api/session")
    resp = client.get("/api/data")
    assert resp.status_code == 200
    assert resp.json() == {"datasets": []}


def test_upload_geojson_dataset(client):
    client.post("/api/session")
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"demand": 100},
                "geometry": {"type": "Point", "coordinates": [0.1, 0.2]},
            },
            {
                "type": "Feature",
                "properties": {"demand": 50},
                "geometry": {"type": "Point", "coordinates": [0.15, 0.25]},
            },
        ],
    }
    files = [
        (
            "files",
            (
                "demand.geojson",
                json.dumps(geojson).encode("utf-8"),
                "application/geo+json",
            ),
        )
    ]
    resp = client.post("/api/data/upload", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["loaded"]) == 1
    assert body["loaded"][0]["num_features"] == 2
    # Round-trip GeoJSON
    resp2 = client.get("/api/data/demand.geojson.geojson")
    assert resp2.status_code == 200
    # Delete
    resp3 = client.delete("/api/data/demand.geojson")
    assert resp3.status_code == 200


def test_chat_stream_without_api_key(client, monkeypatch):
    """Without GEMINI_API_KEY the chat endpoint must respond 503, not crash."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client.post("/api/session")
    # Force a fresh agent creation by clearing any cached one
    from backend.services.session_store import get_default_store

    for sid in list(iter(get_default_store())):
        rec = get_default_store().get(sid)
        if rec:
            rec["_soca_agent"] = None

    resp = client.post("/api/chat/stream", json={"message": "hello"})
    assert resp.status_code == 503
