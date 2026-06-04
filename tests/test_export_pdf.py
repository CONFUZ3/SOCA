"""Tests for the PDF report export and the removal of the extra
(Shapefile / GeoPackage) export endpoints.

The session-cookie plumbing is bypassed by populating a fresh session record
directly via the SessionStore and reusing the returned cookie.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point


@pytest.fixture
def client_with_solution():
    """A TestClient whose session holds a solved problem plus chat history."""
    from backend.main import app
    from backend.services.session_store import get_default_store

    store = get_default_store()

    candidates = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "capacity": [10, 20, 30]},
        geometry=[Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)],
        crs="EPSG:4326",
    )
    demand = gpd.GeoDataFrame(
        {"population": [5, 7, 9, 4]},
        geometry=[Point(0.1, 0.1), Point(0.9, 0.1), Point(0.2, 0.9), Point(0.5, 0.5)],
        crs="EPSG:4326",
    )

    with TestClient(app) as c:
        sess_resp = c.post("/api/session")
        assert sess_resp.status_code == 200
        sid = sess_resp.cookies.get("soca_session")
        assert sid

        rec = store.get(sid)
        assert rec is not None
        ps = rec["problem_state"]
        ps["problem_type"] = "p-median"
        ps["parameters"] = {"n_facilities": 2, "distance_metric": "euclidean"}
        ps["data"] = {"candidate_sites": candidates, "demand_points": demand}
        ps["solution"] = {
            "status": "Optimal",
            "objective_value": 12.5,
            "selected_facilities": [0, 2],
            "assignments": {0: 0, 1: 0, 2: 2, 3: 0},
            "solver": "pulp",
            "distance_metric_used": "euclidean",
            "metrics": {"avg_distance_km": 1.2},
            "warnings": [],
        }
        rec["messages"] = [
            {"role": "user", "content": "Solve a 2-facility p-median."},
            {
                "role": "assistant",
                "content": (
                    "## Solution summary\n"
                    "Selected **facility A** and **facility C**.\n\n"
                    "| Facility | Demand served |\n"
                    "| --- | --- |\n"
                    "| A | 3 |\n"
                    "| C | 1 |\n\n"
                    "- Average distance is `1.2 km`.\n"
                    "- Coverage looks balanced."
                ),
                "tool_calls": ["stage_optimization", "confirm_optimization"],
            },
        ]
        yield c


def test_pdf_export_includes_ai_analysis(client_with_solution):
    resp = client_with_solution.get("/api/export/pdf")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/pdf")
    # A valid PDF and non-trivial payload (narration + tables add bulk).
    assert resp.content[:5] == b"%PDF-"
    assert len(resp.content) > 1500


def test_removed_export_endpoints_are_gone(client_with_solution):
    for path in ("/api/export/shapefile", "/api/export/geopackage"):
        resp = client_with_solution.get(path)
        assert resp.status_code == 404, f"{path} should be removed, got {resp.status_code}"


def test_core_exports_still_work(client_with_solution):
    for path, ctype in (
        ("/api/export/geojson", "application/geo+json"),
        ("/api/export/csv", "text/csv"),
    ):
        resp = client_with_solution.get(path)
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith(ctype)
