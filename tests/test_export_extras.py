"""Round-trip tests for the new Shapefile + GeoPackage exports
and the snapshot-based reproducibility writer.

These tests bypass session-cookie plumbing by populating a fresh session
record directly via the SessionStore and then issuing the export request
with the returned cookie.
"""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point, Polygon


@pytest.fixture
def client_with_solution(tmp_path, monkeypatch):
    """A TestClient whose session already holds a small solved problem."""
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
        ps["data"] = {
            "candidate_sites": candidates,
            "demand_points": demand,
        }
        ps["solution"] = {
            "status": "Optimal",
            "objective_value": 12.5,
            "selected_facilities": [0, 2],
            "assignments": {0: 0, 1: 0, 2: 2, 3: 0},
            "solver": "pulp",
            "distance_metric_used": "euclidean",
            "metrics": {},
            "warnings": [],
        }
        yield c


def test_shapefile_export_roundtrip(client_with_solution):
    resp = client_with_solution.get("/api/export/shapefile")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
        # Each layer present has a .shp + sidecars; metadata sidecar always.
        assert "facilities.shp" in names
        assert "demand.shp" in names
        assert "assignments.shp" in names
        assert "metadata.json" in names

        meta = json.loads(zf.read("metadata.json"))
        assert meta["objective_value"] == 12.5
        assert meta["n_selected"] == 2

        # Round-trip facilities through geopandas to validate geometry.
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            fac = gpd.read_file(str(Path(tmp) / "facilities.shp"))
            assert len(fac) == 2
            assert fac.crs is not None and fac.crs.to_epsg() == 4326
            asg = gpd.read_file(str(Path(tmp) / "assignments.shp"))
            assert len(asg) == 4
            assert (asg.geometry.geom_type == "LineString").all()


def test_geopackage_export_roundtrip(client_with_solution, tmp_path):
    resp = client_with_solution.get("/api/export/geopackage")
    assert resp.status_code == 200, resp.text

    path = tmp_path / "out.gpkg"
    path.write_bytes(resp.content)
    if True:

        layers = set(gpd.list_layers(str(path))["name"]) if hasattr(gpd, "list_layers") else None
        # Fallback for older geopandas without list_layers:
        if layers is None:
            import fiona
            layers = set(fiona.listlayers(str(path)))

        assert {"facilities", "demand", "assignments"}.issubset(layers)

        fac = gpd.read_file(str(path), layer="facilities")
        assert len(fac) == 2
        assert fac.crs is not None and fac.crs.to_epsg() == 4326

        # Metadata table written via sqlite.
        with sqlite3.connect(str(path)) as conn:
            rows = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        assert json.loads(rows["objective_value"]) == 12.5
        assert json.loads(rows["n_selected"]) == 2


def test_repro_logger_snapshot_writes_geopackages(tmp_path, monkeypatch):
    """ReproducibilityLogger.log_run with GeoDataFrames must write a snapshot dir."""
    from utils import repro_logger as rl

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rl, "_runs_dir", lambda: runs_root)

    ReproducibilityLogger = rl.ReproducibilityLogger
    build_run_payload = rl.build_run_payload

    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    demand = gpd.GeoDataFrame(
        {"population": [5, 7]},
        geometry=[Point(0.2, 0.2), Point(0.8, 0.8)],
        crs="EPSG:4326",
    )
    candidates = gpd.GeoDataFrame(
        {"name": ["A", "B"]},
        geometry=[Point(0.1, 0.5), Point(0.9, 0.5)],
        crs="EPSG:4326",
    )

    payload = build_run_payload(
        boundary_polygon=boundary,
        demand_gdf=demand,
        candidates_gdf=candidates,
        distance_method="euclidean",
        solver="pulp",
        solver_params={"n_facilities": 1},
        objective_value=2.0,
        selected_facility_ids=[0],
    )

    solution = {
        "selected_facilities": [0],
        "assignments": {0: 0, 1: 0},
        "objective_value": 2.0,
        "metrics": {"avg_distance": 1.0},
        "warnings": [],
    }

    log_path = ReproducibilityLogger().log_run(
        payload,
        demand_gdf=demand,
        candidates_gdf=candidates,
        boundary_polygon=boundary,
        solution=solution,
    )
    assert log_path.exists()

    rec = json.loads(log_path.read_text())
    assert "snapshot_dir" in rec
    snap_dir = Path(rec["snapshot_dir"])
    assert snap_dir.is_dir()
    assert (snap_dir / "demand.gpkg").exists()
    assert (snap_dir / "candidates.gpkg").exists()
    assert (snap_dir / "aoi.gpkg").exists()
    assert (snap_dir / "solution.json").exists()

    # Round-trip the snapshots back through GeoPandas.
    d = gpd.read_file(str(snap_dir / "demand.gpkg"))
    assert len(d) == 2
    c = gpd.read_file(str(snap_dir / "candidates.gpkg"))
    assert len(c) == 2
    a = gpd.read_file(str(snap_dir / "aoi.gpkg"))
    assert len(a) == 1

    sol = json.loads((snap_dir / "solution.json").read_text())
    assert sol["objective_value"] == 2.0
    assert sol["selected_facilities"] == [0]
