"""Tests for the snapshot-based reproducibility writer."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, Polygon


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
