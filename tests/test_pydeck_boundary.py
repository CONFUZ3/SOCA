"""Tests for the AOI boundary overlay in PyDeckVisualizer.

Network distance is the default metric and the AOI polygon is surfaced as a
dedicated GeoJsonLayer so the user always sees the Area of Interest on the
visualization map. These tests check the layer is built correctly and that
the view-state calculation takes the boundary into account.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from utils.pydeck_visualizer import PyDeckVisualizer


def _aoi_gdf():
    poly = Polygon([(74.30, 31.50), (74.40, 31.50), (74.40, 31.60), (74.30, 31.60)])
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")


def _demand_gdf():
    pts = [Point(74.33, 31.55), Point(74.37, 31.57)]
    return gpd.GeoDataFrame({"id": [1, 2]}, geometry=pts, crs="EPSG:4326")


def test_boundary_layer_added_when_boundary_provided():
    viz = PyDeckVisualizer(basemap_style="light")
    deck = viz.create_map(
        data={"demand_points": _demand_gdf()},
        boundary=_aoi_gdf(),
    )

    # At least one GeoJsonLayer should be present for the AOI.
    layer_types = [type(l).__name__ + ":" + getattr(l, "type", "") for l in deck.layers]
    assert any("GeoJsonLayer" in lt for lt in layer_types), (
        f"Expected a GeoJsonLayer for AOI boundary, got {layer_types}"
    )


def test_boundary_layer_respects_show_boundary_toggle():
    viz = PyDeckVisualizer(basemap_style="light")
    deck = viz.create_map(
        data={"demand_points": _demand_gdf()},
        boundary=_aoi_gdf(),
        viz_config={"show_boundary": False},
    )

    layer_types = [getattr(l, "type", "") for l in deck.layers]
    assert "GeoJsonLayer" not in layer_types, (
        "Boundary layer must be suppressed when show_boundary is False"
    )


def test_boundary_layer_skipped_when_no_polygon_rows():
    viz = PyDeckVisualizer(basemap_style="light")
    point_only = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(74.33, 31.55)], crs="EPSG:4326"
    )
    deck = viz.create_map(
        data={"demand_points": _demand_gdf()},
        boundary=point_only,
    )

    layer_types = [getattr(l, "type", "") for l in deck.layers]
    assert "GeoJsonLayer" not in layer_types


def test_view_state_uses_boundary_when_no_data():
    viz = PyDeckVisualizer(basemap_style="light")
    deck = viz.create_map(
        data={},
        boundary=_aoi_gdf(),
    )
    vs = deck.initial_view_state
    # Camera should be centred on the AOI, not the default NYC fallback.
    assert 31.4 < vs.latitude < 31.7
    assert 74.2 < vs.longitude < 74.5


def test_boundary_reprojected_to_wgs84():
    """A boundary in a projected CRS must be reprojected before rendering."""
    poly = Polygon([(74.30, 31.50), (74.40, 31.50), (74.40, 31.60), (74.30, 31.60)])
    gdf_wgs = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    gdf_proj = gdf_wgs.to_crs("EPSG:32643")

    viz = PyDeckVisualizer(basemap_style="light")
    deck = viz.create_map(
        data={"demand_points": _demand_gdf()},
        boundary=gdf_proj,
    )

    # Find the boundary layer and inspect its data payload.
    boundary_layer = next(
        (l for l in deck.layers if getattr(l, "type", "") == "GeoJsonLayer"),
        None,
    )
    assert boundary_layer is not None
    fc = boundary_layer.data
    coords = fc["features"][0]["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    # Reprojected back to WGS84, coords should be in the AOI lon/lat range.
    assert all(-180 <= x <= 180 for x in xs)
    assert all(-90 <= y <= 90 for y in ys)
    assert min(xs) > 74.0 and max(xs) < 75.0
    assert min(ys) > 31.0 and max(ys) < 32.0
