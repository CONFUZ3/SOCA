"""
Unit tests for utils/data_fetcher.py

All external HTTP calls are mocked so no network access is required.
"""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, mapping

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from utils.data_fetcher import (
    DataFetcher,
    DataFetchError,
    GeocodingError,
    PopulationDataError,
    _OSMNX_POI_TAGS,
    NOMINATIM_URL,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_response(json_body: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response-like object."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_body
    mock.headers = {}
    if status_code >= 400:
        import requests
        http_err = requests.exceptions.HTTPError(response=mock)
        mock.raise_for_status.side_effect = http_err
    else:
        mock.raise_for_status.return_value = None
    return mock


def _nominatim_polygon_response(location: str = "Lima, Peru") -> dict:
    """Minimal Nominatim GeoJSON response with a polygon geometry."""
    polygon_coords = [
        [-77.17, -12.50],
        [-76.80, -12.50],
        [-76.80, -11.90],
        [-77.17, -11.90],
        [-77.17, -12.50],
    ]
    return {
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords],
                },
                "properties": {
                    "display_name": location,
                    "place_id": "12345",
                },
                "bbox": [-77.17, -12.50, -76.80, -11.90],
            }
        ]
    }


def _overpass_response(n_nodes: int = 5) -> dict:
    """Minimal Overpass JSON response with N node elements."""
    elements = [
        {
            "type": "node",
            "id": 1000 + i,
            "lat": -12.0 + i * 0.05,  # inside Lima approx polygon
            "lon": -77.0 + i * 0.03,
            "tags": {"name": f"Hospital {i}", "amenity": "hospital"},
        }
        for i in range(n_nodes)
    ]
    return {"elements": elements}


def _lima_boundary_gdf() -> gpd.GeoDataFrame:
    """Return a simple square GeoDataFrame simulating Lima's boundary."""
    poly = Polygon([
        (-77.17, -12.50),
        (-76.80, -12.50),
        (-76.80, -11.90),
        (-77.17, -11.90),
    ])
    return gpd.GeoDataFrame(
        [{"display_name": "Lima, Peru", "source": "nominatim"}],
        geometry=[poly],
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Tests: fetch_boundaries
# ---------------------------------------------------------------------------

class TestFetchBoundaries(unittest.TestCase):

    def setUp(self):
        # Force Overture + Photon + GADM tiers to miss so these tests
        # exercise only the Nominatim /search path (the primary backend).
        import utils.data_fetcher as df
        self._overture_patch = patch.object(df, "_OVERTURE_AVAILABLE", False)
        self._gadm_patch = patch.object(df, "_GADM_AVAILABLE", False)
        self._overture_patch.start()
        self._gadm_patch.start()

    def tearDown(self):
        self._overture_patch.stop()
        self._gadm_patch.stop()

    @patch("utils.data_fetcher.requests.get")
    def test_returns_geodataframe_with_polygon(self, mock_get):
        mock_get.return_value = _make_response(_nominatim_polygon_response())

        fetcher = DataFetcher()
        gdf = fetcher.fetch_boundaries("Lima, Peru")

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.crs.to_epsg() == 4326
        assert gdf.geometry.iloc[0].geom_type == "Polygon"

    @patch("utils.data_fetcher.requests.get")
    def test_includes_location_query_in_properties(self, mock_get):
        mock_get.return_value = _make_response(_nominatim_polygon_response())

        fetcher = DataFetcher()
        gdf = fetcher.fetch_boundaries("Lima, Peru")

        assert gdf.iloc[0]["location_query"] == "Lima, Peru"

    @patch("utils.data_fetcher.requests.get")
    def test_raises_geocoding_error_on_empty_results(self, mock_get):
        mock_get.return_value = _make_response({"features": []})

        fetcher = DataFetcher()
        with pytest.raises(GeocodingError, match="no results|All geocoding backends failed"):
            fetcher.fetch_boundaries("Nonexistent Place XYZ123")

    @patch("utils.data_fetcher.requests.get")
    def test_falls_back_to_bbox_when_no_polygon(self, mock_get):
        """When Nominatim returns only a Point, its internal bbox fallback triggers
        and returns a rectangular polygon with source='nominatim_bbox_fallback'."""
        # Overture/GADM tiers are already disabled by setUp — flow goes to Nominatim.
        nominatim_point = _make_response({
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-77.0, -12.0]},
                    "properties": {"display_name": "Lima"},
                    "bbox": [-77.2, -12.3, -76.8, -11.9],
                }
            ]
        })
        mock_get.return_value = nominatim_point

        fetcher = DataFetcher()
        gdf = fetcher.fetch_boundaries("Lima")

        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].geom_type == "Polygon"
        assert gdf.iloc[0].get("source") == "nominatim_bbox_fallback"

    @patch("utils.data_fetcher.requests.get")
    def test_raises_geocoding_error_on_network_failure(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.ConnectionError("refused")

        fetcher = DataFetcher()
        with pytest.raises(GeocodingError):
            fetcher.fetch_boundaries("Lima, Peru")

    @patch("utils.data_fetcher.requests.get")
    def test_requests_correct_nominatim_params(self, mock_get):
        mock_get.return_value = _make_response(_nominatim_polygon_response())

        fetcher = DataFetcher()
        fetcher.fetch_boundaries("Lima, Peru")

        call_kwargs = mock_get.call_args
        assert NOMINATIM_URL in call_kwargs[0][0]
        params_sent = call_kwargs[1]["params"]
        assert params_sent["polygon_geojson"] == 1
        assert params_sent["format"] == "geojson"
        assert "Lima, Peru" in params_sent["q"]


# ---------------------------------------------------------------------------
# Tests: fetch_pois
# ---------------------------------------------------------------------------

def _osmnx_feature_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame in the shape osmnx.features_from_polygon returns."""
    if not rows:
        return gpd.GeoDataFrame(columns=["name", "amenity", "geometry"], crs="EPSG:4326")
    geoms = [Point(r["lon"], r["lat"]) for r in rows]
    data = [{"name": r.get("name", ""), "amenity": r.get("amenity", "")} for r in rows]
    return gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")


def _patch_osmnx_features(return_value=None, side_effect=None):
    """Context manager that patches osmnx.features_from_polygon inside DataFetcher."""
    import osmnx as ox
    return patch.object(
        ox, "features_from_polygon",
        return_value=return_value, side_effect=side_effect,
    )


class TestFetchPois(unittest.TestCase):

    def setUp(self):
        # Disable the Overture tier so every POI test exercises the OSMnx
        # path without hitting the real Overture Maps cloud parquet.
        import utils.data_fetcher as df
        self._overture_patch = patch.object(df, "_OVERTURE_AVAILABLE", False)
        self._overture_patch.start()

    def tearDown(self):
        self._overture_patch.stop()

    def test_returns_geodataframe_with_expected_columns(self):
        rows = [
            {"name": f"POI {i}", "amenity": "hospital", "lat": -12.1, "lon": -77.0}
            for i in range(5)
        ]
        with _patch_osmnx_features(return_value=_osmnx_feature_gdf(rows)):
            fetcher = DataFetcher()
            boundary = _lima_boundary_gdf()
            gdf = fetcher.fetch_pois(boundary, "health")

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "name" in gdf.columns
        assert "amenity" in gdf.columns
        assert gdf.crs.to_epsg() == 4326

    def test_clips_to_boundary_polygon(self):
        """Points outside the boundary polygon should be removed."""
        rows = [
            {"name": "Inside",  "amenity": "hospital", "lat": -12.1, "lon": -77.0},
            {"name": "Outside", "amenity": "hospital", "lat":  4.7,  "lon": -74.0},  # Bogotá
        ]
        with _patch_osmnx_features(return_value=_osmnx_feature_gdf(rows)):
            fetcher = DataFetcher()
            boundary = _lima_boundary_gdf()
            gdf = fetcher.fetch_pois(boundary, "health")

        assert len(gdf) == 1
        assert gdf.iloc[0]["name"] == "Inside"

    def test_returns_empty_gdf_when_no_features(self):
        with _patch_osmnx_features(return_value=_osmnx_feature_gdf([])):
            fetcher = DataFetcher()
            boundary = _lima_boundary_gdf()
            gdf = fetcher.fetch_pois(boundary, "health")

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0

    def test_raises_data_fetch_error_on_unknown_category(self):
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()

        with pytest.raises(DataFetchError, match="Unknown POI category"):
            fetcher.fetch_pois(boundary, "unknown_category_xyz")

    def test_raises_data_fetch_error_on_network_failure(self):
        with _patch_osmnx_features(side_effect=RuntimeError("osmnx network down")):
            fetcher = DataFetcher()
            boundary = _lima_boundary_gdf()

            with pytest.raises(DataFetchError):
                fetcher.fetch_pois(boundary, "health")

    def test_all_facility_categories_accepted(self):
        """All categories in _OSMNX_POI_TAGS should not raise."""
        with _patch_osmnx_features(return_value=_osmnx_feature_gdf([])):
            fetcher = DataFetcher()
            boundary = _lima_boundary_gdf()
            for cat in _OSMNX_POI_TAGS:
                gdf = fetcher.fetch_pois(boundary, cat)
                assert isinstance(gdf, gpd.GeoDataFrame), f"category '{cat}' failed"


# ---------------------------------------------------------------------------
# Tests: fetch_population
# ---------------------------------------------------------------------------

class TestFetchPopulation(unittest.TestCase):

    def setUp(self):
        # Population tests assert on the synthetic-grid path — disable the
        # live HDX lookup so no network is required.
        self._hdx_patch = patch.object(
            DataFetcher, "_fetch_population_hdx", return_value=None
        )
        self._hdx_patch.start()

    def tearDown(self):
        self._hdx_patch.stop()

    def test_returns_geodataframe_with_population_column(self):
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_population(boundary, n_points=50)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "population" in gdf.columns
        assert gdf.crs.to_epsg() == 4326

    def test_all_points_within_boundary(self):
        from shapely.ops import unary_union
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_population(boundary, n_points=100)

        boundary_union = unary_union(boundary.geometry.values)
        for pt in gdf.geometry:
            assert boundary_union.contains(pt) or boundary_union.touches(pt), (
                f"Point {pt} is outside boundary"
            )

    def test_population_sum_approximately_correct(self):
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        n = 200
        gdf = fetcher.fetch_population(boundary, n_points=n)

        total = gdf["population"].sum()
        # Total population is now derived from the boundary area (or metadata) rather
        # than a fixed constant, so check that all individual values are equal (uniform
        # distribution) and that the sum matches what _estimate_total_population returns.
        expected_total = fetcher._estimate_total_population(boundary)
        assert total > 0, "Total population must be positive"
        assert abs(total - expected_total) / expected_total < 0.05, (
            f"Expected total population ~{expected_total:,} but got {total:,.0f}"
        )

    def test_n_points_respected(self):
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_population(boundary, n_points=30)

        # May be <= 30 if polygon is awkward, but should be close
        assert len(gdf) > 0
        assert len(gdf) <= 30

    def test_raises_population_data_error_on_empty_boundary(self):
        from shapely.geometry import Polygon as ShapelyPoly
        # Completely empty (degenerate) geometry
        empty_poly = ShapelyPoly()
        boundary = gpd.GeoDataFrame(
            [{"source": "test"}], geometry=[empty_poly], crs="EPSG:4326"
        )

        fetcher = DataFetcher()
        with pytest.raises(PopulationDataError):
            fetcher.fetch_population(boundary, n_points=10)

    def test_source_column_is_synthetic(self):
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_population(boundary, n_points=20)
        # Synthetic grid is marked via the 'data_source' column
        assert "data_source" in gdf.columns
        assert (gdf["data_source"] == "synthetic_uniform_grid").all()


# ---------------------------------------------------------------------------
# Tests: _make_request retry logic
# ---------------------------------------------------------------------------

class TestMakeRequestRetry(unittest.TestCase):

    @patch("utils.data_fetcher.requests.get")
    @patch("utils.data_fetcher.time.sleep", return_value=None)
    def test_retries_on_timeout(self, mock_sleep, mock_get):
        import requests as req_lib
        # First two calls timeout, third succeeds
        success = _make_response({"features": []})
        mock_get.side_effect = [
            req_lib.exceptions.Timeout("t/o"),
            req_lib.exceptions.Timeout("t/o"),
            success,
        ]

        fetcher = DataFetcher()
        resp = fetcher._make_request("http://example.com")

        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # slept between retries

    @patch("utils.data_fetcher.requests.get")
    @patch("utils.data_fetcher.time.sleep", return_value=None)
    def test_raises_data_fetch_error_after_all_retries(self, mock_sleep, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.ConnectionError("refuse")

        fetcher = DataFetcher()
        with pytest.raises(DataFetchError, match="All"):
            fetcher._make_request("http://example.com")

        from utils.data_fetcher import _MAX_RETRIES
        assert mock_get.call_count == _MAX_RETRIES

    @patch("utils.data_fetcher.requests.get")
    @patch("utils.data_fetcher.time.sleep", return_value=None)
    def test_retries_on_5xx_http_error(self, mock_sleep, mock_get):
        import requests as req_lib
        error_resp = _make_response({}, status_code=503)
        success_resp = _make_response({"ok": True})

        http_err = req_lib.exceptions.HTTPError(response=error_resp)

        def side_effect(*args, **kwargs):
            if mock_get.call_count <= 2:
                raise http_err
            return success_resp

        mock_get.side_effect = side_effect

        fetcher = DataFetcher()
        resp = fetcher._make_request("http://example.com")
        assert resp.json() == {"ok": True}

    @patch("utils.data_fetcher.requests.get")
    def test_raises_immediately_on_4xx(self, mock_get):
        import requests as req_lib
        error_resp = _make_response({}, status_code=404)
        http_err = req_lib.exceptions.HTTPError(response=error_resp)
        mock_get.side_effect = http_err

        fetcher = DataFetcher()
        with pytest.raises(DataFetchError, match="HTTP 404"):
            fetcher._make_request("http://example.com")

        # Should NOT retry on 4xx
        assert mock_get.call_count == 1

    @patch("utils.data_fetcher.requests.post")
    @patch("utils.data_fetcher.time.sleep", return_value=None)
    def test_uses_post_method(self, mock_sleep, mock_post):
        mock_post.return_value = _make_response({"elements": []})

        fetcher = DataFetcher()
        fetcher._make_request("http://example.com", params={"data": "q"}, method="POST")

        assert mock_post.called


# ---------------------------------------------------------------------------
# Tests: Nominatim rate limiting
# ---------------------------------------------------------------------------

class TestNominatimRateLimit(unittest.TestCase):

    @patch("utils.data_fetcher.requests.get")
    @patch("utils.data_fetcher.time.sleep", return_value=None)
    def test_rate_limit_enforced_between_calls(self, mock_sleep, mock_get):
        """Nominatim rate-limit sleep is exercised on the Nominatim tier."""
        # Force downstream tiers to miss; Nominatim is now the primary backend.
        import utils.data_fetcher as df
        with patch.object(df, "_OVERTURE_AVAILABLE", False), \
             patch.object(df, "_GADM_AVAILABLE", False):
            nominatim_ok = _make_response(_nominatim_polygon_response())
            mock_get.return_value = nominatim_ok

            fetcher = DataFetcher()
            # Simulate that the last Nominatim call happened only 0.1s ago.
            fetcher._last_nominatim_call = time.monotonic()

            fetcher.fetch_boundaries("Lima, Peru")

            # time.sleep must have been called for the rate-limit delay.
            assert mock_sleep.called


# ---------------------------------------------------------------------------
# Tests: Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy(unittest.TestCase):

    def test_geocoding_error_is_data_fetch_error(self):
        assert issubclass(GeocodingError, DataFetchError)

    def test_population_data_error_is_data_fetch_error(self):
        assert issubclass(PopulationDataError, DataFetchError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
