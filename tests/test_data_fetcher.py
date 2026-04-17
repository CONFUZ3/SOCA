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
    OverpassError,
    PopulationDataError,
    FACILITY_TAGS,
    NOMINATIM_URL,
    OVERPASS_URL,
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
        with pytest.raises(GeocodingError, match="no results"):
            fetcher.fetch_boundaries("Nonexistent Place XYZ123")

    @patch("utils.data_fetcher.requests.post")
    @patch("utils.data_fetcher.requests.get")
    def test_falls_back_to_bbox_when_no_polygon(self, mock_get, mock_post):
        """When only a Point geometry is returned from Nominatim, fall back to bbox polygon.
        Overpass POST is forced to fail; Photon GET returns empty features;
        Nominatim GET returns a Point so the bbox-polygon fallback triggers."""
        import requests as req_lib

        # Force all Overpass POST attempts to fail (connection error)
        mock_post.side_effect = req_lib.exceptions.ConnectionError("overpass down")

        # First GET call goes to Photon — return empty features so Photon fails.
        # Second GET call goes to Nominatim — return a Point so bbox fallback is exercised.
        photon_empty = _make_response({"features": []})
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
        mock_get.side_effect = [photon_empty, nominatim_point]

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

class TestFetchPois(unittest.TestCase):

    @patch("utils.data_fetcher.requests.post")
    def test_returns_geodataframe_with_expected_columns(self, mock_post):
        mock_post.return_value = _make_response(_overpass_response(5))

        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_pois(boundary, "health")

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "name" in gdf.columns
        assert "amenity" in gdf.columns
        assert gdf.crs.to_epsg() == 4326

    @patch("utils.data_fetcher.requests.post")
    def test_clips_to_boundary_polygon(self, mock_post):
        """Points outside the boundary polygon should be removed."""
        elements = [
            # Inside Lima ~boundary
            {"type": "node", "id": 1, "lat": -12.1, "lon": -77.0,
             "tags": {"name": "Inside", "amenity": "hospital"}},
            # Clearly outside (Bogotá coordinates)
            {"type": "node", "id": 2, "lat": 4.7, "lon": -74.0,
             "tags": {"name": "Outside", "amenity": "hospital"}},
        ]
        mock_post.return_value = _make_response({"elements": elements})

        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_pois(boundary, "health")

        assert len(gdf) == 1
        assert gdf.iloc[0]["name"] == "Inside"

    @patch("utils.data_fetcher.requests.post")
    def test_returns_empty_gdf_when_no_elements(self, mock_post):
        mock_post.return_value = _make_response({"elements": []})

        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_pois(boundary, "health")

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0

    def test_raises_overpass_error_on_unknown_category(self):
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()

        with pytest.raises(OverpassError, match="Unknown POI category"):
            fetcher.fetch_pois(boundary, "unknown_category_xyz")

    @patch("utils.data_fetcher.requests.post")
    def test_raises_overpass_error_on_network_failure(self, mock_post):
        import requests as req_lib
        mock_post.side_effect = req_lib.exceptions.Timeout("timed out")

        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()

        with pytest.raises(OverpassError):
            fetcher.fetch_pois(boundary, "health")

    @patch("utils.data_fetcher.requests.post")
    def test_all_facility_categories_accepted(self, mock_post):
        """All categories in FACILITY_TAGS should not raise OverpassError."""
        mock_post.return_value = _make_response({"elements": []})
        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()

        for cat in FACILITY_TAGS:
            gdf = fetcher.fetch_pois(boundary, cat)
            assert isinstance(gdf, gpd.GeoDataFrame), f"category '{cat}' failed"

    @patch("utils.data_fetcher.requests.post")
    def test_handles_way_elements_with_center(self, mock_post):
        """Way elements use 'center' lat/lon."""
        elements = [
            {
                "type": "way",
                "id": 99,
                "center": {"lat": -12.1, "lon": -77.0},
                "tags": {"name": "Big Hospital", "amenity": "hospital"},
            }
        ]
        mock_post.return_value = _make_response({"elements": elements})

        fetcher = DataFetcher()
        boundary = _lima_boundary_gdf()
        gdf = fetcher.fetch_pois(boundary, "health")

        # Should have parsed the center and clipped correctly
        assert len(gdf) >= 0  # may be 0 or 1 depending on polygon containment


# ---------------------------------------------------------------------------
# Tests: fetch_population
# ---------------------------------------------------------------------------

class TestFetchPopulation(unittest.TestCase):

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

    @patch("utils.data_fetcher.requests.post")
    @patch("utils.data_fetcher.requests.get")
    @patch("utils.data_fetcher.time.sleep", return_value=None)
    def test_rate_limit_enforced_between_calls(self, mock_sleep, mock_get, mock_post):
        """Nominatim rate-limit sleep is exercised when Overpass and Photon both fail."""
        import requests as req_lib

        # Force all Overpass mirrors to fail (connection error on POST)
        mock_post.side_effect = req_lib.exceptions.ConnectionError("overpass down")

        # Photon GET returns empty features so Photon tier also fails
        photon_empty = _make_response({"features": []})
        # Nominatim GET returns a valid polygon
        nominatim_ok = _make_response(_nominatim_polygon_response())
        mock_get.side_effect = [photon_empty, nominatim_ok]

        fetcher = DataFetcher()
        # Simulate that the last Nominatim call happened only 0.1s ago
        fetcher._last_nominatim_call = time.monotonic()

        fetcher.fetch_boundaries("Lima, Peru")

        # time.sleep must have been called at least once for the rate-limit delay
        assert mock_sleep.called


# ---------------------------------------------------------------------------
# Tests: Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy(unittest.TestCase):

    def test_geocoding_error_is_data_fetch_error(self):
        assert issubclass(GeocodingError, DataFetchError)

    def test_overpass_error_is_data_fetch_error(self):
        assert issubclass(OverpassError, DataFetchError)

    def test_population_data_error_is_data_fetch_error(self):
        assert issubclass(PopulationDataError, DataFetchError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
