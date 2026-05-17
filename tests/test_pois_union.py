"""Tests for the Overture ∪ Overpass POI union and the Overpass tier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from utils.fetchers.errors import DataFetchError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _boundary_gdf() -> gpd.GeoDataFrame:
    """Square boundary roughly over Nairobi."""
    poly = Polygon([
        (36.70, -1.45),
        (37.05, -1.45),
        (37.05, -1.15),
        (36.70, -1.15),
    ])
    return gpd.GeoDataFrame(
        [{"name": "Nairobi-test"}], geometry=[poly], crs="EPSG:4326"
    )


def _pois_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame(
            columns=["name", "amenity", "geometry"], crs="EPSG:4326"
        )
    geoms = [Point(r["lon"], r["lat"]) for r in rows]
    data = [{"name": r.get("name", ""), "amenity": r.get("amenity", "hospital")}
            for r in rows]
    return gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Overpass tier — query builder + element conversion
# ---------------------------------------------------------------------------

class TestOverpassQueryBuilder:
    def test_query_includes_node_way_relation_per_tag(self):
        from utils.fetchers.pois_overpass import _build_overpass_query

        body = _build_overpass_query(
            [("amenity", "hospital"), ("healthcare", "clinic")],
            south=-1.5, west=36.7, north=-1.2, east=37.0,
        )
        # Three element kinds × two tag pairs
        assert body.count('["amenity"="hospital"]') == 3
        assert body.count('["healthcare"="clinic"]') == 3
        assert "out:json" in body
        assert "out center tags" in body
        assert "-1.5,36.7,-1.2,37.0" in body


class TestElementsToGdf:
    def test_node_emits_point(self):
        from utils.fetchers.pois_overpass import _elements_to_gdf

        elements = [{
            "type": "node",
            "id": 1,
            "lon": 36.8, "lat": -1.3,
            "tags": {"name": "Test Hospital", "amenity": "hospital"},
        }]
        gdf = _elements_to_gdf(elements, "health")
        assert len(gdf) == 1
        assert gdf.iloc[0]["name"] == "Test Hospital"
        assert gdf.iloc[0]["amenity"] == "hospital"
        assert gdf.crs.to_epsg() == 4326

    def test_way_uses_center(self):
        from utils.fetchers.pois_overpass import _elements_to_gdf

        elements = [{
            "type": "way",
            "id": 99,
            "center": {"lon": 36.85, "lat": -1.28},
            "tags": {"name": "Big Clinic", "healthcare": "clinic"},
        }]
        gdf = _elements_to_gdf(elements, "health")
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].x == pytest.approx(36.85)
        # Matched value should be the healthcare=clinic tag value, not category
        assert gdf.iloc[0]["amenity"] == "clinic"

    def test_drops_elements_without_geometry(self):
        from utils.fetchers.pois_overpass import _elements_to_gdf

        elements = [
            {"type": "way", "id": 1, "tags": {"amenity": "hospital"}},  # no center
            {"type": "node", "id": 2, "lon": 36.8, "lat": -1.3,
             "tags": {"amenity": "hospital"}},
        ]
        gdf = _elements_to_gdf(elements, "health")
        assert len(gdf) == 1

    def test_empty_elements_returns_empty_gdf(self):
        from utils.fetchers.pois_overpass import _elements_to_gdf

        gdf = _elements_to_gdf([], "health")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0


class TestFetchPoisViaOverpass:
    def test_populated_response_returns_features(self):
        from utils.fetchers import pois_overpass

        payload = {"elements": [
            {"type": "node", "id": 1, "lon": 36.8, "lat": -1.3,
             "tags": {"name": "Hospital A", "amenity": "hospital"}},
            {"type": "node", "id": 2, "lon": 36.9, "lat": -1.35,
             "tags": {"name": "Clinic B", "amenity": "clinic"}},
        ]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload

        with patch.object(pois_overpass, "make_request", return_value=mock_resp):
            gdf = pois_overpass.fetch_pois_via_overpass(
                (36.7, -1.45, 37.05, -1.15), "health",
            )
        assert len(gdf) == 2
        assert set(gdf["amenity"]) == {"hospital", "clinic"}

    def test_empty_response_returns_empty_gdf(self):
        from utils.fetchers import pois_overpass

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"elements": []}

        with patch.object(pois_overpass, "make_request", return_value=mock_resp):
            gdf = pois_overpass.fetch_pois_via_overpass(
                (36.7, -1.45, 37.05, -1.15), "health",
            )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0

    def test_unknown_category_returns_empty(self):
        from utils.fetchers import pois_overpass

        gdf = pois_overpass.fetch_pois_via_overpass(
            (0, 0, 1, 1), "nonexistent_category",
        )
        assert len(gdf) == 0

    def test_non_json_response_raises(self):
        from utils.fetchers import pois_overpass

        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("not json")

        with patch.object(pois_overpass, "make_request", return_value=mock_resp):
            with pytest.raises(DataFetchError, match="non-JSON"):
                pois_overpass.fetch_pois_via_overpass(
                    (36.7, -1.45, 37.05, -1.15), "health",
                )

    def test_make_request_failure_propagates_as_datafetch(self):
        from utils.fetchers import pois_overpass

        with patch.object(
            pois_overpass, "make_request",
            side_effect=DataFetchError("Overpass 503"),
        ):
            with pytest.raises(DataFetchError, match="Overpass"):
                pois_overpass.fetch_pois_via_overpass(
                    (36.7, -1.45, 37.05, -1.15), "health",
                )


# ---------------------------------------------------------------------------
# Union + dedup
# ---------------------------------------------------------------------------

class TestDedupAndUnion:
    def test_both_empty_returns_empty(self):
        from utils.fetchers.pois import _dedup_and_union

        gdf = _dedup_and_union(_pois_gdf([]), _pois_gdf([]), radius_m=50)
        assert len(gdf) == 0

    def test_only_overture_keeps_all(self):
        from utils.fetchers.pois import _dedup_and_union

        ovr = _pois_gdf([
            {"name": "A", "lon": 36.8, "lat": -1.3},
            {"name": "B", "lon": 36.9, "lat": -1.4},
        ])
        gdf = _dedup_and_union(ovr, _pois_gdf([]), radius_m=50)
        assert len(gdf) == 2
        assert set(gdf["data_source"]) == {"overture_pois"}

    def test_only_overpass_keeps_all(self):
        from utils.fetchers.pois import _dedup_and_union

        osm = _pois_gdf([
            {"name": "X", "lon": 36.8, "lat": -1.3},
        ])
        gdf = _dedup_and_union(_pois_gdf([]), osm, radius_m=50)
        assert len(gdf) == 1
        assert gdf.iloc[0]["data_source"] == "osm_overpass"

    def test_coincident_similar_names_dedup_to_union(self):
        from utils.fetchers.pois import _dedup_and_union

        # Two points within 50 m of each other (~0.0001 degrees ≈ 11 m at equator)
        ovr = _pois_gdf([{"name": "St Mary Hospital", "lon": 36.8000, "lat": -1.3000}])
        osm = _pois_gdf([{"name": "St. Mary's Hospital", "lon": 36.8001, "lat": -1.3001}])

        gdf = _dedup_and_union(ovr, osm, radius_m=50)
        # Overpass duplicate dropped; the single Overture row tagged as union.
        assert len(gdf) == 1
        assert gdf.iloc[0]["data_source"] == "overture+osm"

    def test_distant_points_both_kept(self):
        from utils.fetchers.pois import _dedup_and_union

        ovr = _pois_gdf([{"name": "Hospital A", "lon": 36.80, "lat": -1.30}])
        osm = _pois_gdf([{"name": "Hospital B", "lon": 36.90, "lat": -1.40}])

        gdf = _dedup_and_union(ovr, osm, radius_m=50)
        assert len(gdf) == 2
        assert set(gdf["data_source"]) == {"overture_pois", "osm_overpass"}

    def test_close_but_dissimilar_names_both_kept(self):
        from utils.fetchers.pois import _dedup_and_union

        # Same coordinates, distinct names → likely two facilities sharing a campus
        ovr = _pois_gdf([{"name": "Aga Khan Hospital", "lon": 36.80, "lat": -1.30}])
        osm = _pois_gdf([{"name": "Mama Lucy Clinic", "lon": 36.8001, "lat": -1.3001}])

        gdf = _dedup_and_union(ovr, osm, radius_m=50)
        assert len(gdf) == 2


# ---------------------------------------------------------------------------
# fetch_pois orchestrator
# ---------------------------------------------------------------------------

class TestFetchPoisOrchestrator:
    def test_unknown_category_raises(self):
        from utils.fetchers.pois import fetch_pois

        with pytest.raises(DataFetchError, match="Unknown POI category"):
            fetch_pois(_boundary_gdf(), "definitely_not_a_category")

    def test_overture_empty_overpass_populated_succeeds(self):
        from utils.fetchers import pois

        ovr_empty = _pois_gdf([])
        osm_pop = _pois_gdf([
            {"name": "Hospital A", "lon": 36.8, "lat": -1.3, "amenity": "hospital"},
            {"name": "Clinic B", "lon": 36.85, "lat": -1.32, "amenity": "clinic"},
        ])

        with (
            patch.object(pois, "fetch_pois_via_overture", return_value=ovr_empty),
            patch.object(pois, "fetch_pois_via_overpass", return_value=osm_pop),
        ):
            gdf = pois.fetch_pois(_boundary_gdf(), "health")

        assert len(gdf) == 2
        assert set(gdf["data_source"]) == {"osm_overpass"}
        counts = gdf.attrs["tier_counts"]
        assert counts["overture"] == 0
        assert counts["overpass"] == 2

    def test_overture_populated_overpass_empty_succeeds(self):
        from utils.fetchers import pois

        ovr_pop = _pois_gdf([
            {"name": "Hospital A", "lon": 36.8, "lat": -1.3},
        ])
        osm_empty = _pois_gdf([])

        with (
            patch.object(pois, "fetch_pois_via_overture", return_value=ovr_pop),
            patch.object(pois, "fetch_pois_via_overpass", return_value=osm_empty),
        ):
            gdf = pois.fetch_pois(_boundary_gdf(), "health")

        assert len(gdf) == 1
        assert gdf.iloc[0]["data_source"] == "overture_pois"

    def test_both_empty_raises(self):
        from utils.fetchers import pois

        with (
            patch.object(pois, "fetch_pois_via_overture", return_value=_pois_gdf([])),
            patch.object(pois, "fetch_pois_via_overpass", return_value=_pois_gdf([])),
        ):
            with pytest.raises(DataFetchError, match="No 'health' POIs found"):
                pois.fetch_pois(_boundary_gdf(), "health")

    def test_overture_error_overpass_success_still_succeeds(self):
        """A transient Overture failure must not lose the run when OSM has data."""
        from utils.fetchers import pois

        osm_pop = _pois_gdf([
            {"name": "Hospital A", "lon": 36.8, "lat": -1.3},
        ])

        with (
            patch.object(pois, "fetch_pois_via_overture",
                         side_effect=RuntimeError("Overture S3 down")),
            patch.object(pois, "fetch_pois_via_overpass", return_value=osm_pop),
        ):
            gdf = pois.fetch_pois(_boundary_gdf(), "health")

        assert len(gdf) == 1
        assert gdf.attrs.get("tier_errors", {}).get("overture", "").startswith("Overture S3")

    def test_union_dedups_when_both_populated(self):
        from utils.fetchers import pois

        ovr = _pois_gdf([{"name": "St Mary Hospital", "lon": 36.8000, "lat": -1.3000}])
        osm = _pois_gdf([
            {"name": "St. Mary's Hospital", "lon": 36.8001, "lat": -1.3001},  # dup
            {"name": "Other Clinic", "lon": 36.95, "lat": -1.40},
        ])

        with (
            patch.object(pois, "fetch_pois_via_overture", return_value=ovr),
            patch.object(pois, "fetch_pois_via_overpass", return_value=osm),
        ):
            gdf = pois.fetch_pois(_boundary_gdf(), "health")

        # 1 union-confirmed + 1 OSM-only
        assert len(gdf) == 2
        sources = set(gdf["data_source"])
        assert "overture+osm" in sources
        assert "osm_overpass" in sources
