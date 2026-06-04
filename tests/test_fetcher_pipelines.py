"""Unit tests for the refactored utils/fetchers package.

Covers the concrete fixes called out in the plan:
- HDX population column resolution (priority, year preference, dtype gate,
  metadata-column rejection).
- HDX lat/lon column variant detection + auto-swap.
- Thread-safe Nominatim rate limiter.
- Polygon validation (empty / invalid repair / bbox sanity).
- Scale-mismatch hard rejection.
"""

from __future__ import annotations

import threading
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from utils.fetchers.errors import GeocodingError, PopulationDataError
from utils.fetchers.http import NominatimRateLimiter
from utils.fetchers.population import (
    _resolve_latlon_columns,
    _resolve_population_column,
    _sanitise_latlon,
)
from utils.fetchers.validation import validate_polygon, validate_scale_match


# ---------------------------------------------------------------------------
# _resolve_population_column
# ---------------------------------------------------------------------------

class TestResolvePopulationColumn:
    def test_exact_population_match(self):
        df = pd.DataFrame({"lat": [1.0], "lon": [2.0], "population": [42.0]})
        assert _resolve_population_column(df) == "population"

    def test_exact_pop_name(self):
        df = pd.DataFrame({"pop": [1, 2, 3]})
        assert _resolve_population_column(df) == "pop"

    def test_prefers_most_recent_year(self):
        df = pd.DataFrame({
            "lat": [1.0, 2.0],
            "population_2015": [10, 20],
            "population_2020": [30, 40],
            "population_2010": [5, 6],
        })
        assert _resolve_population_column(df) == "population_2020"

    def test_rejects_metadata_suffix_columns(self):
        # population_method and population_source are text metadata, not counts.
        df = pd.DataFrame({
            "population_method": ["kontur", "kontur"],
            "population_source": ["v1", "v1"],
            "population_year": ["2020", "2020"],
            "population": [100.0, 200.0],
        })
        assert _resolve_population_column(df) == "population"

    def test_rejects_non_numeric_column(self):
        # Only candidate is a string — must not be chosen.
        df = pd.DataFrame({"population": ["ten", "twenty", "thirty"]})
        assert _resolve_population_column(df) is None

    def test_returns_none_when_no_candidates(self):
        df = pd.DataFrame({
            "h3_index": ["abc", "def"],
            "country": ["PAK", "PAK"],
            "region": ["North", "South"],
        })
        assert _resolve_population_column(df) is None

    def test_never_picks_last_column_blindly(self):
        # Legacy bug: fall-through to df.columns[-1] could pick a string column.
        df = pd.DataFrame({
            "latitude": [1.0, 2.0],
            "longitude": [3.0, 4.0],
            "h3_index": ["abc1", "abc2"],   # last column — must NOT be chosen
        })
        assert _resolve_population_column(df) is None

    def test_coerces_mostly_numeric(self):
        df = pd.DataFrame({"population": [1.0, 2.0, 3.0, 4.0, 5.0, "x"]})
        # 1/6 ≈ 16.7% NaN after coerce — outside the 5% default gate.
        assert _resolve_population_column(df) is None
        # Loosen threshold explicitly for this edge.
        assert _resolve_population_column(df, max_na_frac=0.25) == "population"

    def test_fuzzy_contains(self):
        df = pd.DataFrame({"total_persons_est": [10, 20, 30]})
        assert _resolve_population_column(df) == "total_persons_est"


# ---------------------------------------------------------------------------
# _resolve_latlon_columns + _sanitise_latlon
# ---------------------------------------------------------------------------

class TestResolveLatLon:
    def test_standard_names(self):
        df = pd.DataFrame({"latitude": [0.0], "longitude": [0.0]})
        lat, lon = _resolve_latlon_columns(df)
        assert (lat, lon) == ("latitude", "longitude")

    def test_short_names(self):
        df = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
        assert _resolve_latlon_columns(df) == ("lat", "lon")

    def test_mixed_case_and_underscores(self):
        df = pd.DataFrame({"Lat_DD": [0.0], "Lon_DD": [0.0]})
        assert _resolve_latlon_columns(df) == ("Lat_DD", "Lon_DD")

    def test_xy(self):
        df = pd.DataFrame({"y": [0.0], "x": [0.0]})
        assert _resolve_latlon_columns(df) == ("y", "x")

    def test_missing(self):
        df = pd.DataFrame({"foo": [1]})
        assert _resolve_latlon_columns(df) == (None, None)


class TestSanitiseLatLon:
    def test_valid(self):
        df = pd.DataFrame({"lat": [10.0, 20.0], "lon": [30.0, 40.0]})
        out_df, lat, lon = _sanitise_latlon(df, "lat", "lon")
        assert (lat, lon) == ("lat", "lon")
        assert len(out_df) == 2

    def test_swapped_detection(self):
        # 95%+ of rows have "lat" > 90 (clearly longitudes) — should swap.
        df = pd.DataFrame({
            "lat_col": [120.0, -85.0, 150.0, -170.0, 100.0],
            "lon_col": [45.0, 12.0, -30.0, 60.0, 5.0],
        })
        out_df, lat, lon = _sanitise_latlon(df, "lat_col", "lon_col")
        assert lat == "lon_col"
        assert lon == "lat_col"

    def test_unrecoverable_raises(self):
        df = pd.DataFrame({
            "lat": [500.0, 600.0, 700.0],
            "lon": [800.0, 900.0, 1000.0],
        })
        with pytest.raises(PopulationDataError):
            _sanitise_latlon(df, "lat", "lon")


# ---------------------------------------------------------------------------
# NominatimRateLimiter (thread-safety)
# ---------------------------------------------------------------------------

class TestNominatimRateLimiter:
    def test_single_thread_respects_interval(self):
        limiter = NominatimRateLimiter(min_interval_sec=0.2)
        start = time.monotonic()
        limiter.wait()
        limiter.wait()
        limiter.wait()
        elapsed = time.monotonic() - start
        # 3 calls at 0.2s spacing → at least ~0.4s between first and third.
        assert elapsed >= 0.4

    def test_concurrent_threads_serialise(self):
        limiter = NominatimRateLimiter(min_interval_sec=0.1)
        n_threads = 4
        timestamps: list[float] = []
        lock = threading.Lock()

        def worker():
            limiter.wait()
            with lock:
                timestamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        # 4 calls at 0.1s min interval → total >= 0.3s (between first and last).
        assert elapsed >= 0.3
        timestamps.sort()
        deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        # Every subsequent call must be at least ~min_interval after the previous.
        # Allow 10% tolerance for scheduler jitter.
        assert all(d >= 0.09 for d in deltas), deltas


# ---------------------------------------------------------------------------
# validate_polygon
# ---------------------------------------------------------------------------

def _simple_gdf(geom):
    return gpd.GeoDataFrame({"name": ["x"]}, geometry=[geom], crs="EPSG:4326")


class TestValidatePolygon:
    def test_happy_path(self):
        gdf = _simple_gdf(box(-1, -1, 1, 1))
        out = validate_polygon(gdf, source_label="test")
        assert out.crs.to_epsg() == 4326

    def test_empty_raises(self):
        gdf = gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")
        with pytest.raises(GeocodingError):
            validate_polygon(gdf)

    def test_wrong_crs_reprojected(self):
        gdf = gpd.GeoDataFrame(
            {"name": ["x"]},
            geometry=[box(0, 0, 1000, 1000)],
            crs="EPSG:3857",
        )
        out = validate_polygon(gdf)
        assert out.crs.to_epsg() == 4326

    def test_non_polygon_raises(self):
        from shapely.geometry import LineString
        gdf = _simple_gdf(LineString([(0, 0), (1, 1)]))
        with pytest.raises(GeocodingError):
            validate_polygon(gdf)

    def test_repairs_self_intersecting(self):
        # Bowtie self-intersection.
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        assert not bowtie.is_valid
        out = validate_polygon(_simple_gdf(bowtie))
        assert out.geometry.iloc[0].is_valid


# ---------------------------------------------------------------------------
# validate_scale_match
# ---------------------------------------------------------------------------

class TestValidateScaleMatch:
    def test_country_polygon_as_city_is_hard_rejected(self):
        # A polygon covering ~40°×30° = 1200 sq-deg is clearly country scale.
        huge = _simple_gdf(box(-20, -15, 20, 15))
        ok, reason = validate_scale_match(huge, "city")
        assert ok is False
        assert "too large" in reason

    def test_neighborhood_polygon_passes_city_check(self):
        small = _simple_gdf(box(-0.01, -0.01, 0.01, 0.01))
        ok, _ = validate_scale_match(small, "city")
        # City range is 0.01–5 sq-deg; 0.02×0.02 = 0.0004 — off but not > 100× too small
        # (lower bound 0.01; 0.01/100 = 0.0001, and 0.0004 > 0.0001) → soft pass.
        assert ok is True

    def test_city_polygon_passes_city_check(self):
        mid = _simple_gdf(box(-0.5, -0.5, 0.5, 0.5))  # 1 sq-deg
        ok, _ = validate_scale_match(mid, "city")
        assert ok is True


# ---------------------------------------------------------------------------
# Legacy import surface is preserved.
# ---------------------------------------------------------------------------

def test_legacy_import_path_still_works():
    from utils.data_fetcher import (  # noqa: F401
        DataFetcher,
        DataFetchError,
        GeocodingError as LegacyGeocodingError,
        PopulationDataError,
        NOMINATIM_URL,
        _MAX_RETRIES,
    )
    assert LegacyGeocodingError is GeocodingError
    assert _MAX_RETRIES == 3


# ---------------------------------------------------------------------------
# Overture release resolver — offline behaviour
# ---------------------------------------------------------------------------

class TestOvertureReleaseResolver:
    def test_env_override_wins(self, monkeypatch):
        from utils.fetchers import overture_release as mod
        mod.get_overture_release.cache_clear()
        monkeypatch.setenv("OVERTURE_RELEASE", "2099-12-31.7")
        assert mod.get_overture_release() == "2099-12-31.7"
        mod.get_overture_release.cache_clear()

    def test_falls_back_to_default_on_network_error(self, monkeypatch):
        from utils.fetchers import overture_release as mod
        from utils.fetchers.constants import _OVERTURE_DEFAULT_RELEASE
        mod.get_overture_release.cache_clear()
        monkeypatch.delenv("OVERTURE_RELEASE", raising=False)

        def _boom(*a, **kw):
            raise RuntimeError("simulated network failure")
        monkeypatch.setattr(mod, "make_request", _boom, raising=False)
        # Patch at import site too (import is deferred inside the function)
        import utils.fetchers.http as http_mod
        monkeypatch.setattr(http_mod, "make_request", _boom)

        assert mod.get_overture_release() == _OVERTURE_DEFAULT_RELEASE
        mod.get_overture_release.cache_clear()


# ---------------------------------------------------------------------------
# Overture boundary fetcher — DuckDB path is chosen when available
# ---------------------------------------------------------------------------

class TestOvertureBoundaryDispatch:
    def test_duckdb_path_selected_when_available(self, monkeypatch):
        from utils.fetchers import boundaries
        from utils.fetchers import overture_duckdb as od

        monkeypatch.setattr(od, "is_available", lambda: True)
        called = {"duckdb": 0, "pyclient": 0}

        def fake_duckdb(loc, al, sc):
            called["duckdb"] += 1
            import geopandas as gpd
            from shapely.geometry import box
            return gpd.GeoDataFrame([{"name": loc}], geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

        def fake_pyclient(loc, al, sc):
            called["pyclient"] += 1
            raise AssertionError("pyclient should not run")

        monkeypatch.setattr(boundaries, "_fetch_via_duckdb", fake_duckdb)
        monkeypatch.setattr(boundaries, "_fetch_via_pyclient", fake_pyclient)

        gdf = boundaries.fetch_boundary_via_overture("Anywhere", scale="city")
        assert called["duckdb"] == 1
        assert called["pyclient"] == 0
        assert not gdf.empty

    def test_pyclient_fallback_when_duckdb_missing(self, monkeypatch):
        from utils.fetchers import boundaries
        from utils.fetchers import overture_duckdb as od

        monkeypatch.setattr(od, "is_available", lambda: False)
        called = {"duckdb": 0, "pyclient": 0}

        def fake_duckdb(loc, al, sc):
            called["duckdb"] += 1
            raise AssertionError("duckdb path should not run")

        def fake_pyclient(loc, al, sc):
            called["pyclient"] += 1
            import geopandas as gpd
            from shapely.geometry import box
            return gpd.GeoDataFrame([{"name": loc}], geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

        monkeypatch.setattr(boundaries, "_fetch_via_duckdb", fake_duckdb)
        monkeypatch.setattr(boundaries, "_fetch_via_pyclient", fake_pyclient)

        gdf = boundaries.fetch_boundary_via_overture("Anywhere", scale="city")
        assert called["duckdb"] == 0
        assert called["pyclient"] == 1
        assert not gdf.empty
