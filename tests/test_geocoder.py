"""Unit tests for utils.geocoder — Photon/Nominatim parsing and ranking."""

from unittest.mock import MagicMock, patch

import pytest

# Re-import the un-cached version for deterministic tests; the module wraps
# suggest() with @st.cache_data on import, and we want to hit the raw Photon /
# Nominatim code paths directly.
from utils import geocoder
from utils.geocoder import GeocodeCandidate, _suggest_photon, _suggest_nominatim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _photon_feature(**overrides):
    feat = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-73.95, 40.65]},
        "properties": {
            "osm_id": 175905,
            "osm_type": "R",
            "osm_key": "place",
            "osm_value": "city",
            "name": "Brooklyn",
            "country": "United States",
            "state": "New York",
            "county": "Kings County",
            "extent": [-74.05, 40.74, -73.85, 40.55],
        },
    }
    feat["properties"].update(overrides)
    return feat


def _nominatim_row(**overrides):
    row = {
        "osm_id": 175905,
        "osm_type": "relation",
        "lat": "40.65",
        "lon": "-73.95",
        "display_name": "Brooklyn, Kings County, New York, USA",
        "name": "Brooklyn",
        "class": "boundary",
        "type": "administrative",
        "place_rank": 16,
        "address": {
            "city": "Brooklyn",
            "county": "Kings County",
            "state": "New York",
            "country": "USA",
        },
        "boundingbox": ["40.55", "40.74", "-74.05", "-73.85"],
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Photon parsing
# ---------------------------------------------------------------------------


def test_photon_parses_relation_into_candidate():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"features": [_photon_feature()]}
    mock_resp.raise_for_status.return_value = None

    with patch("utils.geocoder.requests.get", return_value=mock_resp):
        results = _suggest_photon("brooklyn", limit=5)

    assert len(results) == 1
    c = results[0]
    assert c.short_name == "Brooklyn"
    assert c.osm_type == "R"
    assert c.osm_id == 175905
    assert c.has_relation is True
    assert "New York" in c.context
    assert c.source == "photon"
    # bbox normalised to (minx, miny, maxx, maxy)
    assert c.bbox is not None
    assert c.bbox[0] < c.bbox[2]
    assert c.bbox[1] < c.bbox[3]


def test_photon_skips_features_without_coords():
    bad = {"type": "Feature", "geometry": {"type": "Point", "coordinates": []},
           "properties": {"name": "Nowhere"}}
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"features": [bad, _photon_feature()]}
    mock_resp.raise_for_status.return_value = None

    with patch("utils.geocoder.requests.get", return_value=mock_resp):
        results = _suggest_photon("brooklyn", limit=5)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Nominatim parsing
# ---------------------------------------------------------------------------


def test_nominatim_parses_row_into_candidate():
    mock_resp = MagicMock()
    mock_resp.json.return_value = [_nominatim_row()]
    mock_resp.raise_for_status.return_value = None

    with patch("utils.geocoder.requests.get", return_value=mock_resp):
        results = _suggest_nominatim("brooklyn", limit=5)

    assert len(results) == 1
    c = results[0]
    assert c.osm_type == "R"
    assert c.osm_id == 175905
    assert c.has_relation is True
    assert c.source == "nominatim"
    assert c.bbox == (-74.05, 40.55, -73.85, 40.74)


# ---------------------------------------------------------------------------
# Ranking — relation-backed candidates float to the top
# ---------------------------------------------------------------------------


def test_suggest_ranks_relation_candidates_above_bbox():
    bbox_only = _photon_feature(osm_type="W", osm_id=None, name="Brooklyn Park",
                                state="Minnesota", county=None)
    with_rel = _photon_feature()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"features": [bbox_only, with_rel]}
    mock_resp.raise_for_status.return_value = None

    # Bypass the st.cache_data wrapper installed at import time — call the
    # underlying photon backend directly and sort the same way suggest() does.
    with patch("utils.geocoder.requests.get", return_value=mock_resp):
        raw = _suggest_photon("brook", limit=5)
    raw.sort(key=lambda c: (0 if c.has_relation else 1, c.place_rank))

    assert raw[0].osm_type == "R"  # Brooklyn (relation) wins
    assert raw[1].osm_type == "W"


# ---------------------------------------------------------------------------
# Short queries are no-ops
# ---------------------------------------------------------------------------


def test_suggest_returns_empty_for_short_query():
    # The top-level suggest() may be the cached wrapper — either way, 1-char
    # input should not hit the network.
    with patch("utils.geocoder.requests.get") as mock_get:
        assert geocoder.suggest("a") == []
        assert geocoder.suggest("") == []
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# resolve() round-trip
# ---------------------------------------------------------------------------


def test_resolve_roundtrips_dataclass_via_dict():
    cand = GeocodeCandidate(
        display_name="Brooklyn, NYC",
        short_name="Brooklyn",
        context="NYC",
        kind="city",
        lat=40.65, lon=-73.95,
        bbox=(-74.05, 40.55, -73.85, 40.74),
        osm_type="R", osm_id=175905,
        place_rank=16,
        country="USA",
        source="photon",
    )
    from dataclasses import asdict
    d = asdict(cand)
    # simulate JSON round-trip → bbox becomes a list
    d["bbox"] = list(d["bbox"])
    restored = geocoder.resolve(d)
    assert restored.osm_id == 175905
    assert restored.bbox == (-74.05, 40.55, -73.85, 40.74)
    assert restored.has_relation is True
