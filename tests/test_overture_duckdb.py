"""Integration tests for the DuckDB Overture path.

These hit the public `s3://overturemaps-us-west-2` bucket and are marked
`integration` — skipped when DuckDB is missing or the network is unreachable.
"""

from __future__ import annotations

import os
import time

import pytest

duckdb = pytest.importorskip("duckdb")

from utils.fetchers import overture_duckdb as od
from utils.fetchers.overture_release import get_overture_release


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_net():
    # Tiny probe — if this fails we're offline and the S3 queries will also fail.
    import socket
    try:
        socket.create_connection(("overturemaps-us-west-2.s3.us-west-2.amazonaws.com", 443), timeout=5)
    except OSError:
        pytest.skip("No network to Overture S3 bucket")


def test_release_resolver_returns_valid_format():
    r = get_overture_release()
    # Format: YYYY-MM-DD.N
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}\.\d+$", r), f"unexpected release: {r}"


def test_query_divisions_lima():
    """Query for 'Lima' in Peru returns a locality row and completes <60 s."""
    t = time.time()
    df = od.query_divisions(
        bbox=(-77.5, -12.5, -76.5, -11.5),
        subtypes=["locality", "localadmin", "county", "region"],
        name_query="lima",
    )
    elapsed = time.time() - t
    assert elapsed < 60, f"query_divisions took {elapsed:.1f}s"
    assert not df.empty
    assert "Lima" in df["name"].astype(str).tolist()


def test_query_division_area_by_id_for_lima():
    """The 'Lima' region row (which has a polygon) returns WKB bytes."""
    df = od.query_divisions(
        bbox=(-77.5, -12.5, -76.5, -11.5),
        subtypes=["region"],
        name_query="lima",
    )
    lima_region = df[df["name"].astype(str).str.lower() == "lima"].iloc[0]
    t = time.time()
    area = od.query_division_area_by_id(
        lima_region["id"], bbox=(-77.5, -12.5, -76.5, -11.5)
    )
    elapsed = time.time() - t
    assert elapsed < 60, f"query_division_area_by_id took {elapsed:.1f}s"
    assert not area.empty
    geom_bytes = bytes(area.iloc[0]["geometry"])
    assert geom_bytes[:2] in (b"\x00\x00", b"\x01\x01", b"\x01\x03", b"\x01\x06")


def test_query_places_tiny_bbox():
    df = od.query_places(
        bbox=(14.50, 35.88, 14.53, 35.91),  # Valletta center
        overture_categories=["hospital", "medical_clinic", "pharmacy"],
    )
    # We accept empty result for very small bbox; just assert the call returns.
    assert df is not None
