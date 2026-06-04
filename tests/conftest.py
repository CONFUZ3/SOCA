"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_shared_network_caches():
    """Clear the process-wide NetworkManager shared caches between tests.

    ``utils.network_manager`` keeps module-level fetch locks and a result
    cache keyed on AOI polygon hashes so that a background prefetch and a
    solve-time fetch coalesce into a single Overpass download. Across
    test functions that same sharing produces ghost cache hits: test B
    sees test A's fake graph and skips its own fetch. Reset before each
    test to keep them independent.
    """
    try:
        from utils import network_manager as _nm
    except Exception:
        yield
        return

    with _nm._SHARED_REGISTRY_LOCK:
        _nm._SHARED_FETCH_LOCKS.clear()
        _nm._SHARED_FETCH_RESULTS.clear()
    yield
    with _nm._SHARED_REGISTRY_LOCK:
        _nm._SHARED_FETCH_LOCKS.clear()
        _nm._SHARED_FETCH_RESULTS.clear()
