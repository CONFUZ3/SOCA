"""
Thin geocoding layer — name → disambiguated candidates with OSM IDs.

Separate from ``utils.data_fetcher`` because geocoding (query → list of likely
matches with context) is a different concern from boundary polygon fetching
(OSM relation id → polygon). Keeping them separate lets the AOI selector offer
an autocomplete UX without entangling it with the multi-tier polygon chain.

Sources (all free / open data):
  1. Photon (photon.komoot.io)  — BSD, Elasticsearch on OSM, supports prefix
     search which is required for responsive autocomplete.
  2. Nominatim (openstreetmap.org) — ToS-required 1s rate limit; used only as
     a fallback when Photon is unreachable.

Every suggest() call emits an activity_log event so users can see which source
produced the results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

from utils.fetchers.http import NominatimRateLimiter

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from utils.activity_log import log_event, timed

logger = logging.getLogger(__name__)

PHOTON_URL = "https://photon.komoot.io"
NOMINATIM_URL = "https://nominatim.openstreetmap.org"

_USER_AGENT = (
    "SOCA-spopt/1.0 (Spatial Optimization Conversational Agent; "
    "academic research; +https://github.com/soca-spopt/soca)"
)

# Min chars before we hit an API — anything shorter is too ambiguous and wastes
# quota on the free Photon instance.
MIN_QUERY_CHARS = 3

# OSM place_rank → coarse "kind" label for UI. (Lower rank = more prominent.)
_RANK_KIND = [
    (4, "country"),
    (8, "region"),
    (12, "state"),
    (16, "county"),
    (19, "city"),
    (22, "town"),
    (25, "suburb"),
    (30, "neighbourhood"),
]


@dataclass
class GeocodeCandidate:
    """A single disambiguated place match."""

    display_name: str            # "Brooklyn, Kings County, NY, USA"
    short_name: str              # "Brooklyn"
    context: str                 # "Kings County, NY, USA" — greyed in UI
    kind: str                    # "city" | "neighbourhood" | "state" | …
    lat: float
    lon: float
    bbox: Optional[tuple[float, float, float, float]] = None  # minx,miny,maxx,maxy
    osm_type: Optional[str] = None   # "R" (relation, preferred), "W", "N"
    osm_id: Optional[int] = None
    place_rank: int = 30
    country: str = ""
    source: str = "photon"           # which backend produced this

    @property
    def has_relation(self) -> bool:
        """True when a real OSM admin-boundary polygon is fetchable."""
        return self.osm_type == "R" and self.osm_id is not None

    def as_label(self) -> str:
        """Two-line label for the autocomplete dropdown."""
        tail = f" · {self.context}" if self.context else ""
        tag = f" [{self.kind}]" if self.kind else ""
        return f"{self.short_name}{tag}{tail}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suggest(query: str, *, limit: int = 8) -> list[GeocodeCandidate]:
    """Return a ranked list of place candidates for *query*.

    Uses Photon as primary, Nominatim as fallback. Candidates with a usable
    OSM relation (real polygon downstream) float to the top.
    """
    q = (query or "").strip()
    if len(q) < MIN_QUERY_CHARS:
        return []

    candidates: list[GeocodeCandidate] = []
    try:
        candidates = _suggest_photon(q, limit=limit)
    except Exception as exc:
        log_event(
            "geocode.suggest", "fail", f"Photon: {exc}", source="Photon"
        )

    if not candidates:
        try:
            candidates = _suggest_nominatim(q, limit=limit)
        except Exception as exc:
            log_event(
                "geocode.suggest", "fail", f"Nominatim: {exc}", source="Nominatim"
            )

    # Rank: candidates with a relation id beat bbox-only; then by place_rank.
    candidates.sort(key=lambda c: (0 if c.has_relation else 1, c.place_rank))
    return candidates[:limit]


def resolve(candidate_dict: dict) -> GeocodeCandidate:
    """Re-hydrate a GeocodeCandidate from a dict (e.g. stored in session_state)."""
    # Normalise bbox — JSON serialisation turns tuples into lists.
    bbox = candidate_dict.get("bbox")
    if isinstance(bbox, list):
        bbox = tuple(bbox)  # type: ignore[assignment]
    data = dict(candidate_dict)
    data["bbox"] = bbox
    return GeocodeCandidate(**data)


# ---------------------------------------------------------------------------
# Photon backend
# ---------------------------------------------------------------------------


def _suggest_photon(query: str, *, limit: int) -> list[GeocodeCandidate]:
    if not _REQUESTS_AVAILABLE:
        return []

    params = {"q": query, "limit": max(limit, 8), "lang": "en"}
    with timed("geocode.suggest", source="Photon", detail=f'"{query}"') as t:
        resp = requests.get(
            f"{PHOTON_URL}/api",
            params=params,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features") or []
        out: list[GeocodeCandidate] = []
        for feat in features:
            cand = _photon_feature_to_candidate(feat)
            if cand is not None:
                out.append(cand)
        t.detail = f'"{query}" → {len(out)} results'
        return out


def _photon_feature_to_candidate(feat: dict) -> Optional[GeocodeCandidate]:
    p = feat.get("properties") or {}
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates")
    if not coords or len(coords) < 2:
        return None

    lon, lat = float(coords[0]), float(coords[1])

    # Photon uses osm_key/osm_value for the category (e.g. place:city).
    osm_key = p.get("osm_key", "")
    osm_value = p.get("osm_value", "")
    kind = _photon_kind(osm_key, osm_value)

    # Place rank isn't returned by Photon, but the type hints at prominence.
    # Mapping kind → approximate rank so Photon + Nominatim results sort
    # together coherently.
    rank = _kind_to_rank(kind)

    # Build a pleasant "context" string from admin fields.
    context_parts = [
        part
        for part in (
            p.get("county"),
            p.get("state"),
            p.get("country"),
        )
        if part
    ]
    context = ", ".join(context_parts)

    short_name = p.get("name") or p.get("city") or p.get("state") or "(unnamed)"
    display_parts = [short_name] + context_parts
    display_name = ", ".join(display_parts)

    extent = p.get("extent")  # [minx, maxy, maxx, miny] in Photon ordering
    bbox: Optional[tuple[float, float, float, float]] = None
    if extent and len(extent) == 4:
        # Photon returns [west, north, east, south]; normalise to min/max.
        w, n, e, s = extent
        bbox = (min(w, e), min(n, s), max(w, e), max(n, s))

    osm_type = p.get("osm_type")  # "R", "W", "N"
    try:
        osm_id = int(p["osm_id"]) if p.get("osm_id") is not None else None
    except (ValueError, TypeError):
        osm_id = None

    return GeocodeCandidate(
        display_name=display_name,
        short_name=short_name,
        context=context,
        kind=kind,
        lat=lat,
        lon=lon,
        bbox=bbox,
        osm_type=osm_type,
        osm_id=osm_id,
        place_rank=rank,
        country=p.get("country", ""),
        source="photon",
    )


def _photon_kind(osm_key: str, osm_value: str) -> str:
    if osm_key == "place":
        # city, town, village, suburb, neighbourhood, country, state, county…
        return osm_value or "place"
    if osm_key == "boundary" and osm_value == "administrative":
        return "admin_area"
    return osm_value or osm_key or "place"


def _kind_to_rank(kind: str) -> int:
    mapping = {
        "country": 4,
        "state": 8,
        "region": 8,
        "county": 12,
        "city": 16,
        "town": 18,
        "village": 20,
        "suburb": 22,
        "neighbourhood": 25,
        "admin_area": 10,
    }
    return mapping.get(kind, 30)


# ---------------------------------------------------------------------------
# Nominatim backend (fallback)
# ---------------------------------------------------------------------------


_nominatim_limiter = NominatimRateLimiter()


def _suggest_nominatim(query: str, *, limit: int) -> list[GeocodeCandidate]:
    if not _REQUESTS_AVAILABLE:
        return []

    _nominatim_limiter.wait()
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": limit,
        "polygon_geojson": 0,  # we only need bbox/context here
    }
    with timed("geocode.suggest", source="Nominatim", detail=f'"{query}"') as t:
        resp = requests.get(
            f"{NOMINATIM_URL}/search",
            params=params,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json() or []
        out: list[GeocodeCandidate] = []
        for row in data:
            cand = _nominatim_row_to_candidate(row)
            if cand is not None:
                out.append(cand)
        t.detail = f'"{query}" → {len(out)} results'
        return out


def _nominatim_row_to_candidate(row: dict) -> Optional[GeocodeCandidate]:
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
    except (KeyError, ValueError, TypeError):
        return None

    addr = row.get("address") or {}
    short_name = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("suburb")
        or addr.get("state")
        or addr.get("country")
        or row.get("name")
        or row.get("display_name", "").split(",")[0]
    )

    ctx_parts = [
        part
        for part in (
            addr.get("county"),
            addr.get("state"),
            addr.get("country"),
        )
        if part
    ]
    context = ", ".join(ctx_parts)

    # Nominatim's boundingbox is [south, north, west, east] as strings.
    bbox_raw = row.get("boundingbox")
    bbox: Optional[tuple[float, float, float, float]] = None
    if bbox_raw and len(bbox_raw) == 4:
        try:
            s, n, w, e = map(float, bbox_raw)
            bbox = (w, s, e, n)
        except (ValueError, TypeError):
            bbox = None

    osm_type_full = row.get("osm_type", "")
    osm_type = {"relation": "R", "way": "W", "node": "N"}.get(osm_type_full)
    try:
        osm_id = int(row["osm_id"]) if row.get("osm_id") is not None else None
    except (ValueError, TypeError):
        osm_id = None

    try:
        rank = int(row.get("place_rank", 30))
    except (ValueError, TypeError):
        rank = 30

    kind = row.get("type") or row.get("class") or "place"

    return GeocodeCandidate(
        display_name=row.get("display_name", short_name),
        short_name=short_name,
        context=context,
        kind=kind,
        lat=lat,
        lon=lon,
        bbox=bbox,
        osm_type=osm_type,
        osm_id=osm_id,
        place_rank=rank,
        country=addr.get("country", ""),
        source="nominatim",
    )


# ---------------------------------------------------------------------------
# Streamlit caching wrapper (optional — import-guarded so tests work headless)
# ---------------------------------------------------------------------------


def _install_streamlit_cache() -> None:
    """Wrap suggest() with st.cache_data so repeated prefixes don't re-query."""
    global suggest
    try:
        import streamlit as st
    except ImportError:
        return

    uncached = suggest

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached(query: str, limit: int = 8) -> list[dict]:
        # Cache serialisable dicts, not dataclasses, so Streamlit's hasher is happy.
        return [asdict(c) for c in uncached(query, limit=limit)]

    def wrapper(query: str, *, limit: int = 8) -> list[GeocodeCandidate]:
        dicts = _cached(query, limit)
        return [resolve(d) for d in dicts]

    suggest = wrapper  # type: ignore[assignment]


_install_streamlit_cache()
