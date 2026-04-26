"""
Geographic scale classification utilities for SOCA data fetching.

Provides pure-logic helpers (no network calls) for:
- Mapping scale names to OSM admin_level targets
- Mapping scale names to Overture bbox buffer sizes
- Computing demand-point counts from boundary area
- Inferring scale from location strings (heuristic fallback)
- Soft-validating that a fetched boundary matches the declared scale
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SCALES = ("country", "region", "city", "neighborhood")

# Target OSM admin_level integer per scale tier
SCALE_ADMIN_LEVELS: dict[str, int] = {
    "country": 3,
    "region": 5,
    "city": 7,
    "neighborhood": 9,
}

# Supported POI/facility categories.  Mirrors DataFetcher.OVERTURE_CATEGORIES
# and is shared by the ADK fetch tool and the optimisation categoriser.
VALID_POI_CATEGORIES: tuple[str, ...] = (
    "health",
    "education",
    "food",
    "finance",
    "fire_station",
    "police",
    "library",
    "transport",
    "water",
    "emergency",
)

# Bbox expansion buffer in degrees per scale tier.
# Overture queries require a bbox seed from a geocoder; the buffer must be
# large enough to contain the actual polygon at that scale.
SCALE_BBOX_BUFFER: dict[str, float] = {
    "country": 15.0,   # Nigeria spans ~15°, France ~10° — 15° is safe
    "region": 5.0,     # Large states (Texas, Punjab) ~5°
    "city": 2.0,       # Metro areas typically < 2°
    "neighborhood": 0.5,  # Sub-city districts typically < 0.2°
}

# Approximate area thresholds (sq-degrees) used for boundary validation.
# Overlapping ranges are intentional — many places sit at the border.
_SCALE_AREA_THRESHOLDS: dict[str, tuple[float, float]] = {
    "country":      (10.0,   float("inf")),
    "region":       (0.5,    50.0),
    "city":         (0.01,   5.0),
    "neighborhood": (0.0001, 0.5),
}

# Synthetic demand-point density: points = clamp(sqrt(area_km2) * K, MIN, MAX)
_DENSITY_K = 8
_MIN_POINTS = 50
_MAX_POINTS = 2000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_admin_level_for_scale(scale: str) -> int:
    """Return the target OSM admin_level integer for *scale*. Defaults to city (7)."""
    return SCALE_ADMIN_LEVELS.get(scale, 7)


def get_bbox_buffer(scale: str) -> float:
    """Return the bbox expansion buffer in degrees for *scale*. Defaults to 2.0."""
    return SCALE_BBOX_BUFFER.get(scale, 2.0)


def compute_n_points_from_area(area_km2: float) -> int:
    """
    Compute a sensible synthetic demand-point count from boundary area.

    Uses a sqrt-based formula so the count scales gracefully across five
    orders of magnitude without any per-scale lookup table:

        n = clamp(sqrt(area_km2) * 8, 50, 2000)

    Calibration examples:
        Miraflores, Lima  ~9 km²     → 50   (floor)
        Brooklyn, NY      ~183 km²   → 108
        Lima metro        ~2 672 km² → 413
        Nairobi           ~700 km²   → 211
        Nigeria           ~924K km²  → 2000 (ceiling)
    """
    if area_km2 <= 0:
        return _MIN_POINTS
    return max(_MIN_POINTS, min(_MAX_POINTS, int(area_km2 ** 0.5 * _DENSITY_K)))


def heuristic_scale_from_location(location: str) -> str:
    """
    Infer geographic scale from a location string without a network call.

    Used as a fallback when the LLM does not provide a ``scale`` field.

    Rules (applied in order):
    1. Contains region-level keywords → "region"
    2. Contains neighborhood-level keywords → "neighborhood"
    3. Has no comma AND no whitespace (single bare token) → "country"
    4. Default → "city"
    """
    loc_lower = location.strip().lower()

    region_keywords = (
        "province", "state", "region", "department", "governorate",
        "oblast", "prefecture", "canton", "lander", "länder",
    )
    neighborhood_keywords = (
        "district", "ward", "neighbourhood", "neighborhood",
        "barrio", "commune", "arrondissement", "upazila", "quartier",
        "sector", "colonia", "municipio",
    )

    if any(kw in loc_lower for kw in region_keywords):
        return "region"
    if any(kw in loc_lower for kw in neighborhood_keywords):
        return "neighborhood"

    # Single bare token with no comma and no internal space → likely a country name
    stripped = loc_lower.strip()
    if "," not in stripped and " " not in stripped and stripped:
        return "country"

    return "city"


def validate_boundary_scale(boundary_gdf: "gpd.GeoDataFrame", scale: str) -> tuple[bool, str]:
    """
    Soft-check whether the fetched boundary's area is plausible for *scale*.

    Uses geographic degree-area from ``total_bounds`` (rough but dependency-free).
    Never raises — always returns a ``(bool, message)`` tuple.

    Returns:
        (True, "")  — area is plausible for the declared scale.
        (False, hint_str) — area mismatch; *hint_str* describes the discrepancy.
    """
    try:
        bounds = boundary_gdf.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
        area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    except Exception:
        return True, ""  # cannot validate — assume OK

    lo, hi = _SCALE_AREA_THRESHOLDS.get(scale, (0.0, float("inf")))
    if lo <= area_deg2 <= hi:
        return True, ""

    # Find which scale the area actually fits
    for candidate_scale, (c_lo, c_hi) in _SCALE_AREA_THRESHOLDS.items():
        if c_lo <= area_deg2 <= c_hi and candidate_scale != scale:
            return (
                False,
                f"fetched boundary area ({area_deg2:.4f} sq-deg) looks more like "
                f"'{candidate_scale}' than declared scale '{scale}'",
            )

    return (
        False,
        f"fetched boundary area ({area_deg2:.4f} sq-deg) is outside expected range "
        f"for scale '{scale}' ({lo}–{hi} sq-deg)",
    )
