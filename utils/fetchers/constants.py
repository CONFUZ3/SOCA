"""Shared constants for the fetchers package."""

from __future__ import annotations


NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HDX_BASE_URL = "https://data.humdata.org/api/3/action"
PHOTON_URL = "https://photon.komoot.io"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Nominatim ToS requires an app-specific UA with valid contact info.
_USER_AGENT = (
    "SOCA-spopt/1.0 (Spatial Optimization Conversational Agent; "
    "academic research; +https://github.com/soca-spopt/soca)"
)

# Overture Maps Place Category Mapping (singular forms, 2024/2025 taxonomy).
OVERTURE_CATEGORIES: dict[str, list[str]] = {
    "health": [
        "hospital", "medical_clinic", "doctor", "pharmacy",
        "medical_center", "health_center", "dentist", "physiotherapist",
        "medical_specialist", "nursing_home",
    ],
    "education": [
        "school",
        "university",
        "college",
        "kindergarten",
        "preschool",
        "college_university",
        "high_school",
    ],
    "food": ["supermarket", "grocery_store", "convenience_store", "market"],
    "finance": ["bank", "atm"],
    "fire_station": ["fire_station"],
    "police": ["police_station"],
    "library": ["library"],
    "transport": ["bus_stop", "train_station", "subway_station", "ferry_terminal", "airport", "transit_stop"],
    "water": ["water_point", "water_well", "water_treatment_plant", "drinking_water"],
    "emergency": ["emergency_shelter", "evacuation_center", "civil_defense"],
}

# OSM (Overpass) tag mapping — paired with OVERTURE_CATEGORIES so the union
# tier (Overture ∪ Overpass) can cover regions where one provider is sparse.
# Each entry is a list of (osm_key, osm_value) pairs queried with `["k"="v"]`.
OSM_AMENITY_TAGS: dict[str, list[tuple[str, str]]] = {
    "health": [
        ("amenity", "hospital"), ("amenity", "clinic"), ("amenity", "doctors"),
        ("amenity", "pharmacy"), ("amenity", "dentist"), ("amenity", "nursing_home"),
        ("healthcare", "hospital"), ("healthcare", "clinic"), ("healthcare", "doctor"),
        ("healthcare", "centre"), ("healthcare", "pharmacy"),
    ],
    "education": [
        ("amenity", "school"), ("amenity", "university"), ("amenity", "college"),
        ("amenity", "kindergarten"),
    ],
    "food": [
        ("shop", "supermarket"), ("shop", "convenience"), ("shop", "grocery"),
        ("amenity", "marketplace"),
    ],
    "finance": [("amenity", "bank"), ("amenity", "atm")],
    "fire_station": [("amenity", "fire_station")],
    "police": [("amenity", "police")],
    "library": [("amenity", "library")],
    "transport": [
        ("highway", "bus_stop"), ("railway", "station"), ("railway", "subway_entrance"),
        ("amenity", "ferry_terminal"), ("aeroway", "aerodrome"),
    ],
    "water": [
        ("amenity", "drinking_water"), ("man_made", "water_well"),
        ("man_made", "water_works"),
    ],
    "emergency": [("amenity", "shelter"), ("emergency", "assembly_point")],
}

# Overture division subtype → approximate OSM admin_level.
_OVERTURE_SUBTYPE_ADMIN_LEVEL: dict[str, int] = {
    "country": 2,
    "region": 4,
    "county": 6,
    "localadmin": 7,
    "locality": 8,
    "neighborhood": 10,
}


_DEFAULT_TOTAL_POPULATION = 100_000

# Retry parameters
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1  # seconds — doubles each retry

# Hard wall-clock for the HDX fetch path (some country files are >300 MB).
_HDX_FETCH_TIMEOUT_SEC = 40

# Hard cap on a single Overture read_all() / DuckDB query call.
_OVERTURE_READ_TIMEOUT_SEC = 90

# Hard cap on a single Overpass POST. Overpass is slow on bboxes >100 km
# across; 90 s leaves room for ~3 retries within the larger fetch budget.
_OVERPASS_QUERY_TIMEOUT_SEC = 90

# Spatial dedup radius (metres) for unioning Overture + Overpass POIs.
# Two points closer than this with similar names are treated as duplicates.
_POI_DEDUP_RADIUS_M = 50.0

# Overture Maps cloud-source base (us-west-2 Amazon S3 public bucket).
_OVERTURE_S3_BASE = "s3://overturemaps-us-west-2/release"

# Pinned fallback release used when STAC discovery fails.
# Format: YYYY-MM-DD.N — see https://stac.overturemaps.org/
_OVERTURE_DEFAULT_RELEASE = "2026-04-15.0"
