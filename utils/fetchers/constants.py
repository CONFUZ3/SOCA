"""Shared constants for the fetchers package."""

from __future__ import annotations


NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HDX_BASE_URL = "https://data.humdata.org/api/3/action"
PHOTON_URL = "https://photon.komoot.io"

# Nominatim ToS requires an app-specific UA with valid contact info.
_USER_AGENT = (
    "SOCA-spopt/1.0 (Spatial Optimization Conversational Agent; "
    "academic research; +https://github.com/soca-spopt/soca)"
)

# Overture Maps Place Category Mapping (singular forms, 2024/2025 taxonomy).
OVERTURE_CATEGORIES: dict[str, list[str]] = {
    "health": ["hospital", "medical_clinic", "doctor", "pharmacy", "medical_center", "health_center"],
    "education": ["school", "university", "college", "kindergarten", "preschool"],
    "food": ["supermarket", "grocery_store", "convenience_store", "market"],
    "finance": ["bank", "atm"],
    "fire_station": ["fire_station"],
    "police": ["police_station"],
    "library": ["library"],
    "transport": ["bus_stop", "train_station", "subway_station", "ferry_terminal", "airport", "transit_stop"],
    "water": ["water_point", "water_well", "water_treatment_plant", "drinking_water"],
    "emergency": ["emergency_shelter", "evacuation_center", "civil_defense"],
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

# Overture Maps cloud-source base (us-west-2 Amazon S3 public bucket).
_OVERTURE_S3_BASE = "s3://overturemaps-us-west-2/release"

# Pinned fallback release used when STAC discovery fails.
# Format: YYYY-MM-DD.N — see https://stac.overturemaps.org/
_OVERTURE_DEFAULT_RELEASE = "2026-04-15.0"
