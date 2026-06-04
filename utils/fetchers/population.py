"""Population-grid fetchers.

Tier order: Kontur (HDX) → synthetic uniform grid.

Exports the pure ``_resolve_population_column`` and ``_resolve_latlon_columns``
helpers that make HDX column handling robust across the many real-world
variants that HDX resources ship with.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import re
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .constants import (
    PHOTON_URL,
    _DEFAULT_TOTAL_POPULATION,
    _HDX_FETCH_TIMEOUT_SEC,
)
from .errors import PopulationDataError
from .http import make_request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-resolution helpers (pure — unit-tested in tests/)
# ---------------------------------------------------------------------------

# Columns that *look* like population but are actually string metadata.
_NON_POP_SUFFIXES = (
    "_method", "_source", "_year", "_note", "_comment", "_type", "_class",
    "_label", "_flag", "_quality", "_version",
)

_EXACT_POP_NAMES = (
    "population", "pop", "persons", "people", "pop_count",
    "residents", "popcount", "pop_total", "population_total",
)

_LAT_CANDIDATES = (
    "latitude", "lat", "y", "ycoord", "y_coord", "lat_deg", "lat_dd",
    "latitud", "latitude_deg",
)
_LON_CANDIDATES = (
    "longitude", "lon", "lng", "long", "x", "xcoord", "x_coord",
    "lon_deg", "lon_dd", "longitud", "longitude_deg",
)


def _normalise(name: str) -> str:
    """Lowercase, replace non-alphanumerics with ``_``, collapse repeats."""
    n = re.sub(r"[^a-z0-9]+", "_", str(name).lower().strip())
    n = re.sub(r"_+", "_", n).strip("_")
    return n


def _year_from_name(name: str) -> int:
    """Extract a 4-digit year from a column name (e.g. 'population_2020').

    Returns 0 if none found. Used to prefer the most recent year when multiple
    ``population_YYYY`` columns are present.
    """
    m = re.search(r"(19|20)\d{2}", str(name))
    return int(m.group(0)) if m else 0


def _is_numeric_after_coerce(series: pd.Series, max_na_frac: float = 0.05) -> bool:
    """True if >=(1-max_na_frac) of values parse to numeric via to_numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    try:
        coerced = pd.to_numeric(series, errors="coerce")
    except Exception:
        return False
    if len(coerced) == 0:
        return False
    na_frac = float(coerced.isna().mean())
    return na_frac <= max_na_frac


def _resolve_population_column(
    df: pd.DataFrame,
    *,
    max_na_frac: float = 0.05,
) -> Optional[str]:
    """Pick the best population-valued numeric column from *df*.

    Priority order:
      1. Exact name match (normalised) from ``_EXACT_POP_NAMES``.
      2. Year-suffixed: ``population_YYYY`` / ``pop_YYYY`` / ``pop_YYYY_n`` —
         pick the one with the *most recent year*.
      3. Prefix match: starts with ``population_`` or ``pop_``.
      4. Contains ``population`` or ``persons`` (fuzzy).

    Columns whose normalised name ends with a known non-population suffix
    (``_method``, ``_source``, ``_year``, …) are rejected outright because
    they're metadata, not counts. Any candidate that does not pass the
    numeric-dtype gate (or survive ``to_numeric`` coercion with <5% NaN) is
    also rejected.

    Returns the original column label (not the normalised form), or ``None``
    when nothing plausible was found. Callers should log the column list and
    either fail the tier or move on — *never* blindly pick ``columns[-1]``.
    """
    if df is None or df.empty:
        return None

    # Build (original, normalised) pairs.
    normalised = {col: _normalise(col) for col in df.columns}

    def _usable(original: str) -> bool:
        n = normalised[original]
        if not n:
            return False
        if n.endswith(_NON_POP_SUFFIXES):
            return False
        return _is_numeric_after_coerce(df[original], max_na_frac=max_na_frac)

    # Tier 1: exact matches.
    for orig, norm in normalised.items():
        if norm in _EXACT_POP_NAMES and _usable(orig):
            return orig

    # Tier 2: year-suffixed (pop_YYYY, population_YYYY, pop_YYYY_n).
    year_candidates = []
    for orig, norm in normalised.items():
        if norm.startswith(("population_", "pop_")) and _year_from_name(norm) > 0:
            if _usable(orig):
                year_candidates.append((orig, _year_from_name(norm)))
    if year_candidates:
        # Most recent year wins; ties broken by shorter name (simpler column).
        year_candidates.sort(key=lambda x: (-x[1], len(normalised[x[0]])))
        return year_candidates[0][0]

    # Tier 3: prefix match without a year.
    for orig, norm in normalised.items():
        if norm.startswith(("population_", "pop_")) and _usable(orig):
            return orig

    # Tier 4: fuzzy contains.
    for orig, norm in normalised.items():
        if ("population" in norm or "persons" in norm) and _usable(orig):
            return orig

    return None


def _resolve_latlon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Return (lat_col, lon_col) original labels, or (None, None) if not found."""
    normalised = {col: _normalise(col) for col in df.columns}

    lat_col = None
    lon_col = None
    for orig, norm in normalised.items():
        if lat_col is None and norm in _LAT_CANDIDATES:
            lat_col = orig
        if lon_col is None and norm in _LON_CANDIDATES:
            lon_col = orig
    return lat_col, lon_col


def _sanitise_latlon(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> tuple[pd.DataFrame, str, str]:
    """Coerce lat/lon to numeric, drop NaN, and auto-swap if a majority of
    rows are out of range (a common "columns labelled wrong" data issue).

    Returns (df_with_coerced_cols, lat_col, lon_col). May raise
    ``PopulationDataError`` if the data cannot be made plausible.
    """
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    if df.empty:
        raise PopulationDataError(
            f"HDX CSV: lat/lon columns '{lat_col}'/'{lon_col}' have no numeric values."
        )

    lat_in_range = df[lat_col].between(-90, 90)
    lon_in_range = df[lon_col].between(-180, 180)
    ok_frac = float((lat_in_range & lon_in_range).mean())

    if ok_frac >= 0.95:
        return df, lat_col, lon_col

    # Try swapping: maybe lat/lon are reversed in the source.
    lat_swapped_in_range = df[lon_col].between(-90, 90)
    lon_swapped_in_range = df[lat_col].between(-180, 180)
    swapped_frac = float((lat_swapped_in_range & lon_swapped_in_range).mean())

    if swapped_frac > ok_frac and swapped_frac >= 0.95:
        logger.warning(
            f"HDX CSV: lat/lon columns appear swapped — using "
            f"'{lon_col}' as lat and '{lat_col}' as lon."
        )
        return df, lon_col, lat_col

    raise PopulationDataError(
        f"HDX CSV: lat/lon columns '{lat_col}'/'{lon_col}' out of WGS-84 range "
        f"({ok_frac * 100:.0f}% valid, {swapped_frac * 100:.0f}% when swapped)."
    )


# ---------------------------------------------------------------------------
# Tier: synthetic grid (last-resort)
# ---------------------------------------------------------------------------

def generate_synthetic_population(
    boundary_gdf: gpd.GeoDataFrame,
    n_points: int,
    *,
    total_population: Optional[int] = None,
    random_seed: int = 42,
) -> gpd.GeoDataFrame:
    """Rejection-sampled uniform grid strictly inside the boundary polygon."""
    try:
        boundary_union = unary_union(
            boundary_gdf.to_crs("EPSG:4326").geometry.values
        )
    except Exception as exc:
        raise PopulationDataError(
            f"Cannot compute boundary union for synthetic grid: {exc}"
        ) from exc

    if boundary_union is None or boundary_union.is_empty:
        raise PopulationDataError("Boundary geometry is empty.")

    bounds = boundary_union.bounds
    bbox_geom = box(*bounds)
    is_plain_bbox = boundary_union.equals(bbox_geom) or (
        abs(boundary_union.area - bbox_geom.area) / max(bbox_geom.area, 1e-10) < 1e-6
    )
    if is_plain_bbox:
        logger.warning(
            "Boundary polygon is a plain bounding-box rectangle; synthetic "
            "points may fall in water."
        )

    minx, miny, maxx, maxy = bounds
    points: list[Point] = []
    MAX_ATTEMPTS = n_points * 50
    attempts = 0
    rng = random.Random(random_seed)

    while len(points) < n_points and attempts < MAX_ATTEMPTS:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        pt = Point(x, y)
        if boundary_union.contains(pt):
            points.append(pt)
        attempts += 1

    if not points:
        raise PopulationDataError(
            "Could not generate any population points within boundary "
            "(polygon may be too small or degenerate)."
        )

    if len(points) < n_points:
        logger.warning(
            f"Only generated {len(points)}/{n_points} synthetic points "
            f"after {MAX_ATTEMPTS} attempts."
        )

    total = total_population if total_population and total_population > 0 else _DEFAULT_TOTAL_POPULATION
    pop_per_point = total / len(points)

    gdf = gpd.GeoDataFrame(
        {"population": [pop_per_point] * len(points)},
        geometry=points,
        crs="EPSG:4326",
    )
    gdf["data_source"] = "synthetic_uniform_grid"
    return gdf


def estimate_total_population(boundary_gdf: gpd.GeoDataFrame) -> int:
    """Best-effort total-population estimate for a boundary.

    Sources tried in order:
      1. ``population`` column on boundary (from Overture/Nominatim metadata).
      2. Area × 250 people/km² rough global average.
      3. _DEFAULT_TOTAL_POPULATION.
    """
    if "population" in boundary_gdf.columns:
        raw = str(boundary_gdf["population"].iloc[0] or "").replace(",", "").strip()
        try:
            pop = int(float(raw))
            if pop > 0:
                return pop
        except (ValueError, TypeError):
            pass
    try:
        area_km2 = boundary_gdf.to_crs("EPSG:6933").geometry.area.sum() / 1e6
        estimated = int(area_km2 * 250)
        if estimated > 0:
            return estimated
    except Exception:
        pass
    return _DEFAULT_TOTAL_POPULATION


# ---------------------------------------------------------------------------
# Tier: HDX / Kontur
# ---------------------------------------------------------------------------

_SKIP_KEYWORDS = (
    "children", "elderly", "youth", "under_five", "reproductive",
    "15_24", "60_plus", "indicator", "health-statistics", "statistics",
    "indicators", "who-",
)

_SPATIAL_POP_KEYWORDS = (
    "kontur", "h3 hexagon", "high resolution population",
    "hrsl", "worldpop", "population density",
)


def _resource_size_mb(r) -> int:
    try:
        return int((r.get("size") or 0)) // 1024 // 1024
    except Exception:
        return 0


def _pick_resource(ds):
    """Return (best_resource, kind) for a Kontur dataset. Prefers newest GPKG."""
    try:
        resources = ds.get_resources()
    except Exception as exc:
        logger.warning(f"HDX: ds.get_resources() raised: {exc}")
        return None, None

    gpkg_matches = []
    csv_matches = []
    for r in resources:
        try:
            fmt = (r.get_format() or "").lower()
            name = (r.get("name") or "").lower()
        except Exception:
            continue
        if any(kw in name for kw in _SKIP_KEYWORDS):
            continue
        if "gpkg" in fmt or "geopackage" in fmt or name.endswith((".gpkg", ".gpkg.gz")):
            gpkg_matches.append(r)
        elif "csv" in fmt or name.endswith((".csv", ".csv.zip")):
            if "part_1" in name or "part_2" in name:
                logger.debug(f"HDX: skipping multi-part CSV resource '{name}'")
                continue
            csv_matches.append(r)

    def _newest(rs):
        return sorted(rs, key=lambda r: r.get("name") or "", reverse=True)[0] if rs else None

    r = _newest(gpkg_matches)
    if r is not None:
        return r, "gpkg"
    r = _newest(csv_matches)
    if r is not None:
        return r, "csv"
    return None, None


def _dataset_matches_country(ds, iso3_upper: str) -> bool:
    try:
        for g in ds.get("groups") or []:
            g_name = (g.get("name") or "").lower().strip()
            if g_name and g_name.upper() == iso3_upper:
                return True
    except Exception:
        pass
    return False


_ISO3_ALIAS_MAP = {
    # Common names/aliases that hdx-python-country's fuzzy matcher mishandles
    # or that are ambiguous (e.g. "Congo" matches COG not COD). Add here when
    # we find real-world inputs that slip through.
    "drc": "COD",
    "dr congo": "COD",
    "dr-congo": "COD",
    "democratic republic of congo": "COD",
    "democratic republic of the congo": "COD",
    "congo kinshasa": "COD",
    "congo-kinshasa": "COD",
    "congo brazzaville": "COG",
    "congo-brazzaville": "COG",
    "republic of congo": "COG",
    "republic of the congo": "COG",
    "uk": "GBR",
    "u.k.": "GBR",
    "britain": "GBR",
    "great britain": "GBR",
    "england": "GBR",
    "scotland": "GBR",
    "wales": "GBR",
    "northern ireland": "GBR",
    "usa": "USA",
    "u.s.": "USA",
    "u.s.a.": "USA",
    "us": "USA",
    "united states": "USA",
    "america": "USA",
    "russia": "RUS",
    "south korea": "KOR",
    "korea south": "KOR",
    "republic of korea": "KOR",
    "north korea": "PRK",
    "korea north": "PRK",
    "dprk": "PRK",
    "vietnam": "VNM",
    "viet nam": "VNM",
    "iran": "IRN",
    "syria": "SYR",
    "laos": "LAO",
    "taiwan": "TWN",
    "burma": "MMR",
    "myanmar": "MMR",
    "ivory coast": "CIV",
    "cote d'ivoire": "CIV",
    "cote divoire": "CIV",
    "czech republic": "CZE",
    "czechia": "CZE",
    "macedonia": "MKD",
    "north macedonia": "MKD",
    "palestine": "PSE",
    "west bank": "PSE",
    "gaza": "PSE",
    "east timor": "TLS",
    "timor leste": "TLS",
    "cape verde": "CPV",
    "swaziland": "SWZ",
    "eswatini": "SWZ",
    "holy see": "VAT",
    "vatican": "VAT",
    "bolivia": "BOL",
    "venezuela": "VEN",
    "tanzania": "TZA",
    "moldova": "MDA",
    "brunei": "BRN",
    "micronesia": "FSM",
}


# Process-local cache for centroid→ISO3 lookups so retries inside a session
# don't re-hit Photon/Nominatim/country.is. Key is a rounded centroid tuple.
_ISO3_REVGEO_CACHE: dict[tuple, tuple[Optional[str], str]] = {}
_ISO3_CACHE_MAXSIZE = 256


def _iso3_cache_set(key, value) -> None:
    if len(_ISO3_REVGEO_CACHE) >= _ISO3_CACHE_MAXSIZE:
        _ISO3_REVGEO_CACHE.pop(next(iter(_ISO3_REVGEO_CACHE)))
    _ISO3_REVGEO_CACHE[key] = value


def _resolve_iso3(boundary_4326: gpd.GeoDataFrame, centroid) -> tuple[Optional[str], str]:
    """Resolve ISO3 country code from boundary metadata or reverse geocode.

    Strategy (first hit wins):
      1. Direct ISO-code columns (``country_code``, ``iso3``, ``iso_a3``,
         ``ISO3166-1``, ``ISO3166-1:alpha3``, ``ISO3166-1:alpha2``).
      2. Country-name columns (``country``, ``country_name``, ``addr:country``).
      3. Last comma-separated token of ``location_query`` / ``display_name``.
      4. Reverse geocode the centroid via Photon, then Nominatim, then
         country.is as a lightweight offline-ish fallback.
    """
    try:
        from hdx.location.country import Country
    except ImportError:
        return None, ""

    def _try_iso(raw: str) -> Optional[str]:
        raw = (raw or "").strip()
        if not raw:
            return None
        # Hand-rolled alias map first — catches ambiguous/common short forms
        # before the fuzzy matcher gets a chance to pick the wrong country
        # (e.g. "Congo" → COG rather than COD).
        lowered = raw.lower().strip().replace(".", "").strip()
        if lowered in _ISO3_ALIAS_MAP:
            return _ISO3_ALIAS_MAP[lowered]
        if len(raw) == 2 and raw.isalpha():
            try:
                code = Country.get_iso3_from_iso2(raw.upper())
                if code:
                    return code
            except Exception:
                pass
        if len(raw) == 3 and raw.isalpha():
            try:
                info = Country.get_country_info_from_iso3(raw.upper())
                if info:
                    return raw.upper()
            except Exception:
                pass
        # Fuzzy match. hdx-python-country returns (iso3, exact_match_bool);
        # exact_match=False still yields a valid code for cases like
        # "Vietnam"→VNM, "Britain"→GBR, "Taiwan"→TWN. Do NOT require the
        # exact-match flag — that was the old bug that produced "ISO3 not
        # found" for dozens of legitimate inputs.
        try:
            code, _exact = Country.get_iso3_country_code_fuzzy(raw)
            if code:
                return code
        except Exception:
            pass
        return None

    display = ""

    # 1. Direct ISO code columns (OSM uses colons in tag names, so match
    #    case-insensitively and tolerate variants).
    direct_code_cols = {
        "country_code", "iso3", "iso_a3", "iso_3166_1_alpha_3",
        "iso3166_1_alpha3", "iso_3166_1_alpha3", "iso_3166_1",
        "iso3166_1", "cc", "cca3", "cca2",
    }
    for col in boundary_4326.columns:
        key = _normalise(col)
        if key in direct_code_cols:
            val = str(boundary_4326[col].iloc[0] or "").strip()
            iso3 = _try_iso(val)
            if iso3:
                return iso3, val

    # 2. Name columns.
    name_cols = {"country", "country_name", "addr_country", "nation"}
    for col in boundary_4326.columns:
        key = _normalise(col)
        if key in name_cols:
            val = str(boundary_4326[col].iloc[0] or "").strip()
            iso3 = _try_iso(val)
            if iso3:
                return iso3, val

    # 3. ``location_query`` / ``display_name`` — last comma-separated token
    #    usually holds the country.
    for col in ("location_query", "display_name", "label"):
        if col in boundary_4326.columns:
            raw_val = str(boundary_4326[col].iloc[0] or "")
            parts = [p.strip() for p in raw_val.split(",") if p.strip()]
            # Try last, second-last (handles "City, State, Country" and
            # "Country, Region" formats).
            for cand in reversed(parts[-3:]):
                iso3 = _try_iso(cand)
                if iso3:
                    return iso3, cand

    # 4. Reverse-geocode the centroid. Round to 0.25° so repeated retries on
    #    the same AOI reuse the cached result (ISO3 boundaries don't change
    #    at that resolution).
    cache_key = (round(float(centroid.x) * 4) / 4, round(float(centroid.y) * 4) / 4)
    if cache_key in _ISO3_REVGEO_CACHE:
        return _ISO3_REVGEO_CACHE[cache_key]

    # 4a. Photon.
    try:
        rev_resp = make_request(
            PHOTON_URL + "/reverse",
            params={"lat": centroid.y, "lon": centroid.x, "lang": "en"},
            timeout=20,
        )
        rev_data = rev_resp.json()
        features = rev_data.get("features", [])
        if features:
            feat = features[0]
            feat_coords = feat.get("geometry", {}).get("coordinates", [])
            if feat_coords and len(feat_coords) >= 2:
                feat_lon, feat_lat = float(feat_coords[0]), float(feat_coords[1])
                if abs(feat_lon - centroid.x) < 15 and abs(feat_lat - centroid.y) < 15:
                    p = feat.get("properties", {}) or {}
                    countrycode = (p.get("countrycode") or "").strip().upper()
                    country_en = (p.get("country") or "").strip()
                    iso3 = _try_iso(countrycode) or _try_iso(country_en)
                    if iso3:
                        result = (iso3, country_en or countrycode)
                        _iso3_cache_set(cache_key, result)
                        return result
                else:
                    logger.warning(
                        f"Photon reverse returned a location far from centroid "
                        f"({feat_lat:.2f},{feat_lon:.2f} vs "
                        f"{centroid.y:.2f},{centroid.x:.2f}); ignoring."
                    )
    except Exception as e:
        logger.debug(f"Photon reverse geocode for ISO3 failed: {e}")

    # 4b. Nominatim — denser admin-boundary coverage than Photon for coastal
    #    / disputed areas.
    try:
        from .http import nominatim_get
        rev_resp = nominatim_get(
            "/reverse",
            params={
                "lat": centroid.y,
                "lon": centroid.x,
                "zoom": 3,
                "format": "json",
            },
            timeout=20,
        )
        rev_data = rev_resp.json() or {}
        addr = rev_data.get("address", {}) or {}
        cc = (addr.get("country_code") or "").strip().upper()
        cname = (addr.get("country") or "").strip()
        iso3 = _try_iso(cc) or _try_iso(cname)
        if iso3:
            result = (iso3, cname or cc)
            _iso3_cache_set(cache_key, result)
            return result
    except Exception as e:
        logger.debug(f"Nominatim reverse geocode for ISO3 failed: {e}")

    # 4c. country.is — a tiny lat/lon → ISO2 service with a MaxMind dataset
    #    that covers essentially all land points (and oceans with the nearest
    #    sovereign territory). Last-resort when both OSM-backed providers
    #    silently return empty results.
    try:
        cis_resp = make_request(
            f"https://api.country.is/{centroid.y},{centroid.x}",
            timeout=10,
        )
        cis_data = cis_resp.json() or {}
        iso2 = (cis_data.get("country") or "").strip().upper()
        iso3 = _try_iso(iso2)
        if iso3:
            result = (iso3, iso2)
            _iso3_cache_set(cache_key, result)
            return result
    except Exception as e:
        logger.debug(f"country.is reverse geocode for ISO3 failed: {e}")

    # Last-ditch: log every field we tried so fallback diagnosis is possible,
    # and surface a user-visible event so the sidebar shows the silent
    # synthetic-data fallback.
    tried_fields = [
        c for c in ("country_code", "iso3", "iso_a3", "country",
                    "country_name", "location_query")
        if c in boundary_4326.columns
    ]
    logger.warning(
        "ISO3 resolution failed (tried fields=%s, centroid=%.3f,%.3f)",
        tried_fields, centroid.y, centroid.x,
    )
    try:
        from utils.activity_log import log_event
    except Exception:
        return None, display
    log_event(
        "population.fetch",
        "warn",
        f"ISO3 unresolved (tried {tried_fields or 'no metadata'}) "
        "→ falling back to synthetic grid",
        source="HDX",
    )
    return None, display


def _read_hdx_csv(
    path_str: str,
    boundary_union,
    bounds: tuple,
    *,
    cleanup: bool,
) -> Optional[gpd.GeoDataFrame]:
    """Read an HDX CSV (or zipped CSV) and convert population rows to Points."""
    import zipfile

    try:
        if path_str.endswith(".zip") or zipfile.is_zipfile(path_str):
            df = pd.read_csv(path_str, compression="zip")
        else:
            df = pd.read_csv(path_str)
    except Exception as e:
        logger.error(f"Failed to read HDX CSV: {e}")
        return None
    finally:
        if cleanup and os.path.exists(path_str):
            try:
                os.unlink(path_str)
            except Exception:
                pass

    if df.empty:
        logger.warning("HDX CSV is empty.")
        return None

    lat_col, lon_col = _resolve_latlon_columns(df)
    if not lat_col or not lon_col:
        logger.error(
            f"HDX CSV missing lat/lon columns. Got: {list(df.columns)[:20]}"
        )
        return None

    try:
        df, lat_col, lon_col = _sanitise_latlon(df, lat_col, lon_col)
    except PopulationDataError as exc:
        logger.error(str(exc))
        return None

    minx, miny, maxx, maxy = bounds
    df_filtered = df[
        (df[lon_col] >= minx) & (df[lon_col] <= maxx) &
        (df[lat_col] >= miny) & (df[lat_col] <= maxy)
    ].copy()
    if df_filtered.empty:
        logger.warning("HDX CSV has no rows within boundary bbox.")
        return None

    pop_col = _resolve_population_column(df_filtered)
    if pop_col is None:
        logger.error(
            "HDX CSV: no usable numeric population column found. "
            f"Columns: {list(df_filtered.columns)[:30]}"
        )
        return None

    pop_values = pd.to_numeric(df_filtered[pop_col], errors="coerce")

    rows = []
    for lat, lon, pop in zip(
        df_filtered[lat_col].values,
        df_filtered[lon_col].values,
        pop_values.values,
    ):
        if pd.isna(pop) or pop <= 0:
            continue
        try:
            pt = Point(float(lon), float(lat))
        except (TypeError, ValueError):
            continue
        if boundary_union.contains(pt):
            rows.append({"population": float(pop), "geometry": pt})

    if not rows:
        logger.warning("HDX CSV: no points fell inside boundary polygon.")
        return None

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf["data_source"] = "hdx_facebook_population"
    logger.info(
        f"HDX CSV: loaded {len(gdf)} points using lat='{lat_col}', "
        f"lon='{lon_col}', pop='{pop_col}'."
    )
    return gdf


def _read_hdx_gpkg(
    path_str: str,
    boundary_union,
    bounds: tuple,
    *,
    is_cached: bool,
) -> Optional[gpd.GeoDataFrame]:
    """Read an HDX GeoPackage (Kontur H3 hexagons). Handles gzip-wrapped files.

    Temp-file cleanup runs in ``finally`` regardless of whether the gzip-detect
    step raises — previously, extra_files were only tracked after the try, so
    a failure during detection left temp files on disk.
    """
    import gzip as _gzip
    import shutil

    gpkg_path = path_str
    extra_files: set[str] = set()

    try:
        try:
            with open(path_str, "rb") as _f:
                is_gzip = _f.read(2) == b"\x1f\x8b"
            if is_gzip:
                gpkg_path = path_str + "_decompressed.gpkg"
                extra_files.add(gpkg_path)
                with _gzip.open(path_str, "rb") as _gz, open(gpkg_path, "wb") as _out:
                    shutil.copyfileobj(_gz, _out)
            elif not path_str.endswith(".gpkg"):
                gpkg_path = path_str + ".gpkg"
                extra_files.add(gpkg_path)
                shutil.copy(path_str, gpkg_path)
        except Exception as prep_exc:
            logger.warning(f"HDX GeoPackage pre-processing failed: {prep_exc}")

        minx, miny, maxx, maxy = bounds
        try:
            bbox_3857 = gpd.GeoSeries(
                [box(minx, miny, maxx, maxy)], crs="EPSG:4326"
            ).to_crs("EPSG:3857").total_bounds
        except Exception:
            bbox_3857 = None

        try:
            read_kwargs: dict = {}
            if bbox_3857 is not None:
                read_kwargs["bbox"] = tuple(bbox_3857)
            try:
                pop_gdf_raw = gpd.read_file(gpkg_path, **read_kwargs)
            except Exception:
                pop_gdf_raw = gpd.read_file(gpkg_path)
        except Exception as e:
            logger.error(f"Failed to read HDX GeoPackage: {e}")
            return None

        pop_gdf_raw = pop_gdf_raw.to_crs("EPSG:4326")
        pop_gdf_raw = pop_gdf_raw.cx[minx:maxx, miny:maxy]
        if pop_gdf_raw.empty:
            logger.warning("HDX GeoPackage has no features within boundary bbox.")
            return None

        clipped = gpd.clip(pop_gdf_raw, boundary_union)
        if clipped.empty:
            logger.warning("HDX GeoPackage: no features intersect boundary polygon.")
            return None

        pop_col = _resolve_population_column(clipped.drop(columns=["geometry"]))
        if pop_col is None:
            logger.warning(
                "HDX GeoPackage has no usable population column. "
                f"Columns: {list(clipped.columns)[:30]}"
            )
            return None

        clipped = clipped.copy()
        clipped[pop_col] = pd.to_numeric(clipped[pop_col], errors="coerce")
        clipped = clipped[clipped[pop_col] > 0]
        if clipped.empty:
            return None

        centroids_3857 = gpd.GeoSeries(
            clipped.to_crs("EPSG:3857").geometry.centroid,
            crs="EPSG:3857",
        )
        clipped["geometry"] = centroids_3857.to_crs("EPSG:4326").values
        # Edge hexagons clipped by gpd.clip can produce centroids that fall
        # fractionally outside the boundary due to floating-point precision.
        # Re-apply the boundary filter to the centroid points.
        try:
            clipped = clipped[clipped.geometry.within(boundary_union)].copy()
        except Exception:
            pass
        clipped = clipped.rename(columns={pop_col: "population"})
        result = clipped[["population", "geometry"]].copy()
        result["data_source"] = "hdx_kontur_population"
        logger.info(
            f"HDX GPKG: loaded {len(result)} cells using pop column '{pop_col}'."
        )
        return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")

    finally:
        paths_to_clean = set(extra_files)
        if not is_cached:
            paths_to_clean.add(path_str)
        for p in paths_to_clean:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


def fetch_population_hdx(boundary_gdf: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """Attempt to fetch Kontur/HDX population data for a boundary.

    Returns None (no raise) when data is unavailable so the pipeline can move
    on to the WorldPop / synthetic tiers.
    """
    try:
        from hdx.api.configuration import Configuration
        from hdx.data.dataset import Dataset
        from hdx.location.country import Country
    except ImportError:
        logger.warning("hdx-python-api not installed; HDX tier skipped.")
        return None

    try:
        boundary_4326 = boundary_gdf.to_crs("EPSG:4326")
        boundary_union = unary_union(boundary_4326.geometry.values)
        if boundary_union is None or boundary_union.is_empty:
            return None

        bounds = boundary_union.bounds
        centroid = boundary_union.centroid

        iso3, display = _resolve_iso3(boundary_4326, centroid)
        if not iso3:
            logger.warning("HDX: could not resolve ISO3 country code; skipping.")
            return None

        try:
            country_en = Country.get_country_name_from_iso3(iso3) or display or iso3
        except Exception:
            country_en = display or iso3
        logger.info(f"HDX population: resolved country as '{country_en}' ({iso3})")

        try:
            Configuration.create(
                hdx_site="prod",
                user_agent="SOCA_spopt_agent",
                hdx_read_only=True,
            )
        except Exception:
            pass  # already configured

        iso3_upper = iso3.upper()
        iso3_lower = iso3.lower()

        best_ds = None
        target_resource = None
        resource_kind = None

        # Primary path: Solr filter query on HDX's search. Restricting by
        # ``organization:kontur AND groups:{iso3_lower}`` returns every Kontur
        # dataset tagged for the country regardless of slug/name variations.
        # This is dramatically more reliable than guessing slugs because
        # Kontur uses the UN long name (e.g. "iran-islamic-republic-of",
        # "united-kingdom-of-great-britain-and-northern-ireland") with no
        # stable transformation from ISO3.
        try:
            kontur_hits = Dataset.search_in_hdx(
                "kontur population",
                fq=f"organization:kontur AND groups:{iso3_lower}",
                rows=10,
            )
        except Exception as search_exc:
            logger.warning(f"HDX Kontur Solr search failed: {search_exc}")
            kontur_hits = []

        for cand_ds in kontur_hits:
            name = (cand_ds.get("name") or "").lower()
            # Skip kontur-boundaries-* (admin polygons, not population rasters).
            if not name.startswith("kontur-population-"):
                continue
            r, kind = _pick_resource(cand_ds)
            if r is not None:
                best_ds, target_resource, resource_kind = cand_ds, r, kind
                logger.info(f"HDX: matched Kontur dataset '{name}' for {iso3_upper}")
                break

        # Secondary path: other high-quality spatial population datasets
        # (WorldPop / HRSL) when Kontur doesn't publish the country. Use the
        # same groups filter; keep text query broad.
        if target_resource is None:
            try:
                other_hits = Dataset.search_in_hdx(
                    "high resolution population density",
                    fq=f"groups:{iso3_lower}",
                    rows=15,
                )
            except Exception as search_exc:
                logger.warning(f"HDX fallback Solr search failed: {search_exc}")
                other_hits = []
            for cand_ds in other_hits:
                title_lower = (cand_ds.get("title") or "").lower()
                if not any(kw in title_lower for kw in _SPATIAL_POP_KEYWORDS):
                    continue
                r, kind = _pick_resource(cand_ds)
                if r is not None:
                    best_ds, target_resource, resource_kind = cand_ds, r, kind
                    logger.info(
                        f"HDX: fallback dataset '{cand_ds.get('name')}' for {iso3_upper}"
                    )
                    break

        if target_resource is None:
            logger.warning(
                f"No HDX population resource found for '{country_en}' ({iso3_upper})."
            )
            return None

        # Download (or hit the on-disk cache).
        import hashlib

        cache_dir = Path(
            os.environ.get("SOCA_HDX_CACHE_DIR")
            or Path.home() / ".cache" / "soca" / "hdx"
        )
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            cache_dir = None

        res_name = target_resource.get("name") or "resource"
        res_id = target_resource.get("id") or hashlib.sha1(
            res_name.encode("utf-8", "ignore")
        ).hexdigest()[:16]
        cache_path: Optional[Path] = None
        if cache_dir is not None:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", res_name)[:80]
            cache_path = cache_dir / f"{res_id}_{safe_name}"

        path_str: Optional[str] = None
        if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
            logger.info(
                f"HDX: using cached '{res_name}' from {cache_path} "
                f"({cache_path.stat().st_size // 1024 // 1024}MB)"
            )
            path_str = str(cache_path)
        else:
            logger.info(
                f"HDX: downloading '{res_name}' ({resource_kind}, "
                f"{_resource_size_mb(target_resource)}MB) from '{best_ds.get('title')}'"
            )
            try:
                dl_result = target_resource.download()
            except Exception as e:
                logger.error(f"Failed to download HDX resource: {e}")
                return None
            if isinstance(dl_result, (list, tuple)) and len(dl_result) == 2:
                _url, dl_path = dl_result
            else:
                dl_path = dl_result
            if not dl_path:
                logger.error("HDX resource download returned no path.")
                return None
            dl_path = str(dl_path)
            if cache_path is not None:
                import shutil as _shutil
                try:
                    _shutil.move(dl_path, cache_path)
                    path_str = str(cache_path)
                except Exception as mv_exc:
                    logger.debug(f"HDX cache move failed: {mv_exc}")
                    path_str = dl_path
            else:
                path_str = dl_path

        is_cached = (cache_path is not None and str(cache_path) == path_str)

        if resource_kind == "csv":
            return _read_hdx_csv(path_str, boundary_union, bounds, cleanup=not is_cached)
        elif resource_kind == "gpkg":
            return _read_hdx_gpkg(path_str, boundary_union, bounds, is_cached=is_cached)

        return None

    except Exception as exc:
        logger.warning(f"HDX population fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Orchestrator: fetch_population()
# ---------------------------------------------------------------------------

def fetch_population(
    boundary_gdf: gpd.GeoDataFrame,
    n_points: Optional[int] = None,
    random_seed: int = 42,
) -> gpd.GeoDataFrame:
    """Fetch population grid: Kontur (HDX) → synthetic uniform grid."""
    if n_points is None:
        try:
            from utils.scale_classifier import compute_n_points_from_area
            area_km2 = boundary_gdf.to_crs("EPSG:6933").geometry.area.sum() / 1e6
            n_points = compute_n_points_from_area(area_km2)
        except Exception:
            n_points = 200

    # Tier 1: Kontur/HDX (wall-clock capped in a worker thread).
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fetch_population_hdx, boundary_gdf)
            try:
                pop_gdf = fut.result(timeout=_HDX_FETCH_TIMEOUT_SEC)
                if pop_gdf is not None and len(pop_gdf) > 0:
                    return pop_gdf
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"HDX population fetch timed out after {_HDX_FETCH_TIMEOUT_SEC}s; "
                    "falling back to synthetic grid."
                )
                from utils.activity_log import log_event
                log_event(
                    "population.fetch",
                    "warn",
                    f"HDX download timed out after {_HDX_FETCH_TIMEOUT_SEC}s "
                    "(large country file) — using synthetic population grid instead. "
                    "Results will be approximate.",
                    source="HDX",
                )
    except Exception as exc:
        logger.warning(f"HDX tier failed: {exc}")
        from utils.activity_log import log_event
        log_event(
            "population.fetch",
            "warn",
            f"HDX download failed ({exc}) — using synthetic population grid instead. "
            "Results will be approximate.",
            source="HDX",
        )

    # Tier 2: synthetic uniform grid.
    total = estimate_total_population(boundary_gdf)
    return generate_synthetic_population(
        boundary_gdf, n_points=n_points,
        total_population=total, random_seed=random_seed,
    )
