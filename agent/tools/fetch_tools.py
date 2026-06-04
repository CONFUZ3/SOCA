"""
ADK tool: fetch_city_data

Fetches administrative boundary, population demand grid, and/or POIs for a
named place from public sources (Overture Maps / OpenStreetMap / HDX).
Results are written directly into the Streamlit session data store via the
thread-local state bridge.
"""

import re
import logging
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from utils.scale_classifier import (
    SCALE_ADMIN_LEVELS,
    VALID_POI_CATEGORIES,
    VALID_SCALES,
)

from .state_bridge import get_data, get_problem_state, get_aoi, get_aoi_boundary_gdf

logger = logging.getLogger(__name__)


def _activity_bus():
    """Return the already-loaded FastAPI event bus, if this process has one."""
    import sys

    module = sys.modules.get("backend.services.event_bus")
    if module is None:
        return None
    get_default_bus = getattr(module, "get_default_bus", None)
    if get_default_bus is None:
        return None
    try:
        return get_default_bus()
    except Exception:
        return None


def _current_activity_session_id() -> Optional[str]:
    bus = _activity_bus()
    if bus is None:
        return None
    try:
        return bus.current_session()
    except Exception:
        return None


def _bind_activity_session(session_id: Optional[str]) -> Optional[str]:
    bus = _activity_bus()
    if bus is None:
        return None
    try:
        previous = bus.current_session()
        bus.bind_session(session_id)
        return previous
    except Exception:
        return None


def _step_boundaries(
    fetcher,
    processor,
    location: str,
    admin_level: int,
    scale: str,
    data_store: dict,
    slug: str,
) -> tuple:
    """Fetch boundary polygon. Returns (gdf_or_None, key_or_None, summaries, errors)."""
    from utils.data_fetcher import DataFetchError
    from utils.scale_classifier import validate_boundary_scale
    from utils.activity_log import timed

    summaries: list = []
    try:
        with timed(
            "fetch.boundary",
            detail=f"Finding the AOI boundary for {location}",
        ) as step:
            gdf = fetcher.fetch_boundaries(
                location, admin_level=admin_level, scale=scale
            )
            step.detail = f"AOI boundary ready for {location}"
        gdf = processor.preprocess_data(gdf)
        key = f"boundary_{slug}"
        gdf.attrs["source"] = "auto_fetched"
        data_store[key] = gdf
        try:
            valid, hint = validate_boundary_scale(gdf, scale)
            if not valid:
                summaries.append(f"Note: {hint}")
        except Exception as exc:
            # Validation must not break the fetch — but the user should know
            # the scale check was inconclusive instead of silently passing.
            logger.warning("boundary scale validation failed: %s", exc)
            summaries.append(f"Note: boundary scale check inconclusive ({exc}).")
        msg = (
            f"Boundary ({location}): 1 polygon "
            f"[scale={scale}, admin_level={admin_level}]"
        )
        summaries.append(msg)
        logger.info("fetch_city_data: %s", msg)
        return gdf, key, summaries, []
    except DataFetchError as exc:
        err = f"Boundary fetch failed for '{location}': {exc}"
        logger.error("fetch_city_data: %s", err)
        return None, None, [], [err]


def _step_population(
    fetcher,
    processor,
    location: str,
    scale: str,
    boundary_gdf,
    data_store: dict,
    slug: str,
) -> tuple:
    """Fetch population demand grid. Returns (key_or_None, summaries, errors)."""
    from utils.data_fetcher import DataFetchError
    from utils.activity_log import timed

    if boundary_gdf is None:
        return None, [], ["Population step skipped: boundary not available."]
    try:
        with timed(
            "fetch.population",
            detail=f"Loading population grid for {location}",
        ) as step:
            gdf = fetcher.fetch_population(boundary_gdf)
            step.detail = f"Population grid ready for {location}"
        gdf = processor.preprocess_data(gdf)
        key = f"demand_{slug}"
        gdf.attrs["source"] = "auto_fetched"
        data_store[key] = gdf
        src = (
            gdf["data_source"].iloc[0]
            if "data_source" in gdf.columns and len(gdf) > 0
            else "synthetic_uniform_grid"
        )
        src_label = "real data" if src != "synthetic_uniform_grid" else "synthetic"
        msg = (
            f"Population grid ({location}): {len(gdf)} demand points "
            f"[{src_label}, scale={scale}]"
        )
        logger.info("fetch_city_data: %s", msg)
        return key, [msg], []
    except DataFetchError as exc:
        err = f"Population fetch failed for '{location}': {exc}"
        logger.error("fetch_city_data: %s", err)
        return None, [], [err]


def _step_pois(
    fetcher,
    processor,
    location: str,
    poi_category: str,
    boundary_gdf,
    data_store: dict,
    slug: str,
) -> tuple:
    """Fetch POI candidate sites. Returns (key_or_None, summaries, errors)."""
    from utils.data_fetcher import DataFetchError
    from utils.activity_log import timed

    if boundary_gdf is None:
        return None, [], [
            f"POI step ('{poi_category}') skipped: boundary not available."
        ]
    try:
        with timed(
            "fetch.pois",
            detail=f"Loading {poi_category} facility locations",
        ) as step:
            gdf = fetcher.fetch_pois(boundary_gdf, poi_category)
            step.detail = f"{poi_category.title()} facility locations ready"
        # Preserve provenance attrs across DataProcessor.preprocess_data.
        tier_counts = gdf.attrs.get("tier_counts", {})
        tier_errors = gdf.attrs.get("tier_errors", {})
        gdf = processor.preprocess_data(gdf)
        if tier_counts:
            gdf.attrs["tier_counts"] = tier_counts
        if tier_errors:
            gdf.attrs["tier_errors"] = tier_errors
        key = f"{poi_category}_facilities_{slug}"
        gdf.attrs["source"] = "auto_fetched"
        data_store[key] = gdf
        if tier_counts:
            msg = (
                f"{poi_category.title()} facilities ({location}): "
                f"{len(gdf)} points (Overture: {tier_counts.get('overture', 0)}, "
                f"OSM: {tier_counts.get('overpass', 0)})"
            )
        else:
            msg = (
                f"{poi_category.title()} facilities ({location}): "
                f"{len(gdf)} points from public sources"
            )
        logger.info("fetch_city_data: %s", msg)
        return key, [msg], []
    except DataFetchError as exc:
        err = (
            f"{poi_category.title()} facilities fetch failed "
            f"for '{location}': {exc}"
        )
        logger.error("fetch_city_data: %s", err)
        return None, [], [err]


def _fetch_city_data_sync(
    location: str,
    scale: str,
    admin_level: int,
    include_boundaries: bool,
    include_population: bool,
    include_pois: bool,
    poi_category: str,
    data_store: dict,
    aoi_info,
    aoi_gdf,
    tool_context: Optional[ToolContext],
    activity_session_id: Optional[str],
) -> dict:
    """Synchronous body of fetch_city_data — runs in a thread pool worker."""
    from utils.data_fetcher import DataFetcher
    from utils.data_processor import DataProcessor
    from utils.activity_log import log_event

    previous_session = _bind_activity_session(activity_session_id)
    try:
        scale = scale.strip().lower()
        if scale not in VALID_SCALES:
            from utils.scale_classifier import heuristic_scale_from_location
            scale = heuristic_scale_from_location(location)
            logger.info("fetch_city_data: scale inferred as '%s' for '%s'", scale, location)

        if not isinstance(admin_level, int) or not (2 <= admin_level <= 10):
            admin_level = SCALE_ADMIN_LEVELS.get(scale, 7)
            logger.info("fetch_city_data: admin_level defaulted to %d", admin_level)

        if poi_category not in VALID_POI_CATEGORIES:
            poi_category = "health"

        fetcher = DataFetcher()
        processor = DataProcessor()
        slug = re.sub(r"[^a-z0-9]+", "_", location.lower()).strip("_") or "aoi"

        fetched_datasets: list = []
        summaries: list = []
        errors: list = []
        boundary_gdf = None

        # AOI short-circuit: reuse user-defined AOI as the boundary
        if aoi_info is not None and aoi_gdf is not None and len(aoi_gdf) > 0:
            boundary_gdf = aoi_gdf
            aoi_name = aoi_info.get("name", "AOI")
            log_event(
                "fetch.boundary",
                "ok",
                f"Using the selected AOI boundary for {aoi_name}",
            )
            summaries.append(
                f"Using user-defined AOI '{aoi_name}' "
                f"({aoi_info.get('area_km2', 0):,.1f} km²) as boundary — skipping geocoding."
            )
            logger.info("fetch_city_data: AOI short-circuit for '%s'", aoi_name)
        elif include_boundaries:
            boundary_gdf, key, new_summaries, new_errors = _step_boundaries(
                fetcher, processor, location, admin_level, scale, data_store, slug
            )
            summaries.extend(new_summaries)
            errors.extend(new_errors)
            if key:
                fetched_datasets.append(key)

        if include_population:
            key, new_summaries, new_errors = _step_population(
                fetcher, processor, location, scale, boundary_gdf, data_store, slug
            )
            summaries.extend(new_summaries)
            errors.extend(new_errors)
            if key:
                fetched_datasets.append(key)

        if include_pois:
            key, new_summaries, new_errors = _step_pois(
                fetcher, processor, location, poi_category, boundary_gdf, data_store, slug
            )
            summaries.extend(new_summaries)
            errors.extend(new_errors)
            if key:
                fetched_datasets.append(key)

        # Update data_summary in ADK session state so the agent knows what's available
        if tool_context is not None:
            existing = dict(tool_context.state.get("data_summary") or {})
            for key in fetched_datasets:
                gdf = data_store.get(key)
                if gdf is not None:
                    existing[key] = {
                        "num_features": len(gdf),
                        "geometry_type": (
                            gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown"
                        ),
                        "columns": [c for c in gdf.columns if c != "geometry"],
                        "source": "auto_fetched",
                    }
            tool_context.state["data_summary"] = existing

        overall_status = "error" if (not fetched_datasets and errors) else (
            "partial" if errors else "success"
        )

        return {
            "status": overall_status,
            "fetched_datasets": fetched_datasets,
            "summaries": summaries,
            "errors": errors,
        }
    finally:
        _bind_activity_session(previous_session)


async def fetch_city_data(
    location: str,
    scale: str = "city",
    admin_level: int = 7,
    include_boundaries: bool = True,
    include_population: bool = True,
    include_pois: bool = False,
    poi_category: str = "health",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Fetch geographic data for a named place automatically from public sources.

    Retrieves administrative boundary polygon, population demand grid, and
    optionally Points of Interest. No manual upload is required. Datasets are
    stored in the session and will appear on the map.

    AOI awareness: if the user has already defined an Area of Interest in the
    app (boundary_aoi present in the session), this tool REUSES that polygon
    as the boundary and only fetches population/POIs clipped to it. In that
    case the `location` argument is used only as a label; no geocoding runs.

    Args:
        location: Place name with country, e.g. "Lima, Peru" or "Nairobi, Kenya".
                  If an AOI is already set, this is just a label — pass the AOI
                  name (available via get_data_status).
        scale: Geographic scope – one of: "country", "region", "city",
               "neighborhood". Determines the boundary admin level used.
        admin_level: OSM admin_level integer (2-10). Typical values:
                     country=3, region=5, city=7, neighborhood=9.
        include_boundaries: Whether to fetch the administrative boundary polygon.
        include_population: Whether to fetch a population demand grid.
        include_pois: Whether to fetch existing facility locations as candidates.
        poi_category: Facility category for POI fetch. One of: health,
                      education, food, finance, fire_station, police, library,
                      transport, water, emergency.

    Returns:
        dict with keys:
          status (str): "success" or "partial" or "error"
          fetched_datasets (list[str]): names of datasets written to session
          summaries (list[str]): human-readable summary per step
          errors (list[str]): per-step error messages (non-fatal)
    """
    import asyncio

    # Capture thread-local state bridge values on the event-loop thread before
    # handing off to a worker thread where thread-locals won't be set.
    data_store = get_data()
    aoi_info = get_aoi()
    aoi_gdf = get_aoi_boundary_gdf()
    activity_session_id = _current_activity_session_id()

    return await asyncio.to_thread(
        _fetch_city_data_sync,
        location,
        scale,
        admin_level,
        include_boundaries,
        include_population,
        include_pois,
        poi_category,
        data_store,
        aoi_info,
        aoi_gdf,
        tool_context,
        activity_session_id,
    )


def _fetch_custom_data_sync(
    data_type: str,
    location: str,
    query: str,
    osm_tags_dict: dict,
    data_store: dict,
    aoi_gdf,
    tool_context: Optional[ToolContext],
    activity_session_id: Optional[str],
) -> dict:
    from utils.fetchers import source_registry
    from utils.data_fetcher import DataFetcher, DataFetchError
    from utils.activity_log import timed, log_event

    previous_session = _bind_activity_session(activity_session_id)
    try:
        plugin = source_registry.get(data_type)
        if plugin is None:
            return {
                "status": "error",
                "dataset_key": None,
                "summary": None,
                "error": f"Unknown plugin '{data_type}'.",
                "available": source_registry.list_available(),
            }

        ok, msg = plugin.validate_params(query=query, osm_tags=osm_tags_dict)
        if not ok:
            return {
                "status": "error",
                "dataset_key": None,
                "summary": None,
                "error": f"Invalid params for plugin '{data_type}': {msg}",
            }

        boundary_gdf = aoi_gdf
        if boundary_gdf is None or len(boundary_gdf) == 0:
            if not location.strip():
                return {
                    "status": "error",
                    "dataset_key": None,
                    "summary": None,
                    "error": "No AOI is set and no location was provided.",
                }
            try:
                with timed("fetch.boundary", detail=f"Resolving boundary for {location}"):
                    boundary_gdf = DataFetcher().fetch_boundaries(location)
            except DataFetchError as exc:
                return {
                    "status": "error",
                    "dataset_key": None,
                    "summary": None,
                    "error": f"Boundary resolution failed for '{location}': {exc}",
                }

        slug = re.sub(r"[^a-z0-9]+", "_", (location or "aoi").lower()).strip("_") or "aoi"
        key = f"{data_type}_{slug}"

        try:
            with timed(
                f"fetch.{data_type}",
                detail=f"Fetching {data_type} data",
                source=plugin.name,
            ) as step:
                gdf = plugin.fetch(boundary_gdf, query=query, osm_tags=osm_tags_dict)
                step.detail = f"{data_type}: {len(gdf)} features"
        except Exception as exc:
            log_event(
                f"fetch.{data_type}",
                "fail",
                str(exc),
                source=plugin.name,
            )
            return {
                "status": "error",
                "dataset_key": None,
                "summary": None,
                "error": f"Plugin '{data_type}' fetch failed: {exc}",
            }

        gdf.attrs["source"] = f"plugin:{data_type}"
        data_store[key] = gdf

        if tool_context is not None:
            existing = dict(tool_context.state.get("data_summary") or {})
            existing[key] = {
                "num_features": len(gdf),
                "geometry_type": (
                    gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown"
                ),
                "columns": [c for c in gdf.columns if c != "geometry"],
                "source": f"plugin:{data_type}",
            }
            tool_context.state["data_summary"] = existing

        return {
            "status": "ok",
            "dataset_key": key,
            "summary": {
                "num_features": len(gdf),
                "plugin": data_type,
            },
            "error": None,
        }
    finally:
        _bind_activity_session(previous_session)


async def fetch_custom_data(
    data_type: str,
    location: str = "",
    query: str = "",
    osm_tags: str = "",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Fetch a non-standard dataset from a registered source plugin.

    Use this when fetch_city_data does not cover the data type needed. The
    data_type must be one of the registered plugin names; call get_data_status
    to see which plugins are available.

    Args:
        data_type: Plugin name (e.g. 'hdx_generic', 'overpass_custom').
        location: Place name, used to resolve boundary if no AOI is loaded.
        query: Free-form search query (required for hdx_generic).
        osm_tags: JSON-encoded OSM tag dict (required for overpass_custom),
                  e.g. '{"amenity":"school"}'.
        tool_context: Injected by ADK; do not pass manually.

    Returns:
        dict with keys: status, dataset_key, summary, error
    """
    import asyncio
    import json

    osm_tags_dict: dict = {}
    if osm_tags:
        try:
            parsed = json.loads(osm_tags)
            if isinstance(parsed, dict):
                osm_tags_dict = parsed
        except json.JSONDecodeError as exc:
            return {
                "status": "error",
                "dataset_key": None,
                "summary": None,
                "error": f"osm_tags is not valid JSON: {exc}",
            }

    data_store = get_data()
    aoi_gdf = get_aoi_boundary_gdf()
    activity_session_id = _current_activity_session_id()

    return await asyncio.to_thread(
        _fetch_custom_data_sync,
        data_type,
        location,
        query,
        osm_tags_dict,
        data_store,
        aoi_gdf,
        tool_context,
        activity_session_id,
    )
