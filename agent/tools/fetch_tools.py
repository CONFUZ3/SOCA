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

from .state_bridge import get_data, get_problem_state, get_aoi, get_aoi_boundary_gdf

logger = logging.getLogger(__name__)

# Supported POI categories (mirrors DataFetcher.OVERTURE_CATEGORIES keys)
_VALID_POI_CATEGORIES = (
    "health", "education", "food", "finance", "fire_station",
    "police", "library", "transport", "water", "emergency",
)

_VALID_SCALES = ("country", "region", "city", "neighborhood")
_SCALE_ADMIN_LEVEL = {"country": 3, "region": 5, "city": 7, "neighborhood": 9}


def fetch_city_data(
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
    from utils.data_fetcher import DataFetcher, DataFetchError
    from utils.data_processor import DataProcessor
    from utils.scale_classifier import validate_boundary_scale

    # Normalise scale
    scale = scale.strip().lower()
    if scale not in _VALID_SCALES:
        from utils.scale_classifier import heuristic_scale_from_location
        scale = heuristic_scale_from_location(location)
        logger.info("fetch_city_data: scale inferred as '%s' for '%s'", scale, location)

    # Normalise admin_level
    if not isinstance(admin_level, int) or not (2 <= admin_level <= 10):
        admin_level = _SCALE_ADMIN_LEVEL.get(scale, 7)
        logger.info("fetch_city_data: admin_level defaulted to %d", admin_level)

    # Validate POI category
    if poi_category not in _VALID_POI_CATEGORIES:
        poi_category = "health"

    fetcher = DataFetcher()
    processor = DataProcessor()
    data_store = get_data()

    slug = re.sub(r"[^a-z0-9]+", "_", location.lower()).strip("_") or "aoi"

    fetched_datasets: list = []
    summaries: list = []
    errors: list = []
    boundary_gdf = None

    # ---- AOI short-circuit: reuse user-defined AOI as the boundary ----
    aoi_info = get_aoi()
    aoi_gdf = get_aoi_boundary_gdf()
    if aoi_info is not None and aoi_gdf is not None and len(aoi_gdf) > 0:
        boundary_gdf = aoi_gdf
        aoi_name = aoi_info.get("name", "AOI")
        summaries.append(
            f"Using user-defined AOI '{aoi_name}' "
            f"({aoi_info.get('area_km2', 0):,.1f} km²) as boundary — skipping geocoding."
        )
        logger.info("fetch_city_data: AOI short-circuit for '%s'", aoi_name)
    elif include_boundaries:
        # ---- Step 1: Boundaries (only when no AOI) ----
        try:
            gdf = fetcher.fetch_boundaries(
                location, admin_level=admin_level, scale=scale
            )
            gdf = processor.preprocess_data(gdf)
            key = f"boundary_{slug}"
            gdf.attrs["source"] = "auto_fetched"
            data_store[key] = gdf
            boundary_gdf = gdf

            # Soft-validate scale
            try:
                valid, hint = validate_boundary_scale(gdf, scale)
                if not valid:
                    summaries.append(f"Note: {hint}")
            except Exception:
                pass

            msg = f"Boundary ({location}): 1 polygon [scale={scale}, admin_level={admin_level}]"
            summaries.append(msg)
            fetched_datasets.append(key)
            logger.info("fetch_city_data: %s", msg)
        except DataFetchError as exc:
            err = f"Boundary fetch failed for '{location}': {exc}"
            errors.append(err)
            logger.error("fetch_city_data: %s", err)

    # ---- Step 2: Population / demand grid ----
    if include_population:
        if boundary_gdf is None:
            errors.append("Population step skipped: boundary not available.")
        else:
            try:
                gdf = fetcher.fetch_population(boundary_gdf)
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
                summaries.append(msg)
                fetched_datasets.append(key)
                logger.info("fetch_city_data: %s", msg)
            except DataFetchError as exc:
                err = f"Population fetch failed for '{location}': {exc}"
                errors.append(err)
                logger.error("fetch_city_data: %s", err)

    # ---- Step 3: POIs ----
    if include_pois:
        if boundary_gdf is None:
            errors.append(f"POI step ('{poi_category}') skipped: boundary not available.")
        else:
            try:
                gdf = fetcher.fetch_pois(boundary_gdf, poi_category)
                gdf = processor.preprocess_data(gdf)
                key = f"{poi_category}_facilities_{slug}"
                gdf.attrs["source"] = "auto_fetched"
                data_store[key] = gdf

                msg = (
                    f"{poi_category.title()} facilities ({location}): "
                    f"{len(gdf)} points from public sources"
                )
                summaries.append(msg)
                fetched_datasets.append(key)
                logger.info("fetch_city_data: %s", msg)
            except DataFetchError as exc:
                err = f"{poi_category.title()} facilities fetch failed for '{location}': {exc}"
                errors.append(err)
                logger.error("fetch_city_data: %s", err)

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
