"""
AOI selector — the first step in the SOCA flow.

Renders a full-width map where the user can either:
  1. Type a place name — live autocomplete dropdown (Photon/Nominatim) shows
     disambiguated candidates with admin context. Clicking one fetches the
     real boundary polygon straight from OSM using the relation id, so
     "Brooklyn, NY" never gets confused with "Brooklyn Park, MN".
  2. Draw a polygon / rectangle directly on the map.
  3. Refine a searched boundary by editing its vertices — the boundary is
     injected into the Draw FeatureGroup so Leaflet.Draw's edit tool can
     operate on it.

The map has a basemap toggle (CartoDB Positron for readability,
Esri World Imagery for satellite precision). Every data fetch emits events
to the shared activity log so users can see which open-data source served
them — no black boxes.

Returns the confirmed AOI dict (or None while the user is still working).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import folium
import geopandas as gpd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
from streamlit_folium import st_folium

from utils.activity_log import log_event, render_log
from utils.geocoder import GeocodeCandidate, suggest as geocoder_suggest

logger = logging.getLogger(__name__)

# Guardrails (km²)
MIN_AOI_KM2 = 0.5
MAX_AOI_KM2 = 50_000.0

# Autocomplete config
_MIN_SUGGEST_CHARS = 3
_MAX_SUGGEST_SHOWN = 6

# Basemap options (label → folium tiles argument)
_BASEMAPS = {
    "Light (CartoDB)": "CartoDB positron",
    "Street (OSM)": "OpenStreetMap",
    "Satellite (Esri)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the "
        "GIS User Community",
    ),
}

# Pre-resolved example chips — no geocoding round-trip needed, but they go
# through the normal fetch path so the user sees the activity log.
_EXAMPLES = ["Lima, Peru", "Brooklyn, New York", "Nairobi, Kenya", "Mirpur, Dhaka"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _area_km2(geom) -> float:
    """Geodesic area in km² using EPSG:6933 equal-area projection."""
    gs = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:6933")
    return float(gs.area.iloc[0]) / 1_000_000.0


def _simplify_for_edit(gdf: gpd.GeoDataFrame, tolerance: float = 0.001) -> gpd.GeoDataFrame:
    """Simplify a polygon so the Draw plugin stays responsive when editing."""
    try:
        simplified = gdf.copy()
        simplified["geometry"] = simplified.geometry.simplify(
            tolerance, preserve_topology=True
        )
        return simplified
    except Exception:
        return gdf


def _geojson_to_geom(feature: dict):
    """Extract shapely geom from a GeoJSON Feature or Geometry dict."""
    if feature is None:
        return None
    if feature.get("type") == "Feature":
        return shape(feature["geometry"])
    return shape(feature)


def _validate(geom) -> tuple[bool, Optional[str], float]:
    """Check polygon validity and area bounds. Returns (ok, err_msg, area_km2)."""
    if geom is None or geom.is_empty:
        return False, "No polygon drawn.", 0.0

    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return False, f"AOI must be a polygon (got {geom.geom_type}).", 0.0

    if not geom.is_valid:
        try:
            geom = make_valid(geom)
        except Exception:
            return False, "Polygon is self-intersecting or invalid.", 0.0

    area = _area_km2(geom)
    if area < MIN_AOI_KM2:
        return False, f"AOI too small ({area:.3f} km²). Minimum is {MIN_AOI_KM2} km².", area
    if area > MAX_AOI_KM2:
        return False, f"AOI too large ({area:,.0f} km²). Maximum is {MAX_AOI_KM2:,.0f} km².", area

    # Sanity: lat/lon bounds
    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
        return False, "Polygon coordinates outside lat/lon range.", area

    return True, None, area


# ---------------------------------------------------------------------------
# Fetch pipeline
# ---------------------------------------------------------------------------


def _get_fetcher():
    from utils.data_fetcher import DataFetcher
    if st.session_state.get("data_fetcher") is None:
        st.session_state.data_fetcher = DataFetcher()
    return st.session_state.data_fetcher


def _fetch_boundary_for(
    *,
    query: str,
    hint: Optional[GeocodeCandidate] = None,
) -> None:
    """Fetch a boundary for *query* (optionally with a candidate hint) and stash
    the result into session for the map to pick up.

    Emits activity-log events throughout so the user sees which tier responded.
    """
    label = hint.display_name if hint is not None else query
    try:
        with st.status(f"Finding boundary for '{label}'…", expanded=True) as s:
            # Live event stream: we re-render the same events the fetcher emits
            # into the status panel so the user sees them as they arrive.
            _event_count_before = len(_activity_buffer())

            fetcher = _get_fetcher()
            gdf = fetcher.fetch_boundaries(query, hint=hint)

            # After the fetch, render the new events for this operation.
            for evt in _activity_buffer()[_event_count_before:]:
                s.write(evt.format())

            s.update(label=f"Boundary loaded · {label}", state="complete")
    except Exception as exc:
        logger.warning("AOI fetch failed for %s: %s", label, exc)
        log_event("boundary.fetch", "fail", f"{label}: {exc}", source="pipeline")
        st.error(
            f"Could not find '{label}'. The activity log below has details. "
            "You can also draw the AOI manually on the map."
        )
        return

    if gdf is None or len(gdf) == 0:
        st.error(f"No boundary returned for '{label}'.")
        return

    gdf = _simplify_for_edit(gdf)
    geom = gdf.geometry.iloc[0]

    # Preserve country metadata from the resolved boundary so downstream
    # population lookups (e.g. HDX Kontur) can resolve ISO3 without a second
    # network round-trip. The fetcher already extracted these from Nominatim's
    # addressdetails; fall back to the hint candidate's country name if the
    # fetcher-specific columns are missing (e.g. Overture / GADM tiers).
    first_row = gdf.iloc[0] if len(gdf) else None
    country_name = ""
    country_code = ""
    if first_row is not None:
        country_name = str(first_row.get("country") or "").strip()
        country_code = str(first_row.get("country_code") or "").strip().upper()
    if not country_name and hint is not None:
        country_name = (hint.country or "").strip()

    st.session_state["_aoi_search_result"] = {
        "name": hint.short_name if hint is not None else query,
        "display_name": hint.display_name if hint is not None else query,
        "geojson": json.loads(gdf.to_json()),
        "bounds": list(gdf.total_bounds),
        "admin_level": gdf.attrs.get("admin_level") if hasattr(gdf, "attrs") else None,
        "osm_id": hint.osm_id if hint is not None else None,
        "source_tier": str(gdf.iloc[0].get("source", "unknown")) if len(gdf) else "unknown",
        "country": country_name,
        "country_code": country_code,
    }
    st.session_state["_aoi_map_key"] = st.session_state.get("_aoi_map_key", 0) + 1
    st.rerun()


def _activity_buffer():
    """Shared ring buffer used by activity_log.log_event."""
    return st.session_state.setdefault("_activity_log", [])


# ---------------------------------------------------------------------------
# Session-state plumbing
# ---------------------------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("_aoi_search_result", None)
    st.session_state.setdefault("_aoi_current_geom", None)
    st.session_state.setdefault("_aoi_current_name", None)
    st.session_state.setdefault("_aoi_current_source", None)
    st.session_state.setdefault("_aoi_map_key", 0)
    st.session_state.setdefault("_aoi_search_input", "")
    st.session_state.setdefault("_aoi_basemap", next(iter(_BASEMAPS)))


def _clear_selection() -> None:
    for k in ("_aoi_search_result", "_aoi_current_geom", "_aoi_current_name", "_aoi_current_source"):
        st.session_state[k] = None
    st.session_state["_aoi_map_key"] += 1


# ---------------------------------------------------------------------------
# Map construction
# ---------------------------------------------------------------------------


def _make_base_map(center: tuple[float, float], zoom: int) -> folium.Map:
    basemap_key = st.session_state.get("_aoi_basemap") or next(iter(_BASEMAPS))
    tiles_cfg = _BASEMAPS.get(basemap_key, "CartoDB positron")
    if isinstance(tiles_cfg, tuple):
        url, attr = tiles_cfg
        fmap = folium.Map(location=list(center), zoom_start=zoom, tiles=None)
        folium.TileLayer(tiles=url, attr=attr, name=basemap_key, control=False).add_to(fmap)
    else:
        fmap = folium.Map(location=list(center), zoom_start=zoom, tiles=tiles_cfg)
    return fmap


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render_aoi_selector() -> Optional[dict]:
    """Render the AOI picker. Returns the confirmed AOI dict or None."""
    _init_state()

    st.markdown("### Step 1 of 2 · Define your Area of Interest")
    st.caption(
        "Type a place name for live suggestions, or draw a polygon / rectangle "
        "on the map. You can also edit a fetched boundary's vertices before confirming."
    )

    # --- Search bar with autocomplete -----------------------------------
    search_col, basemap_col = st.columns([3, 1])
    with search_col:
        query = st.text_input(
            "Search place",
            value=st.session_state.get("_aoi_search_input", ""),
            label_visibility="collapsed",
            placeholder="Start typing — e.g. 'Brooklyn', 'Lima', 'Mirpur'…",
            key="_aoi_search_input",
        )
    with basemap_col:
        st.selectbox(
            "Basemap",
            options=list(_BASEMAPS.keys()),
            key="_aoi_basemap",
            label_visibility="collapsed",
        )

    _render_suggestions(query)

    # Example chips (only shown when search is empty)
    if not query.strip():
        st.caption("Or try an example:")
        ex_cols = st.columns(len(_EXAMPLES))
        for i, ex in enumerate(_EXAMPLES):
            with ex_cols[i]:
                if st.button(ex, key=f"_aoi_example_{i}", use_container_width=True):
                    st.session_state["_aoi_search_input"] = ex
                    _fetch_boundary_for(query=ex, hint=None)

    # --- Map + details side panel --------------------------------------
    map_col, info_col = st.columns([3, 1])

    search_result = st.session_state.get("_aoi_search_result")

    if search_result is not None:
        minx, miny, maxx, maxy = search_result["bounds"]
        center = ((miny + maxy) / 2.0, (minx + maxx) / 2.0)
        fmap = _make_base_map(center, zoom=10)
        fmap.fit_bounds([[miny, minx], [maxy, maxx]])
    else:
        fmap = _make_base_map((20.0, 0.0), zoom=2)

    # Put the fetched boundary INTO the Draw FeatureGroup so Leaflet.Draw's
    # edit tool can operate on it directly. Without this, boundaries render
    # as read-only overlays and the "refine vertices" promise is broken.
    draw_feature_group = folium.FeatureGroup(name="AOI")

    if search_result is not None:
        folium.GeoJson(
            search_result["geojson"],
            name=f"Boundary — {search_result['name']}",
            style_function=lambda _f: {
                "color": "#1f77b4",
                "weight": 2,
                "fillColor": "#1f77b4",
                "fillOpacity": 0.15,
            },
        ).add_to(draw_feature_group)

    draw_feature_group.add_to(fmap)

    Draw(
        feature_group=draw_feature_group,
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)

    with map_col:
        out = st_folium(
            fmap,
            height=520,
            use_container_width=True,
            returned_objects=["all_drawings", "last_active_drawing"],
            key=f"_aoi_map_{st.session_state['_aoi_map_key']}",
        )

    # Resolve the current candidate geometry.
    # Priority: user-edited drawing > searched boundary.
    current_geom = None
    current_name = None
    current_source = None

    drawings = (out or {}).get("all_drawings") or []
    last_active = (out or {}).get("last_active_drawing")

    if last_active:
        current_geom = _geojson_to_geom(last_active)
        if search_result is not None:
            current_name = search_result["name"]
            current_source = "search+refined"
        else:
            current_name = "Custom polygon"
            current_source = "drawn"
    elif drawings:
        current_geom = _geojson_to_geom(drawings[-1])
        current_name = "Custom polygon"
        current_source = "drawn"
    elif search_result is not None:
        feat = search_result["geojson"]["features"][0]
        current_geom = _geojson_to_geom(feat)
        current_name = search_result["name"]
        current_source = f"search ({search_result.get('source_tier', 'unknown')})"

    st.session_state["_aoi_current_geom"] = current_geom
    st.session_state["_aoi_current_name"] = current_name
    st.session_state["_aoi_current_source"] = current_source

    # --- Info panel ----------------------------------------------------
    confirmed_aoi: Optional[dict] = None
    with info_col:
        st.markdown("#### Selected AOI")
        if current_geom is None:
            st.info("No AOI yet. Search a place, pick a suggestion, or draw a polygon on the map.")
            st.button("Confirm AOI", disabled=True, use_container_width=True)
        else:
            ok, err, area = _validate(current_geom)
            minx, miny, maxx, maxy = current_geom.bounds

            st.markdown(f"**Name:** {current_name}")
            st.markdown(f"**Source:** `{current_source}`")
            st.markdown(f"**Area:** {area:,.2f} km²")
            st.markdown(f"**Bounds:** {minx:.3f}, {miny:.3f} → {maxx:.3f}, {maxy:.3f}")

            if not ok:
                st.error(err)

            cbtn_col, clearbtn_col = st.columns(2)
            with clearbtn_col:
                if st.button("Clear", use_container_width=True):
                    _clear_selection()
                    st.rerun()
            with cbtn_col:
                confirm = st.button(
                    "Confirm AOI  ▶",
                    disabled=not ok,
                    type="primary",
                    use_container_width=True,
                )

            if confirm and ok:
                confirmed_aoi = {
                    "name": current_name,
                    "source": current_source,
                    "geometry_geojson": mapping(current_geom),
                    "bbox": [minx, miny, maxx, maxy],
                    "area_km2": area,
                    "admin_level": (search_result or {}).get("admin_level"),
                    "osm_id": (search_result or {}).get("osm_id"),
                    "country": (search_result or {}).get("country", ""),
                    "country_code": (search_result or {}).get("country_code", ""),
                }

    # --- Activity log (always visible — transparency) -------------------
    render_log(expanded=False)

    return confirmed_aoi


# ---------------------------------------------------------------------------
# Autocomplete suggestions
# ---------------------------------------------------------------------------


def _render_suggestions(query: str) -> None:
    """Render the autocomplete dropdown below the search box.

    Only queries the geocoder when the input is at least _MIN_SUGGEST_CHARS.
    Each candidate renders as a button — clicking it fetches the boundary
    using the candidate's OSM relation id (fastest and most accurate path).
    """
    q = (query or "").strip()
    if len(q) < _MIN_SUGGEST_CHARS:
        return

    try:
        candidates = geocoder_suggest(q, limit=_MAX_SUGGEST_SHOWN)
    except Exception as exc:
        st.caption(f"Autocomplete unavailable: {exc}")
        return

    if not candidates:
        st.caption("No matching places. Press Enter to try as a raw query, or draw on the map.")
        return

    st.caption(f"Suggestions ({len(candidates)}) — click one to load its boundary:")
    for i, cand in enumerate(candidates):
        tag = f" · {cand.kind}" if cand.kind else ""
        poly_badge = " ✓ polygon" if cand.has_relation else " · bbox only"
        label = f"**{cand.short_name}**{tag}{poly_badge}"
        if cand.context:
            label += f"  \n_{cand.context}_"
        if st.button(
            label,
            key=f"_aoi_suggest_{i}_{cand.osm_id or hash(cand.display_name)}",
            use_container_width=True,
        ):
            _fetch_boundary_for(query=cand.display_name, hint=cand)


# ---------------------------------------------------------------------------
# Downstream contract
# ---------------------------------------------------------------------------


def aoi_to_boundary_gdf(aoi: dict) -> gpd.GeoDataFrame:
    """Convert a confirmed AOI dict into a one-row boundary GeoDataFrame (EPSG:4326).

    Carries through country metadata (country name, ISO 3166-1 alpha-2 code)
    and a composed ``location_query`` trailer so downstream consumers such
    as :py:meth:`DataFetcher._fetch_population_hdx` can resolve the ISO3
    country code without a separate reverse-geocode call.
    """
    geom = shape(aoi["geometry_geojson"])
    name = aoi.get("name", "AOI")
    country = str(aoi.get("country") or "").strip()
    country_code = str(aoi.get("country_code") or "").strip().upper()

    # "Name, Country" gives HDX's ISO3 fuzzy matcher a high-signal string
    # even when explicit country/country_code columns are absent (manual
    # drawn polygons, legacy AOIs persisted without the new fields, etc.).
    location_query = f"{name}, {country}" if country else name

    row: dict = {
        "name": [name],
        "location_query": [location_query],
        "country": [country],
        "country_code": [country_code],
    }
    gdf = gpd.GeoDataFrame(row, geometry=[geom], crs="EPSG:4326")
    gdf.attrs["source"] = f"aoi_{aoi.get('source', 'user')}"
    return gdf
