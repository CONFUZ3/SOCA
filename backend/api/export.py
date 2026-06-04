"""Export endpoints — GeoJSON / CSV / PDF."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from backend.deps import resolve_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["export"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_solution(record: Dict[str, Any]) -> Dict[str, Any]:
    ps = record.get("problem_state") or {}
    sol = ps.get("solution")
    if not sol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No solution available. Run an optimisation first.",
        )
    return sol


def _candidate_gdf(record: Dict[str, Any]):
    ps = record.get("problem_state") or {}
    data = ps.get("data") or {}
    for key in ("candidate_sites", "generated_candidates"):
        if key in data:
            return data[key]
    for name in data:
        n = name.lower()
        if "candidate" in n or "facilit" in n or "generated" in n:
            return data[name]
    return None


def _demand_gdf(record: Dict[str, Any]):
    ps = record.get("problem_state") or {}
    data = ps.get("data") or {}
    for name in data:
        n = name.lower()
        if n.startswith("demand") or "population" in n:
            return data[name]
    return None


def _to_4326(gdf):
    if gdf is None:
        return None
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            return gdf.to_crs("EPSG:4326")
    except Exception:
        pass
    return gdf


def _first(row: Any, keys: tuple) -> Any:
    for k in keys:
        try:
            val = row[k]
        except Exception:
            continue
        if val is not None:
            return val
    return None


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# Tools whose turns produce solution-narrative text worth carrying into the
# PDF report (the LLM's written analysis of the current solution).
_NARRATION_TOOLS = frozenset(
    {"confirm_optimization", "run_sensitivity_analysis", "analyze_existing_facilities"}
)


def _analysis_narratives(record: Dict[str, Any]) -> List[str]:
    """Return the LLM's written analysis tied to the current solution.

    Walks the chat history back to the most recent ``confirm_optimization``
    turn and returns that assistant message plus any later solution-related
    narration (sensitivity / facility analysis). Falls back to the last
    non-empty assistant message when no optimisation turn is found.
    """
    messages = record.get("messages") or []

    start = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and "confirm_optimization" in (
            m.get("tool_calls") or []
        ):
            start = i
            break

    if start is None:
        for m in reversed(messages):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                return [m["content"].strip()]
        return []

    out: List[str] = []
    for idx, m in enumerate(messages[start:], start=start):
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if idx == start or set(m.get("tool_calls") or []) & _NARRATION_TOOLS:
            out.append(content)
    return out


# ---------------------------------------------------------------------------
# GET /api/export/geojson
# ---------------------------------------------------------------------------


@router.get("/geojson")
def export_geojson(ctx=Depends(resolve_session)) -> Response:
    _, record = ctx
    sol = _require_solution(record)
    ps = record.get("problem_state") or {}

    cand_gdf = _to_4326(_candidate_gdf(record))
    demand_gdf = _to_4326(_demand_gdf(record))
    selected: List[int] = [int(i) for i in (sol.get("selected_facilities") or [])]
    assignments: Dict[Any, Any] = sol.get("assignments") or {}

    features: List[Dict[str, Any]] = []

    # Selected facilities
    if cand_gdf is not None and selected:
        for i in selected:
            if i < 0 or i >= len(cand_gdf):
                continue
            row = cand_gdf.iloc[i]
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            props: Dict[str, Any] = {
                "kind": "facility",
                "facility_idx": int(i),
            }
            for key in ("name", "id", "capacity", "cost", "label"):
                try:
                    val = row[key]
                    if val is not None:
                        props[key] = val if not hasattr(val, "item") else val.item()
                except Exception:
                    continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": json.loads(
                        cand_gdf.iloc[[i]].geometry.to_json()
                    )["features"][0]["geometry"],
                    "properties": props,
                }
            )

    # Assignment lines
    if assignments and cand_gdf is not None and demand_gdf is not None:
        for d_idx, f_idx in assignments.items():
            try:
                di, fi = int(d_idx), int(f_idx)
            except Exception:
                continue
            if di < 0 or di >= len(demand_gdf):
                continue
            if fi < 0 or fi >= len(cand_gdf):
                continue
            d_pt = demand_gdf.geometry.iloc[di]
            f_pt = cand_gdf.geometry.iloc[fi]
            if d_pt is None or f_pt is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [d_pt.x, d_pt.y],
                            [f_pt.x, f_pt.y],
                        ],
                    },
                    "properties": {
                        "kind": "assignment",
                        "demand_idx": di,
                        "facility_idx": fi,
                    },
                }
            )

    fc = {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "problem_type": ps.get("problem_type"),
            "parameters": ps.get("parameters") or {},
            "objective_value": sol.get("objective_value"),
            "solver": sol.get("solver"),
            "status": sol.get("status"),
            "distance_metric_used": sol.get("distance_metric_used"),
            "n_selected": len(selected),
        },
        "features": features,
    }

    body = json.dumps(fc).encode("utf-8")
    fname = f"soca-solution-{_timestamp_slug()}.geojson"
    return Response(
        content=body,
        media_type="application/geo+json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ---------------------------------------------------------------------------
# GET /api/export/csv — selected facilities as CSV
# ---------------------------------------------------------------------------


@router.get("/csv")
def export_csv(ctx=Depends(resolve_session)) -> Response:
    _, record = ctx
    sol = _require_solution(record)

    cand_gdf = _to_4326(_candidate_gdf(record))
    selected: List[int] = [int(i) for i in (sol.get("selected_facilities") or [])]
    if cand_gdf is None or not selected:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No selected facilities available to export.",
        )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "facility_idx",
            "name",
            "id",
            "lat",
            "lon",
            "capacity",
            "cost",
            "num_assigned_demands",
        ]
    )

    assignments = sol.get("assignments") or {}
    assigned_count: Dict[int, int] = {}
    for _, fi in assignments.items():
        try:
            key = int(fi)
            assigned_count[key] = assigned_count.get(key, 0) + 1
        except Exception:
            continue

    for i in selected:
        if i < 0 or i >= len(cand_gdf):
            continue
        row = cand_gdf.iloc[i]
        geom = row.geometry
        if geom is None or geom.is_empty:
            lon, lat = "", ""
        else:
            try:
                c = geom.centroid
                lon, lat = f"{c.x:.6f}", f"{c.y:.6f}"
            except Exception:
                lon, lat = "", ""
        def _get(keys):
            for k in keys:
                try:
                    v = row[k]
                    if v is not None:
                        return v
                except Exception:
                    continue
            return ""

        writer.writerow(
            [
                int(i),
                _get(("name", "label", "place_name")),
                _get(("id", "site_id", "osm_id")),
                lat,
                lon,
                _get(("capacity", "cap", "max_capacity")),
                _get(("cost", "fixed_cost", "facility_cost", "opening_cost")),
                assigned_count.get(int(i), 0),
            ]
        )

    body = buf.getvalue().encode("utf-8")
    fname = f"soca-facilities-{_timestamp_slug()}.csv"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ---------------------------------------------------------------------------
# GET /api/export/pdf — summary report
# ---------------------------------------------------------------------------


def _format_metric(key: str, value: Any) -> str:
    try:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            if abs(value) >= 1000:
                return f"{value:,.1f}"
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)
    except Exception:
        return str(value)


@router.get("/pdf")
def export_pdf(ctx=Depends(resolve_session)) -> Response:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "PDF export requires the 'reportlab' package. "
                "Install it with: pip install reportlab"
            ),
        ) from exc

    _, record = ctx
    sol = _require_solution(record)
    ps = record.get("problem_state") or {}
    aoi = ps.get("aoi") or {}
    params = ps.get("parameters") or {}
    metrics = sol.get("metrics") or {}
    warnings = list(sol.get("warnings") or [])
    selected = sol.get("selected_facilities") or []

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="SOCA Optimisation Report",
    )

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    mono = ParagraphStyle(
        "mono",
        parent=body,
        fontName="Courier",
        fontSize=9,
        leading=11,
    )
    md_styles = {
        "body": ParagraphStyle(
            "md_body", parent=body, fontSize=9.5, leading=13, spaceAfter=2
        ),
        "head": ParagraphStyle(
            "md_head",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            spaceBefore=4,
            spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "md_bullet",
            parent=body,
            fontSize=9.5,
            leading=13,
            leftIndent=14,
            bulletIndent=2,
            spaceAfter=1,
        ),
        "cell": ParagraphStyle(
            "md_cell", parent=body, fontSize=8.5, leading=11
        ),
    }

    elements: List[Any] = []
    elements.append(Paragraph("SOCA Optimisation Report", h1))
    elements.append(
        Paragraph(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            body,
        )
    )
    elements.append(Spacer(1, 0.2 * inch))

    # AOI
    elements.append(Paragraph("Area of interest", h2))
    aoi_rows = [
        ["Name", aoi.get("name") or "—"],
        ["Source", aoi.get("source") or "—"],
        [
            "Area (km²)",
            _format_metric("area", aoi.get("area_km2"))
            if aoi.get("area_km2") is not None
            else "—",
        ],
    ]
    elements.append(_kv_table(aoi_rows, Table, TableStyle, colors))
    elements.append(Spacer(1, 0.15 * inch))

    # Problem
    elements.append(Paragraph("Problem", h2))
    prob_rows = [
        ["Type", str(ps.get("problem_type") or "—")],
        ["Variant", str(params.get("variant") or "base")],
        ["Solver", str(sol.get("solver") or "—")],
        ["Status", str(sol.get("status") or "—")],
        [
            "Distance metric",
            str(sol.get("distance_metric_used") or params.get("distance_metric") or "—"),
        ],
    ]
    for k, v in params.items():
        if k == "variant":
            continue
        prob_rows.append([str(k), _format_metric(k, v)])
    elements.append(_kv_table(prob_rows, Table, TableStyle, colors))
    elements.append(Spacer(1, 0.15 * inch))

    # Result metrics
    elements.append(Paragraph("Result", h2))
    result_rows = [
        [
            "Objective value",
            _format_metric("objective_value", sol.get("objective_value"))
            if sol.get("objective_value") is not None
            else "—",
        ],
        ["Facilities selected", str(len(selected))],
        [
            "Solver time",
            f"{sol.get('solver_time_seconds'):.2f} s"
            if isinstance(sol.get("solver_time_seconds"), (int, float))
            else "—",
        ],
    ]
    for k, v in metrics.items():
        result_rows.append([str(k), _format_metric(k, v)])
    elements.append(_kv_table(result_rows, Table, TableStyle, colors))

    if warnings:
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph("Warnings", h2))
        for w in warnings:
            elements.append(Paragraph(f"• {w}", body))

    # AI analysis — the LLM's written narration of the current solution.
    narratives = _analysis_narratives(record)
    if narratives:
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("AI analysis", h2))
        for n_idx, narrative in enumerate(narratives):
            if n_idx > 0:
                elements.append(Spacer(1, 0.1 * inch))
            elements.extend(
                _render_markdown(
                    narrative,
                    md_styles,
                    Table,
                    TableStyle,
                    colors,
                    Paragraph,
                    Spacer,
                    inch,
                )
            )

    # Selected facilities table
    cand_gdf = _to_4326(_candidate_gdf(record))
    if cand_gdf is not None and selected:
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Selected facilities", h2))
        rows: List[List[str]] = [["#", "idx", "name", "lat", "lon", "capacity", "cost"]]
        for n, i in enumerate(selected, start=1):
            try:
                i = int(i)
            except Exception:
                continue
            if i < 0 or i >= len(cand_gdf):
                continue
            row = cand_gdf.iloc[i]
            geom = row.geometry
            try:
                c = geom.centroid
                lat_s, lon_s = f"{c.y:.5f}", f"{c.x:.5f}"
            except Exception:
                lat_s, lon_s = "—", "—"
            name = _first(row, ("name", "label", "place_name")) or "—"
            cap = _first(row, ("capacity", "cap", "max_capacity"))
            cost = _first(row, ("cost", "fixed_cost", "facility_cost"))
            rows.append(
                [
                    str(n),
                    str(i),
                    str(name)[:28],
                    lat_s,
                    lon_s,
                    _format_metric("capacity", cap) if cap is not None else "—",
                    _format_metric("cost", cost) if cost is not None else "—",
                ]
            )
        t = Table(rows, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        elements.append(t)

    doc.build(elements)
    fname = f"soca-report-{_timestamp_slug()}.pdf"
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _md_inline(text: str) -> str:
    """Convert inline markdown (bold / italic / code) to reportlab markup.

    XML special characters are escaped first so the LLM text can't break the
    Paragraph parser.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def _md_table(block, Table, TableStyle, colors, Paragraph, cell_style):
    """Build a reportlab Table from a block of markdown pipe-table lines."""
    rows = []
    for ln in block:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # Skip separator rows like |---|:--:|
        if cells and all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    data = [
        [Paragraph(_md_inline(c), cell_style) for c in (r + [""] * (width - len(r)))]
        for r in rows
    ]
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return t


def _render_markdown(md, styles, Table, TableStyle, colors, Paragraph, Spacer, inch):
    """Convert a markdown string into a list of reportlab flowables.

    Handles headings, bullet / numbered lists, pipe tables, and inline
    bold/italic/code. Unknown syntax falls back to a plain paragraph.
    """
    body = styles["body"]
    head = styles["head"]
    bullet = styles["bullet"]
    cell = styles["cell"]

    flow: List[Any] = []
    lines = md.splitlines()
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            flow.append(Spacer(1, 0.05 * inch))
            i += 1
            continue

        # Pipe-table block
        if stripped.startswith("|") and "|" in stripped[1:]:
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            tbl = _md_table(block, Table, TableStyle, colors, Paragraph, cell)
            if tbl is not None:
                flow.append(tbl)
                flow.append(Spacer(1, 0.06 * inch))
            continue

        hmatch = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if hmatch:
            flow.append(Paragraph(_md_inline(hmatch.group(2)), head))
            i += 1
            continue

        bmatch = re.match(r"^[-*•]\s+(.*)$", stripped)
        if bmatch:
            flow.append(Paragraph(_md_inline(bmatch.group(1)), bullet, bulletText="•"))
            i += 1
            continue

        nmatch = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if nmatch:
            flow.append(
                Paragraph(
                    _md_inline(nmatch.group(2)), bullet, bulletText=f"{nmatch.group(1)}."
                )
            )
            i += 1
            continue

        flow.append(Paragraph(_md_inline(stripped), body))
        i += 1
    return flow


def _kv_table(rows, Table, TableStyle, colors):
    t = Table(rows, colWidths=[120, 360], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9.5),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9.5),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.darkslategray),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    return t
