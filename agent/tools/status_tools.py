"""
ADK tool: get_data_status

Returns a snapshot of available datasets and current problem state so the
agent can answer user questions like "what data is loaded?" without needing
to reconstruct this from conversation history.
"""

import logging
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from .state_bridge import get_data, get_problem_state

logger = logging.getLogger(__name__)


def get_data_status(
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Return the current session data availability and staged parameters.

    Use this to check what datasets are loaded, what optimization parameters
    are currently set, and whether a solution already exists.

    Returns:
        dict with keys:
          datasets (list[dict]): each has name, num_features, geometry_type, source.
              POI datasets additionally include `amenity_subtype_counts`
              (dict of subtype → count, e.g. {"primary_school": 12,
              "school": 30, "preschool": 4}), `amenity_subtype_total`,
              and `amenity_subtype_unique` so the agent can answer
              breakdown questions without a separate query tool.
          problem_type (str | None): currently identified problem type
          parameters (dict): current optimization parameters
          pending_optimization (bool): True if stage_optimization was called
          solution_available (bool): True if a solution has been computed
          solution_status (str | None): "optimal", "feasible", or "error"
    """
    data_store = get_data()
    ps = get_problem_state()

    dataset_summaries = []
    for name, gdf in data_store.items():
        try:
            geom_type = (
                gdf.geometry.type.unique()[0] if len(gdf) > 0 else "Unknown"
            )
        except Exception:
            geom_type = "Unknown"
        summary = {
            "name": name,
            "num_features": len(gdf),
            "geometry_type": geom_type,
            "source": gdf.attrs.get("source", "uploaded"),
        }
        # For POI datasets, expose the exact subtype breakdown so the agent
        # can answer questions like "how many primary_school vs school?"
        # without needing a separate query tool.
        if "amenity" in gdf.columns and len(gdf) > 0:
            try:
                counts = (
                    gdf["amenity"].astype(str).value_counts().to_dict()
                )
                summary["amenity_subtype_counts"] = {
                    str(k): int(v) for k, v in counts.items()
                }
                summary["amenity_subtype_total"] = int(sum(counts.values()))
                summary["amenity_subtype_unique"] = len(counts)
            except Exception as exc:
                logger.warning("amenity breakdown failed for %s: %s", name, exc)
        dataset_summaries.append(summary)

    solution = ps.get("solution")
    pending = (
        (tool_context.state.get("pending_optimization") is not None)
        if tool_context is not None
        else False
    )

    return {
        "datasets": dataset_summaries,
        "problem_type": ps.get("problem_type"),
        "parameters": ps.get("parameters", {}),
        "pending_optimization": pending,
        "solution_available": solution is not None,
        "solution_status": solution.get("status") if solution else None,
    }
