"""Reproducibility logger for optimization runs.

Writes a JSON record per ``confirm_optimization`` call to the configured
``RUNS_DIR`` so any past optimization can be inspected after the fact: the
inputs that were used, the seed, the solver and parameters, and the result.

Replay is intentionally a stub. True bit-exact replay would require freezing
upstream data (HDX, Overture, OSM) which we don't control. Snapshot-based
replay (writing the actual GeoDataFrames + graph alongside the JSON) is a
documented future extension.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

logger = logging.getLogger(__name__)


def get_seed() -> int:
    """Return the global random seed used by stochastic steps.

    Reads ``SOCA_RANDOM_SEED`` from the environment (override) or
    ``config.settings.RANDOM_SEED`` (default 42).
    """
    env = os.environ.get("SOCA_RANDOM_SEED")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        from config.settings import settings
        return int(settings.RANDOM_SEED)
    except Exception:
        return 42


def _runs_dir() -> Path:
    try:
        from config.settings import settings
        d = Path(settings.RUNS_DIR)
    except Exception:
        d = Path("runs").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_safe(obj: Any) -> Any:
    """Best-effort coercion to JSON-serialisable primitives.

    Falls back to ``str(obj)`` for unknown types so logging is never blocked
    by an exotic value in the payload.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    try:
        # numpy scalars / arrays
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


class ReproducibilityLogger:
    """Writes one JSON file per optimization run to ``RUNS_DIR``."""

    def log_run(self, payload: Dict[str, Any]) -> Path:
        """Persist *payload* to ``RUNS_DIR/<uuid>.json`` and return its path.

        Appends an ISO-8601 timestamp if missing. Never raises — write
        failures are logged at WARNING and the original requested path is
        returned unchanged for the caller's audit trail.
        """
        run_id = payload.get("run_id") or str(uuid.uuid4())
        payload = dict(payload)
        payload["run_id"] = run_id
        payload.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        path = _runs_dir() / f"{run_id}.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(payload), f, indent=2)
            logger.info("ReproducibilityLogger: wrote %s", path)
        except Exception as exc:
            logger.warning("ReproducibilityLogger: write failed (%s)", exc)
        return path


def replay(run_path: Path) -> NoReturn:
    """Stub replay — raises ``NotImplementedError``.

    Bit-exact replay would require deterministic upstream fetches (HDX,
    Overture, OSM) which we don't control. A future snapshot mode could
    write the actual demand / candidates / graph alongside the JSON.
    """
    raise NotImplementedError(
        "Replay is not implemented. Logged data is at "
        f"{run_path}; a future snapshot-based replay mode is planned. "
        "TODO: persist demand_gdf / candidates_gdf / graph alongside the "
        "JSON to enable deterministic re-runs."
    )


def build_run_payload(
    *,
    boundary_polygon=None,
    demand_gdf=None,
    candidates_gdf=None,
    distance_method: Optional[str] = None,
    solver: Optional[str] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    objective_value: Optional[float] = None,
    selected_facility_ids: Optional[list] = None,
    random_seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical run payload from solver inputs/outputs.

    Pulls ``data_source`` columns when present so the log records *which*
    fetcher path produced demand and candidates. ``boundary_wkt`` is captured
    at WKT-string fidelity.
    """
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "boundary_wkt": None,
        "demand_source": None,
        "demand_n_points": 0,
        "candidates_source": None,
        "candidates_n": 0,
        "distance_method": distance_method,
        "solver": solver,
        "solver_params": solver_params or {},
        "random_seed": random_seed if random_seed is not None else get_seed(),
        "objective_value": objective_value,
        "selected_facility_ids": list(selected_facility_ids or []),
    }

    try:
        if boundary_polygon is not None:
            payload["boundary_wkt"] = getattr(boundary_polygon, "wkt", None)
    except Exception:
        pass

    try:
        if demand_gdf is not None and len(demand_gdf) > 0:
            payload["demand_n_points"] = int(len(demand_gdf))
            if "data_source" in demand_gdf.columns:
                payload["demand_source"] = str(demand_gdf["data_source"].iloc[0])
    except Exception:
        pass

    try:
        if candidates_gdf is not None and len(candidates_gdf) > 0:
            payload["candidates_n"] = int(len(candidates_gdf))
            if "data_source" in candidates_gdf.columns:
                payload["candidates_source"] = str(candidates_gdf["data_source"].iloc[0])
    except Exception:
        pass

    if extra:
        payload["extra"] = extra
    return payload
