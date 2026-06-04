"""Global SOCA configuration.

Single source of truth for model IDs, solver time budgets, Gurobi tuning
parameters, and network-fetch limits.  Values can be overridden from the
environment so ops can tune a deployment without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Application configuration (read from env when available)."""

    # --- API keys ----------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # --- LLM ---------------------------------------------------------------
    # Default to the current public ID.  Override via GEMINI_MODEL env var
    # for experimentation (e.g. "gemini-2.0-flash-exp").
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    MAX_TOKENS: int = _env_int("GEMINI_MAX_TOKENS", 4096)
    GEMINI_TEMPERATURE: float = _env_float("GEMINI_TEMPERATURE", 0.5)

    # --- Solver time budgets (seconds) ------------------------------------
    # MIP (Gurobi/CBC) time limit for the main optimisation call.
    SOLVER_MIP_TIME_LIMIT: float = _env_float("SOCA_MIP_TIME_LIMIT", 120.0)
    # Genetic-algorithm fallback budget invoked when MIP times out.
    SOLVER_GA_TIME_LIMIT: float = _env_float("SOCA_GA_TIME_LIMIT", 120.0)
    # Hard wall-clock guard around the whole solver.solve() call.  Includes
    # Python-side model building (which is NOT counted by Gurobi's TimeLimit).
    SOLVER_WALL_CLOCK_TIMEOUT: float = _env_float(
        "SOCA_SOLVER_WALL_CLOCK", 420.0
    )
    # Gurobi optimality gap tolerance (1 % by default).
    MIP_GAP: float = _env_float("SOCA_MIP_GAP", 0.01)
    # Automatically switch to the GA path when problem size exceeds this
    # many (demand × candidate) pairs -- MIP model building becomes painful
    # before Gurobi's TimeLimit can kick in.
    MIP_MODEL_SIZE_LIMIT: int = _env_int("SOCA_MIP_MODEL_SIZE_LIMIT", 300_000)

    # --- Gurobi tuning -----------------------------------------------------
    GUROBI_PRESOLVE: int = _env_int("SOCA_GUROBI_PRESOLVE", 2)
    GUROBI_CUTS: int = _env_int("SOCA_GUROBI_CUTS", 2)
    GUROBI_HEURISTICS: float = _env_float("SOCA_GUROBI_HEURISTICS", 0.05)
    # MIPFocus=1 prioritises finding feasible solutions quickly; ideal when
    # the wall-clock budget is tight.
    GUROBI_MIP_FOCUS: int = _env_int("SOCA_GUROBI_MIP_FOCUS", 1)
    GUROBI_THREADS: int = _env_int("SOCA_GUROBI_THREADS", 0)  # 0 = auto

    # --- Network / OSMnx limits -------------------------------------------
    # Max wait for NetworkManager.get_graph() before falling back to
    # geodesic distance.
    NETWORK_FETCH_TIMEOUT: float = _env_float("SOCA_NETWORK_FETCH_TIMEOUT", 90.0)
    # Max wall-clock budget for the per-destination Dijkstra sweep in
    # DistanceCalculator._network_distance().
    NETWORK_DIJKSTRA_BUDGET_SECONDS: float = _env_float(
        "SOCA_DIJKSTRA_BUDGET", 300.0
    )
    # osmnx requests_timeout (per HTTP call, not total).
    OSMNX_REQUESTS_TIMEOUT: int = _env_int("SOCA_OSMNX_REQUESTS_TIMEOUT", 120)
    # Area-based cutoff: if the AOI exceeds this area (km^2), skip the
    # road-network fetch entirely and fall back to geodesic distance.
    # Empirically, Overpass downloads for AOIs above ~10 000 km^2 reliably
    # take 1–7 minutes and dominate the end-to-end solve time. Users can
    # opt in to a full network fetch by setting strict_network=True.
    NETWORK_FETCH_MAX_AREA_KM2: float = _env_float(
        "SOCA_NETWORK_FETCH_MAX_AREA_KM2", 10_000.0
    )
    # Auto-downgrade distance metric to "euclidean" when boundary area exceeds
    # this threshold (km^2). Keeps network distance the default for urban
    # AOIs while avoiding multi-minute Overpass downloads on regional ones.
    NETWORK_AUTO_EUCLIDEAN_AREA_KM2: float = _env_float(
        "SOCA_NETWORK_AUTO_EUCLIDEAN_AREA_KM2", 2_000.0
    )
    # Default driving speed (km/h) used by sensitivity / repro logging when
    # converting network length (m) to travel time. Configurable so callers
    # can model walk vs. drive scenarios without editing code.
    DEFAULT_DRIVE_SPEED_KMH: float = _env_float("SOCA_DEFAULT_DRIVE_SPEED_KMH", 30.0)

    # --- Candidate generation ---------------------------------------------
    # Hard cap on candidate sites returned by generate_candidate_sites. When
    # the road network has more nodes than this we KDTree-thin them down.
    MAX_CANDIDATE_SITES: int = _env_int("SOCA_MAX_CANDIDATE_SITES", 500)
    # Initial minimum inter-point distance (metres) used by KDTree thinning;
    # the threshold doubles each iteration until the count is under
    # MAX_CANDIDATE_SITES.
    CANDIDATE_THINNING_MIN_DIST_M: float = _env_float(
        "SOCA_CANDIDATE_THINNING_MIN_DIST_M", 200.0
    )

    # --- Reproducibility --------------------------------------------------
    # Global random seed used by candidate generation, synthetic-population
    # fallback, and any other stochastic step that needs to be replayable.
    RANDOM_SEED: int = _env_int("SOCA_RANDOM_SEED", 42)
    # Directory for run logs (one JSON per optimization).
    RUNS_DIR: Path = Path(os.environ.get("SOCA_RUNS_DIR", "runs")).resolve() \
        if os.environ.get("SOCA_RUNS_DIR") else Path(__file__).parent.parent / "runs"

    # --- CRS ---------------------------------------------------------------
    CRS_STANDARD: str = "EPSG:4326"  # WGS84 for lat/lon
    CRS_PROJECTED: str = "EPSG:3857"  # Web Mercator for fallback distances

    # --- File upload limits -----------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = _env_int("SOCA_MAX_UPLOAD_SIZE_MB", 50)
    ALLOWED_EXTENSIONS: list = [".geojson", ".json", ".csv", ".shp", ".zip"]

    # --- ADK ---------------------------------------------------------------
    ADK_APP_NAME: str = "soca"

    # --- Academic ----------------------------------------------------------
    CITATION_STYLE: str = "APA"

    # --- Paths -------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    TEMP_DIR: Path = BASE_DIR / "temp"
    DOCS_DIR: Path = BASE_DIR / "docs"
    TEST_DATA_DIR: Path = BASE_DIR / "tests" / "test_data"

    # --- Legacy aliases ---------------------------------------------------
    # SOLVER_TIME_LIMIT was exported to old app.py; keep as an alias so
    # legacy Streamlit imports still resolve.  New code MUST use
    # SOLVER_MIP_TIME_LIMIT / SOLVER_WALL_CLOCK_TIMEOUT.
    SOLVER_TIME_LIMIT: float = SOLVER_MIP_TIME_LIMIT

    @classmethod
    def validate(cls) -> bool:
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Configure it in .env or the environment."
            )
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return True

    @classmethod
    def check_gurobi(cls) -> bool:
        try:
            import gurobipy  # noqa: F401
            return True
        except ImportError:
            return False


settings = Settings()
