"""
Thread-local bridge giving ADK tool functions access to Streamlit session objects.

Streamlit session state is thread-local and cannot be passed as function
arguments into ADK tool callables (they would not be JSON-serialisable and
are not picklable).  Before every Runner.run() call, SOCAAgent calls
set_current_context() to stash references in a threading.local().  Tool
functions then call get_data(), get_problem_state(), etc. to read/write the
live Streamlit objects on the same thread.

This is safe because Streamlit processes one user interaction per thread and
Runner.run() is synchronous – all tool calls happen on the calling thread.
"""

import threading
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_bridge = threading.local()


def set_current_context(
    data: Dict[str, Any],
    problem_state: Dict[str, Any],
    problem_registry: Any,
    generated_sites_count: int = 100,
    generated_sites_seed: Optional[int] = None,
) -> None:
    """Store live session references for the current thread.

    Call this immediately before Runner.run() so that tool functions executed
    in that call can access the correct Streamlit session objects.
    """
    _bridge.data = data
    _bridge.problem_state = problem_state
    _bridge.registry = problem_registry
    _bridge.generated_sites_count = generated_sites_count
    _bridge.generated_sites_seed = generated_sites_seed
    logger.debug("state_bridge: context set (data keys=%s)", list(data.keys()))


def get_data() -> Dict[str, Any]:
    """Return the GeoDataFrame data dict for the current session."""
    return getattr(_bridge, "data", {})


def get_problem_state() -> Dict[str, Any]:
    """Return the full problem_state dict for the current session."""
    return getattr(_bridge, "problem_state", {})


def get_problem_registry() -> Any:
    """Return the ProblemRegistry for the current session."""
    return getattr(_bridge, "registry", None)


def get_generated_sites_count() -> int:
    return getattr(_bridge, "generated_sites_count", 100)


def get_generated_sites_seed() -> Optional[int]:
    return getattr(_bridge, "generated_sites_seed", None)
