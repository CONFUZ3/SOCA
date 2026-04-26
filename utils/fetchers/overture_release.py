"""Overture Maps release discovery.

Resolves a concrete `YYYY-MM-DD.N` release string, used to pin the
`s3://overturemaps-us-west-2/release/<RELEASE>/...` parquet path.

Resolution order:
  1. ``OVERTURE_RELEASE`` environment variable (explicit override).
  2. STAC catalog at https://stac.overturemaps.org/catalog.json.
  3. Pinned fallback in ``constants._OVERTURE_DEFAULT_RELEASE``.

The STAC lookup is cached for the lifetime of the process; one network
call on first use, never repeated.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from .constants import _OVERTURE_DEFAULT_RELEASE

logger = logging.getLogger(__name__)

_STAC_CATALOG_URL = "https://stac.overturemaps.org/catalog.json"
_RELEASE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}\.\d+)")


@lru_cache(maxsize=1)
def get_overture_release() -> str:
    """Return the Overture release string to use for all DuckDB queries."""
    env_override = os.environ.get("OVERTURE_RELEASE", "").strip()
    if env_override:
        logger.info("Overture release from env: %s", env_override)
        return env_override

    try:
        from .http import make_request  # local import: avoid circular at package load

        resp = make_request(_STAC_CATALOG_URL, timeout=10)
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "Overture STAC discovery failed (%s); using pinned default %s",
            exc, _OVERTURE_DEFAULT_RELEASE,
        )
        return _OVERTURE_DEFAULT_RELEASE

    candidates: list[str] = []
    for link in data.get("links", []) or []:
        for field in ("href", "title", "id"):
            val = link.get(field)
            if isinstance(val, str):
                m = _RELEASE_RE.search(val)
                if m:
                    candidates.append(m.group(1))

    if not candidates:
        logger.warning(
            "Overture STAC catalog had no parseable releases; using %s",
            _OVERTURE_DEFAULT_RELEASE,
        )
        return _OVERTURE_DEFAULT_RELEASE

    # Lexicographic max is chronological because the format is YYYY-MM-DD.N.
    release = max(set(candidates))
    logger.info("Overture release from STAC: %s", release)
    return release
