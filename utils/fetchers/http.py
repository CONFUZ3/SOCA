"""HTTP plumbing: shared Session, retry loop, thread-safe Nominatim limiter."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import requests

from .constants import _MAX_RETRIES, _RETRY_BASE_DELAY, _USER_AGENT, NOMINATIM_URL
from .errors import DataFetchError

logger = logging.getLogger(__name__)


class NominatimRateLimiter:
    """Thread-safe enforcer of Nominatim's 1 req/s ToS limit.

    The previous implementation read/wrote ``_last_nominatim_call`` without a
    lock, so concurrent threads could both observe "enough time has passed"
    before either updated the timestamp — producing a burst of requests that
    violates the 1 req/s policy. This wrapper makes the check-sleep-update
    sequence atomic.
    """

    def __init__(self, min_interval_sec: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._last_call: float = 0.0
        self._min_interval = float(min_interval_sec)

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


def make_request(
    url: str,
    params: Optional[dict] = None,
    method: str = "GET",
    timeout: int = 30,
) -> requests.Response:
    """Send an HTTP request with exponential-backoff retries.

    Retries on: connect errors, read timeouts, 429 (honoring Retry-After),
    403/408/502/503/504 (public mirrors like Overpass return 403 to signal
    "go away for a bit", so treat as transient), and all 5xx server errors.
    Fails fast on other 4xx.
    """
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "en",
    }

    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        delay = _RETRY_BASE_DELAY * (2 ** attempt)  # 1, 2, 4 s
        try:
            if method.upper() == "POST":
                resp = requests.post(
                    url, data=params, headers=headers, timeout=timeout
                )
            else:
                resp = requests.get(
                    url, params=params, headers=headers, timeout=timeout
                )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay))
                logger.warning(
                    f"Rate limited by {url}; sleeping {retry_after}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            logger.warning(
                f"Timeout on attempt {attempt + 1}/{_MAX_RETRIES} to {url}; "
                f"retrying in {delay}s"
            )
            time.sleep(delay)

        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            logger.warning(
                f"Connection error on attempt {attempt + 1}/{_MAX_RETRIES} "
                f"to {url}; retrying in {delay}s"
            )
            time.sleep(delay)

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            # 403/408/502/503/504 are treated as transient — public Overpass
            # and Nominatim mirrors return 403 to throttle, not to auth-deny.
            if status in (403, 408, 502, 503, 504) or status >= 500:
                last_exc = exc
                logger.warning(
                    f"HTTP {status} on attempt {attempt + 1}/{_MAX_RETRIES} "
                    f"to {url}; retrying in {delay}s"
                )
                time.sleep(delay)
            else:
                raise DataFetchError(
                    f"HTTP {status} error from {url}: {exc}"
                ) from exc

        except Exception as exc:
            raise DataFetchError(
                f"Unexpected error requesting {url}: {exc}"
            ) from exc

    raise DataFetchError(
        f"All {_MAX_RETRIES} attempts failed for {url}. Last error: {last_exc}"
    )


# Shared process-wide limiter. Used by every Nominatim caller across threads.
nominatim_limiter = NominatimRateLimiter(min_interval_sec=1.0)


def nominatim_get(path: str, params: dict, timeout: int = 20) -> requests.Response:
    """Rate-limited GET against the Nominatim public API."""
    nominatim_limiter.wait()
    return make_request(NOMINATIM_URL + path, params=params, timeout=timeout)
