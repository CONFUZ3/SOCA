"""Unit tests for utils.activity_log — ring buffer, event formatting, timed ctx."""

import time

import pytest

from utils import activity_log as al


@pytest.fixture(autouse=True)
def _clean_fallback():
    al._FALLBACK.clear()
    yield
    al._FALLBACK.clear()


def test_log_event_appends_to_fallback_when_no_streamlit_ctx():
    evt = al.log_event("boundary.fetch", "ok", "Lima", source="Overture", duration_ms=123)
    events = al.get_events()
    assert len(events) == 1
    assert events[0] is evt
    assert evt.stage == "boundary.fetch"
    assert evt.status == "ok"
    assert evt.source == "Overture"
    assert evt.duration_ms == 123


def test_log_event_ring_buffer_cap():
    for i in range(al._MAX_EVENTS + 20):
        al.log_event("test", "info", f"event {i}")
    events = al.get_events()
    assert len(events) == al._MAX_EVENTS
    # Most recent survives; oldest gets evicted.
    assert events[-1].detail == f"event {al._MAX_EVENTS + 19}"
    assert events[0].detail == f"event {20}"


def test_event_format_includes_glyph_source_and_duration():
    evt = al.log_event("boundary.fetch", "ok", "polygon, 138 vertices",
                       source="Overpass", duration_ms=820)
    line = evt.format()
    assert "✓" in line
    assert "boundary.fetch" in line
    assert "Overpass" in line
    assert "820" in line
    assert "polygon, 138 vertices" in line


def test_event_format_fail_uses_cross_glyph():
    evt = al.log_event("boundary.fetch", "fail", "HTTP 429", source="Nominatim")
    assert "✗" in evt.format()


def test_has_errors_reflects_fail_status():
    al.log_event("a", "ok")
    assert not al.has_errors()
    al.log_event("b", "fail", "boom")
    assert al.has_errors()


def test_timed_context_emits_try_then_ok():
    with al.timed("boundary.fetch", source="Overture", detail="Lima"):
        pass
    events = al.get_events()
    assert [e.status for e in events] == ["try", "ok"]
    assert events[1].duration_ms is not None and events[1].duration_ms >= 0


def test_timed_context_emits_fail_on_exception():
    with pytest.raises(RuntimeError):
        with al.timed("boundary.fetch", source="Overpass", detail="XYZ"):
            raise RuntimeError("nope")
    events = al.get_events()
    assert [e.status for e in events] == ["try", "fail"]
    assert "RuntimeError" in events[1].detail


def test_timed_detail_can_be_updated_mid_block():
    with al.timed("boundary.fetch", source="OSM", detail="start") as t:
        t.detail = "updated"
    events = al.get_events()
    # 'try' sees the initial detail; 'ok' sees the updated one.
    assert events[0].detail == "start"
    assert events[1].detail == "updated"


def test_clear_events_empties_buffer():
    al.log_event("a", "ok")
    al.log_event("b", "ok")
    al.clear_events()
    assert al.get_events() == []
