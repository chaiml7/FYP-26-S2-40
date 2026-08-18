from unittest.mock import MagicMock

from backend.services.guest_preview_service import (
    GUEST_PREVIEW_LIMIT,
    guest_previews_exhausted,
    record_guest_preview,
    remaining_guest_previews,
)


def _fake_request(session=None):
    request = MagicMock()
    request.session = session if session is not None else {}
    return request


def test_remaining_previews_defaults_to_limit_for_new_visitor():
    request = _fake_request()

    assert remaining_guest_previews(request) == GUEST_PREVIEW_LIMIT
    assert not guest_previews_exhausted(request)


def test_record_guest_preview_decrements_remaining_count():
    request = _fake_request()

    assert record_guest_preview(request) == GUEST_PREVIEW_LIMIT - 1
    assert record_guest_preview(request) == GUEST_PREVIEW_LIMIT - 2
    assert remaining_guest_previews(request) == GUEST_PREVIEW_LIMIT - 2


def test_guest_previews_exhausted_after_limit_reached():
    request = _fake_request()

    for _ in range(GUEST_PREVIEW_LIMIT):
        record_guest_preview(request)

    assert remaining_guest_previews(request) == 0
    assert guest_previews_exhausted(request)


def test_counter_resets_on_a_new_day():
    request = _fake_request()
    for _ in range(GUEST_PREVIEW_LIMIT):
        record_guest_preview(request)
    assert guest_previews_exhausted(request)

    request.session["guest_preview"]["date"] = "2000-01-01"

    assert remaining_guest_previews(request) == GUEST_PREVIEW_LIMIT
    assert not guest_previews_exhausted(request)
