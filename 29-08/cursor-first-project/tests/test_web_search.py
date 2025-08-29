"""Tests for the `search_web` function in `web_search.py`."""

from typing import Iterator, List

import pytest

import web_search


def test_search_web_raises_on_empty_query() -> None:
    """An empty query should raise a ValueError."""
    with pytest.raises(ValueError):
        web_search.search_web("")


class DummyDDGS:
    """A minimal stand-in for DDGS used to control test inputs."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, *_args, **_kwargs) -> Iterator[dict]:
        """Yield simulated items with null/missing fields."""
        items: List[dict] = [
            {"title": None, "href": "https://example.com/a", "body": None},
            {"title": "Has URL key", "url": "https://example.com/b", "snippet": None},
            {"title": "Missing URL entirely", "body": "desc"},  # should be skipped
            {"title": "Has body", "href": "https://example.com/c", "body": "desc"},
        ]
        yield from items


def test_search_web_handles_null_values_and_missing_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure items with null/missing fields are normalized or skipped as expected."""
    # Monkeypatch the DDGS class used inside the module
    monkeypatch.setattr(web_search, "DDGS", DummyDDGS)

    results = web_search.search_web("test", max_results=5)

    # One item is skipped due to missing URL; expect 3 normalized results
    assert len(results) == 3

    # Validate normalization and presence of keys
    for res in results:
        assert set(res.keys()) == {"title", "url", "snippet"}
        assert isinstance(res["url"], str) and res["url"]
