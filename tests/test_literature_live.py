"""Live literature retrieval (Europe PMC), exercised against a recorded response.

The network is replaced with `data/fixtures/europepmc_sample.json`, so the parser,
the stance heuristic and the fallback behaviour are all covered offline.
"""

import json

import requests

from src.tools import literature


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fixture():
    with open("data/fixtures/europepmc_sample.json", encoding="utf-8") as handle:
        return json.load(handle)


def _reset(monkeypatch):
    monkeypatch.setattr(literature, "_LIVE_DISABLED", False)


def test_live_search_parses_results(monkeypatch):
    _reset(monkeypatch)
    # search_live imports requests lazily, so patch the module itself
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(_fixture()))

    items = literature.search_live("SHP099", "SHP2")

    assert len(items) == 2
    assert items[0].source == "literature"
    assert items[0].polarity == 1                       # "potent ... inhibits"
    assert items[1].polarity == -1                      # "did not inhibit ... no significant"
    assert items[0].meta["study_type"] == "live-europepmc"
    assert "SHP099" in items[0].claim


def test_live_search_disables_itself_after_a_failure(monkeypatch):
    _reset(monkeypatch)

    def boom(*args, **kwargs):
        raise ConnectionError("blocked by proxy")

    monkeypatch.setattr(requests, "get", boom)

    assert literature.search_live("SHP099", "SHP2") == []
    assert literature._LIVE_DISABLED is True
    # a second call must not even try
    assert literature.search_live("SHP099", "SHP2") == []


def test_local_corpus_still_answers_when_live_is_off(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setattr(literature, "_LIVE_DISABLED", True)
    items = literature.search("SHP099", "SHP2", "KRAS-mutant NSCLC")
    assert items and all(item.source == "literature" for item in items)
