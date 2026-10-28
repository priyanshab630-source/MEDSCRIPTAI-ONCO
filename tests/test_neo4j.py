"""Neo4j backend and loader, exercised with a fake driver (no database needed)."""

import json

import pytest

from scripts.load_neo4j import CONSTRAINT, UPSERT, load
from src.core.schemas import EvidenceItem


class FakeResult:
    """Behaves like a Neo4j Result: iterable, plus single()."""

    def __init__(self, records=None):
        self._records = records or []

    def single(self):
        return self._records[0] if self._records else None

    def __iter__(self):
        return iter(self._records)


class FakeSession:
    def __init__(self, records=None):
        self.calls = []
        self.records = records or []

    def run(self, query, **params):
        self.calls.append((query, params))
        return FakeResult(self.records)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


def test_loader_creates_constraint_then_upserts():
    triples = [
        {"subject": "SHP099", "relation": "INHIBITS", "object": "SHP2",
         "polarity": 1, "source": "ChemMedChem 2016"},
        {"subject": "RMC-4550", "relation": "FAILED_IN", "object": "PHASE1_SOLID_TUMOUR",
         "polarity": -1, "source": "ClinicalTrials 2021"},
    ]
    session = FakeSession(records=[{"edges": 2}])
    stats = load(session, triples)

    assert session.calls[0][0] == CONSTRAINT
    query, params = session.calls[1]
    assert query == UPSERT
    assert len(params["rows"]) == 2
    assert params["rows"][1]["polarity"] == -1
    assert stats == {"rows": 2, "edges": 2}


def test_loader_reads_the_shipped_snapshot(tmp_path):
    path = tmp_path / "triples.json"
    path.write_text(json.dumps([
        {"subject": "A", "relation": "BINDS", "object": "B", "polarity": 1, "source": "x"}
    ]))

    from scripts.load_neo4j import load_file

    session = FakeSession(records=[{"edges": 1}])
    stats = load_file(FakeDriver(session), path)
    assert stats["rows"] == 1


def test_neo4j_backend_parses_paths_into_evidence():
    from src.tools.knowledge_graph import Neo4jKG

    records = [{
        "rels": ["INHIBITS"],
        "nodes": ["SHP099", "SHP2"],
        "polarities": [1],
    }, {
        "rels": ["SHOWED_RESISTANCE_IN", "ASSOCIATED_WITH"],
        "nodes": ["SHP099", "KRAS_MUTANT_CRC", "SHP2"],
        "polarities": [-1, 1],
    }]
    session = FakeSession(records=records)
    kg = Neo4jKG(driver=FakeDriver(session))

    evidence = kg.evidence("SHP099", "SHP2")
    assert len(evidence) == 2
    assert all(isinstance(item, EvidenceItem) for item in evidence)
    assert evidence[0].polarity == 1
    assert evidence[1].polarity == -1            # any negative hop flips the claim
    assert evidence[1].meta["hops"] == 2
    assert "inhibits" in evidence[0].claim.lower()


def test_neo4j_driver_is_optional():
    pytest.importorskip("neo4j")
    assert hasattr(EvidenceItem, "to_dict")
