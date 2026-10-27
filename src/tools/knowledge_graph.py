"""Biomedical knowledge graph access.

Two backends behind one interface:
  * LocalKG  - the JSON snapshot in data/kg_triples.json (default, offline)
  * Neo4jKG  - the real thing, used automatically when NEO4J_URI is set

Both return `EvidenceItem`s so downstream agents never care which one ran.
"""

import json

import config
from src.core.schemas import EvidenceItem


class LocalKG:
    """In-memory triples with a two-hop walk. Small but honest."""

    backend = "local-json"

    def __init__(self, path=None):
        path = path or (config.DATA_DIR / "kg_triples.json")
        with open(path, encoding="utf-8") as handle:
            self.triples = json.load(handle)

    def _edges_from(self, node):
        return [t for t in self.triples if t["subject"].lower() == node.lower()]

    def _edges_to(self, node):
        return [t for t in self.triples if t["object"].lower() == node.lower()]

    def paths(self, drug: str, target: str) -> list:
        """Direct edges plus one intermediate node between drug and target."""
        found = []

        for triple in self.triples:
            pair = {triple["subject"].lower(), triple["object"].lower()}
            if pair == {drug.lower(), target.lower()}:
                found.append([triple])

        intermediates = {t["object"] for t in self._edges_from(drug)} | \
                        {t["subject"] for t in self._edges_to(drug)}
        for middle in intermediates:
            if middle.lower() in (drug.lower(), target.lower()):
                continue
            first = [t for t in self.triples
                     if t["subject"].lower() == drug.lower()
                     and t["object"].lower() == middle.lower()]
            second = [t for t in self.triples
                      if {t["subject"].lower(), t["object"].lower()}
                      == {middle.lower(), target.lower()}]
            for edge_a in first[:1]:
                for edge_b in second[:1]:
                    found.append([edge_a, edge_b])
        return found

    def evidence(self, drug: str, target: str) -> list:
        items = []
        for path in self.paths(drug, target):
            polarity = 1
            if any(t.get("polarity", 1) < 0 for t in path):
                polarity = -1
            claims = " -> ".join(
                f"{t['subject']} {t['relation'].replace('_', ' ').lower()} {t['object']}"
                for t in path
            )
            weight = 0.55 if len(path) == 1 else 0.3
            items.append(EvidenceItem(
                source="knowledge_graph",
                claim=claims,
                polarity=polarity,
                weight=weight,
                meta={
                    "hops": len(path),
                    "sources": [t.get("source", "curated") for t in path],
                    "backend": self.backend,
                },
            ))
        return items


class Neo4jKG:
    """Same interface, backed by a real Neo4j instance."""

    backend = "neo4j"

    def __init__(self):
        from neo4j import GraphDatabase          # imported lazily on purpose

        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

    def evidence(self, drug: str, target: str) -> list:
        query = """
        MATCH path = (d {name: $drug})-[*1..2]-(t {name: $target})
        RETURN [r IN relationships(path) | type(r)] AS rels,
               [n IN nodes(path) | n.name] AS nodes,
               [r IN relationships(path) | coalesce(r.polarity, 1)] AS polarities
        LIMIT 25
        """
        items = []
        with self.driver.session() as session:
            for record in session.run(query, drug=drug, target=target):
                polarity = -1 if any(p < 0 for p in record["polarities"]) else 1
                claim = " -> ".join(
                    f"{a} {b.replace('_', ' ').lower()} {c}"
                    for a, b, c in zip(record["nodes"], record["rels"], record["nodes"][1:])
                )
                items.append(EvidenceItem(
                    source="knowledge_graph", claim=claim, polarity=polarity,
                    weight=0.55 if len(record["rels"]) == 1 else 0.3,
                    meta={"hops": len(record["rels"]), "backend": self.backend},
                ))
        return items

    def close(self):
        self.driver.close()


def get_backend():
    """Neo4j when configured, local JSON otherwise."""
    if config.NEO4J_URI:
        try:
            return Neo4jKG()
        except Exception as exc:                 # noqa: BLE001 - fall back loudly
            print(f"[kg] Neo4j unavailable ({exc}); using local JSON snapshot")
    return LocalKG()
