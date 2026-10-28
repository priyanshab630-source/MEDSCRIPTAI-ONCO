"""Load data/kg_triples.json into Neo4j.

    # with docker compose
    docker compose up -d neo4j
    python scripts/load_neo4j.py --uri bolt://localhost:7687 \
        --user neo4j --password testpassword

    # then point the app at it
    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword \
        python main.py --drug SHP099 --target SHP2

Every edge is stored as `:RELATED` with the semantic relation on a property, so
the same node can carry many edge types without a schema migration:

    (:Entity {name: "SHP099"})-[:RELATED {relation: "INHIBITS", polarity: 1}]->(:Entity {name: "SHP2"})
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

CONSTRAINT = (
    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
)

UPSERT = """
UNWIND $rows AS row
MERGE (s:Entity {name: row.subject})
MERGE (o:Entity {name: row.object})
MERGE (s)-[r:RELATED {relation: row.relation}]->(o)
SET r.polarity = row.polarity,
    r.source = row.source,
    r.updated_at = datetime()
RETURN count(r) AS edges
"""


def load(session, triples: list) -> dict:
    """Create the constraint, then upsert every triple. Returns counts."""
    session.run(CONSTRAINT)
    rows = [
        {
            "subject": t["subject"],
            "relation": t["relation"],
            "object": t["object"],
            "polarity": int(t.get("polarity", 1)),
            "source": t.get("source", "curated"),
        }
        for t in triples
    ]
    result = session.run(UPSERT, rows=rows)
    record = result.single() if result is not None else None
    edges = (record["edges"] if record else 0) or len(rows)
    return {"rows": len(rows), "edges": edges}


def load_file(driver, path=None) -> dict:
    path = Path(path) if path else (config.DATA_DIR / "kg_triples.json")
    with open(path, encoding="utf-8") as handle:
        triples = json.load(handle)
    with driver.session() as session:
        return load(session, triples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Load the KG snapshot into Neo4j")
    parser.add_argument("--uri", default=config.NEO4J_URI or "bolt://localhost:7687")
    parser.add_argument("--user", default=config.NEO4J_USER or "neo4j")
    parser.add_argument("--password", default=config.NEO4J_PASSWORD or "testpassword")
    parser.add_argument("--file", default=None, help="triples JSON (defaults to data/kg_triples.json)")
    args = parser.parse_args(argv)

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        stats = load_file(driver, args.file)
        print(f"Loaded {stats['rows']} triple(s) into {args.uri} "
              f"(edges reported by Neo4j: {stats['edges']})")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
