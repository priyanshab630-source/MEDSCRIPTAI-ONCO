"""Agent 2 - Evidence Agent.

Retrieves literature, knowledge-graph paths and assay rows. On the second
adaptive loop it widens the search (live Europe PMC + disease-level records)
because the first pass came back thin.
"""

import json

import config
from src.core.state import log
from src.tools import knowledge_graph, literature

SYSTEM = (
    "You extract drug-target evidence. For each snippet say SUPPORTS, "
    "CONTRADICTS or NEUTRAL. Never invent evidence."
)


def _assays(drug: str, target: str) -> list:
    """Assay rows from the local table, as EvidenceItems."""
    from src.core.schemas import EvidenceItem

    path = config.DATA_DIR / "assays.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)

    items = []
    for row in rows:
        if row["drug"].lower() != drug.lower():
            continue
        if row["target"].lower() != target.lower():
            continue
        items.append(EvidenceItem(
            source="assay",
            claim=(f"{row['assay_type']} {row['value']} {row['unit']} "
                   f"for {drug} on {target} ({row['cell_line']})"),
            polarity=int(row["polarity"]),
            weight=0.7,
            meta={"source": row.get("source", "internal"), "cell_line": row["cell_line"]},
        ))
    return items


def run(state):
    hypothesis = state["hypothesis"]
    wide = state.get("loops", 0) > 0

    items = literature.search(hypothesis.drug, hypothesis.target, hypothesis.disease)
    if wide:
        items += literature.search_live(hypothesis.drug, hypothesis.target)

    kg = knowledge_graph.get_backend()
    kg_items = kg.evidence(hypothesis.drug, hypothesis.target)
    if hasattr(kg, "close"):
        kg.close()

    assays = _assays(hypothesis.drug, hypothesis.target)

    state["evidence"] = items + kg_items + assays
    return log(
        state, "evidence_agent",
        f"{len(items)} literature, {len(kg_items)} graph path(s), "
        f"{len(assays)} assay row(s)" + (" [wide search]" if wide else ""),
    )
