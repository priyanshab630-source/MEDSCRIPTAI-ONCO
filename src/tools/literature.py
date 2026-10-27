"""Literature evidence retrieval over a curated local corpus.

Offline by default (data/literature.json). `search_live()` shows where a real
Europe PMC / PubMed call would slot in - same return type, so the Evidence
agent does not change.
"""

import json

import config
from src.core.schemas import EvidenceItem


def _load() -> list:
    with open(config.DATA_DIR / "literature.json", encoding="utf-8") as handle:
        return json.load(handle)


def relevance(record: dict, drug: str, target: str, disease: str) -> float:
    """How relevant one corpus record is to this hypothesis, in [0, 1].

    Drug-level hits matter most; a paper about the target in the right disease
    is still useful context, but it is not evidence about *this* drug.
    """
    haystack = " ".join([
        record.get("title", ""), record.get("text", ""),
        record.get("drug", ""), record.get("target", ""),
        record.get("disease", ""),
    ]).lower()

    score = 0.0
    if drug.lower() in haystack:
        score += 0.6
    if target.lower() in haystack:
        score += 0.25
    if disease and disease.lower() in haystack:
        score += 0.15
    return round(min(1.0, score), 3)


def search(drug: str, target: str, disease: str = "") -> list:
    """Return EvidenceItems for every corpus record that mentions the pair."""
    items = []
    for record in _load():
        rel = relevance(record, drug, target, disease)
        if rel < 0.4:
            continue
        # Recency weighting: 2024 evidence counts more than 2014 evidence.
        year = int(record.get("year", 2015))
        recency = max(0.3, min(1.0, 0.3 + (year - 2010) * 0.07))
        items.append(EvidenceItem(
            source="literature",
            claim=f"{record['title']} ({year}): {record['text']}",
            polarity=int(record.get("stance", 0)),
            weight=round(recency * rel, 3),
            meta={"id": record.get("id"), "year": year, "relevance": rel,
                  "study_type": record.get("study_type", "preclinical")},
        ))
    return items


_LIVE_DISABLED = False


def search_live(drug: str, target: str, limit: int = 10) -> list:
    """Optional live retrieval from Europe PMC (needs network, no key)."""
    global _LIVE_DISABLED
    if _LIVE_DISABLED:
        return []

    import requests

    query = f'("{drug}" OR "{target}") AND (drug-target OR inhibitor)'
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    try:
        response = requests.get(
            url, params={"query": query, "format": "json", "pageSize": limit},
            timeout=20,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
    except Exception as exc:                     # noqa: BLE001
        # One failure is enough: stop retrying and stay on the local corpus.
        _LIVE_DISABLED = True
        print(f"[literature] live search unavailable ({type(exc).__name__}); "
              "using the local corpus for the rest of this run")
        return []

    items = []
    for result in results:
        abstract = result.get("abstractText", "")
        stance = 0
        lowered = abstract.lower()
        if any(word in lowered for word in ("no significant", "failed to", "did not")):
            stance = -1
        elif any(word in lowered for word in ("potent", "sensitive", "inhibits")):
            stance = 1
        items.append(EvidenceItem(
            source="literature",
            claim=f"{result.get('title', 'Untitled')} ({result.get('pubYear', '')}): "
                  f"{abstract[:300]}",
            polarity=stance,
            weight=0.6,
            meta={"id": result.get("id"), "year": result.get("pubYear"),
                  "study_type": "live-europepmc"},
        ))
    return items
