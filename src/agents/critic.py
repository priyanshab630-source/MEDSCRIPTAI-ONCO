"""Agent 4 - Falsification / Critic Agent.

Its job is to argue *against* the hypothesis. It is scored on finding
disconfirming evidence, not on agreement - which is what stops the pipeline
from drifting into confirmation bias.
"""

from src.core.state import log
from src.tools import llm

SYSTEM = (
    "You are a falsification critic. Given a drug-target claim and its evidence, "
    "list the strongest reasons the claim could be wrong. Never add new facts."
)


def _rule_based_contradictions(hypothesis, evidence, computations) -> list:
    notes = []

    negative = [e for e in evidence if e.polarity < 0]
    for item in negative[:4]:
        notes.append(f"contradicting {item.source} evidence: {item.claim[:160]}")

    positive = [e for e in evidence if e.polarity > 0]
    if positive and negative:
        notes.append(
            f"evidence is split: {len(positive)} supporting vs "
            f"{len(negative)} contradicting item(s)"
        )

    if computations:
        scores = [c.score for c in computations]
        spread = max(scores) - min(scores)
        if spread > 0.30:
            best = max(computations, key=lambda c: c.score)
            worst = min(computations, key=lambda c: c.score)
            notes.append(
                f"computational disagreement of {spread:.2f}: {best.name} "
                f"({best.score:.2f}) vs {worst.name} ({worst.score:.2f})"
            )

    if not any(e.source == "knowledge_graph" for e in evidence):
        notes.append("no knowledge-graph path connects the drug and target")

    if not any(e.source == "assay" for e in evidence):
        notes.append("no measured assay data - potency claim is unverified")

    if hypothesis.disease and not any(
        hypothesis.disease.lower() in e.claim.lower() for e in evidence
    ):
        notes.append(
            f"no retrieved evidence links the pair to {hypothesis.disease}"
        )
    return notes


def run(state):
    hypothesis = state["hypothesis"]
    evidence = state.get("evidence", [])
    computations = state.get("computations", [])

    notes = _rule_based_contradictions(hypothesis, evidence, computations)

    summary = "\n".join(f"- {e.claim[:200]}" for e in evidence[:6]) or "none"
    text = llm.chat(
        SYSTEM,
        f"Claim: {hypothesis.drug} modulates {hypothesis.target}.\n"
        f"Evidence:\n{summary}\n\nList at most 4 reasons this could be wrong.",
    )
    if text:
        notes += [line.strip("-*0123456789. ") for line in text.splitlines()
                  if line.strip()][:4]

    state["contradictions"] = notes
    return log(state, "critic", f"found {len(notes)} disconfirming consideration(s)")
