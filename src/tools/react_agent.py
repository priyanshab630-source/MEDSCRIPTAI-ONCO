"""ChemCrow-style tool orchestration.

The Analysis agent does not call the tools in a fixed order any more: this loop
picks a tool, records the observation, and decides what to call next. Two
policies share one interface:

  * **LLM policy** - when an LLM is configured, it sees the tool catalogue and the
    observations so far and answers `TOOL: <name>`.
  * **deterministic policy** - otherwise it walks a priority order and skips tools
    whose inputs are missing. Same transcript shape, no model required, which is
    what makes the runs reproducible and testable.

Every call is appended to a transcript that ends up in the verdict report, so
"which tools produced this verdict" is answerable.
"""

import json

import config
from src.tools import docking, gnn, knowledge_graph, literature, qsar
from src.tools import llm

SYSTEM = (
    "You orchestrate chemistry tools to test a drug-target hypothesis. "
    "Reply with exactly one line: TOOL: <tool_name>. Available tools: "
    "rdkit_descriptors, gnn_binding, docking, kg_lookup, literature_search."
)

PRIORITY = ["rdkit_descriptors", "docking", "gnn_binding", "kg_lookup",
            "literature_search"]


def _run_tool(name: str, hypothesis, profile: dict) -> dict:
    """Execute one tool and return a compact observation."""
    if name == "rdkit_descriptors":
        descriptors = qsar.compute(hypothesis.smiles)
        if "error" in descriptors:
            return {"tool": name, "ok": False, "observation": descriptors["error"]}
        score, detail = qsar.score_against_profile(descriptors, profile)
        return {"tool": name, "ok": True,
                "observation": f"QSAR profile fit {score:.2f}; {detail}",
                "score": score}

    if name == "gnn_binding":
        score, detail = gnn.predict_binding(hypothesis.smiles, profile)
        return {"tool": name, "ok": True,
                "observation": f"GNN binding proxy {score:.2f}; {detail[:120]}",
                "score": score}

    if name == "docking":
        result = docking.score(hypothesis.smiles, hypothesis.target,
                               profile.get("pocket", {}))
        return {"tool": name, "ok": True,
                "observation": f"surrogate docking {result['kcal_per_mol']} kcal/mol "
                               f"(score {result['score01']:.2f})",
                "score": result["score01"]}

    if name == "kg_lookup":
        backend = knowledge_graph.get_backend()
        try:
            items = backend.evidence(hypothesis.drug, hypothesis.target)
        finally:
            if hasattr(backend, "close"):
                backend.close()
        summary = "; ".join(item.claim for item in items[:3]) or "no path found"
        return {"tool": name, "ok": True,
                "observation": f"{len(items)} graph path(s): {summary[:160]}"}

    if name == "literature_search":
        items = literature.search(hypothesis.drug, hypothesis.target,
                                  hypothesis.disease)
        stances = sum(1 for item in items if item.polarity > 0)
        against = sum(1 for item in items if item.polarity < 0)
        return {"tool": name, "ok": True,
                "observation": f"{len(items)} paper(s): {stances} support, "
                               f"{against} contradict"}

    return {"tool": name, "ok": False, "observation": f"unknown tool: {name}"}


def _choose_llm(transcript: list, hypothesis, available_tools: list):
    """Ask the model which tool to run next; returns a name or None."""
    history = "\n".join(f"- {step['tool']}: {step['observation'][:120]}"
                        for step in transcript) or "(nothing yet)"
    text = llm.chat(
        SYSTEM,
        f"Hypothesis: {hypothesis.drug} modulates {hypothesis.target}.\n"
        f"Observations so far:\n{history}\n"
        f"Unused tools: {', '.join(available_tools) or 'none'}.\n"
        "Which tool next?",
    )
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("TOOL:"):
            candidate = line.split(":", 1)[1].strip().strip("`'\" ").lower()
            if candidate in available_tools:
                return candidate
    return None


def run(hypothesis, profile: dict, max_steps: int = None) -> dict:
    """Run the tool loop. Returns {"mode", "steps": [...]}."""
    max_steps = max_steps or int(getattr(config, "MAX_TOOL_STEPS", 4))
    transcript, used = [], []
    mode = "llm" if llm.available() else "deterministic"

    if not hypothesis.smiles:                     # nothing to compute on
        return {"mode": "skipped", "steps": [],
                "note": "no SMILES supplied"}

    while len(transcript) < max_steps:
        remaining = [name for name in PRIORITY if name not in used]
        if not remaining:
            break

        choice = _choose_llm(transcript, hypothesis, remaining) if mode == "llm" else None
        if choice is None:                        # fall back to the priority order
            choice = remaining[0]

        step = _run_tool(choice, hypothesis, profile)
        used.append(choice)
        transcript.append(step)

        # Stop early once we have both a computational and an evidence signal.
        has_compute = any(s["tool"] in ("rdkit_descriptors", "gnn_binding", "docking")
                          for s in transcript)
        has_evidence = any(s["tool"] in ("kg_lookup", "literature_search")
                           for s in transcript)
        if has_compute and has_evidence and len(transcript) >= 3:
            break

    return {"mode": mode, "steps": transcript}
