"""Agent 1 - Hypothesis Planner.

Turns a one-line claim into an explicit, checkable plan. Nothing is retrieved
or scored here; the agent decides *what evidence would settle the claim*,
which is what makes the later abstention criteria meaningful.
"""

from src.core.state import log
from src.tools import llm

SYSTEM = (
    "You are a hypothesis planner for a precision-oncology research framework. "
    "Given a drug-target-disease claim, list the evidence types that would "
    "confirm or falsify it. Be terse and concrete."
)


def _rule_based_plan(hypothesis) -> list:
    return [
        f"literature: does any study test {hypothesis.drug} against "
        f"{hypothesis.target}?",
        f"knowledge_graph: is there a {hypothesis.drug}-{hypothesis.target} path "
        "with supporting mechanism?",
        f"assay: are there measured potency/response values for this pair?",
        f"computational: do QSAR, GNN and docking agree that {hypothesis.drug} "
        f"is compatible with the {hypothesis.target} pocket?",
        "falsification: search specifically for disconfirming results, "
        "resistance mutations and negative trials",
    ]


def run(state):
    hypothesis = state["hypothesis"]
    plan = _rule_based_plan(hypothesis)

    text = llm.chat(
        SYSTEM,
        f"Claim: {hypothesis.drug} modulates {hypothesis.target} in "
        f"{hypothesis.disease or 'an unspecified indication'}.\n"
        "List 5 evidence checks, one per line.",
    )
    if text:
        plan = [line.strip("-*0123456789. ") for line in text.splitlines() if line.strip()][:6]

    state["plan"] = plan
    state["loops"] = 0
    state["next_action"] = "evidence"
    return log(state, "hypothesis_planner",
               f"planned {len(plan)} evidence checks for {hypothesis.key()}")
