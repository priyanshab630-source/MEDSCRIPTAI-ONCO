"""The 6-agent pipeline, on either engine.

    hypothesis_planner -> evidence -> analysis -> critic -> adaptive_planner
                                     ^                          |
                                     +---- (loop: widen) -------+
                                                                |
                                                          decision -> END

Set `USE_LANGGRAPH=1` to run the same graph on the real LangGraph engine; without
it the dependency-free shim in `src/core/graph.py` is used. Both paths are
covered by tests/test_engine.py.
"""

from src.agents import adaptive, analysis, critic, decision, evidence, planner
from src.core import tracing
from src.core.engine import build_state_graph
from src.core.state import OncoState


def _build():
    graph, end, engine = build_state_graph(OncoState)

    graph.add_node("hypothesis_planner", planner.run)
    graph.add_node("evidence_agent", evidence.run)
    graph.add_node("computational_analysis", analysis.run)
    graph.add_node("critic", critic.run)
    graph.add_node("adaptive_planner", adaptive.run)
    graph.add_node("decision_agent", decision.run)

    graph.set_entry_point("hypothesis_planner")
    graph.add_edge("hypothesis_planner", "evidence_agent")
    graph.add_edge("evidence_agent", "computational_analysis")
    graph.add_edge("computational_analysis", "critic")
    graph.add_edge("critic", "adaptive_planner")
    graph.add_conditional_edges(
        "adaptive_planner",
        lambda state: state.get("next_action", "decision"),
        {"evidence": "evidence_agent", "decision": "decision_agent"},
    )
    graph.add_edge("decision_agent", end)
    return graph.compile(), engine


def build_graph():
    """The compiled graph (used by tests and the API health check)."""
    return _build()[0]


def run_pipeline(hypothesis, trace: bool = True):
    """Hypothesis in, VerdictReport out, with one trace record per run."""
    compiled, engine = _build()
    tracer = tracing.Tracer("onco.validate", {
        "drug": hypothesis.drug, "target": hypothesis.target,
        "disease": hypothesis.disease, "engine": engine,
    })

    final_state = compiled.invoke({
        "hypothesis": hypothesis, "loops": 0, "trace": [],
    })
    report = final_state["report"]
    report.engine = engine

    for step in final_state.get("trace", []):
        tracer.event(step["agent"], {"message": step["message"]})
    if trace:
        tracer.end({"verdict": report.verdict,
                    "overall_uncertainty": report.overall_uncertainty,
                    "dominant_dimension": report.dominant_dimension})
    return report
