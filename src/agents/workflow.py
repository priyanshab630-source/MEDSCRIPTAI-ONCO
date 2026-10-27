"""The 6-agent LangGraph-style pipeline.

    hypothesis_planner -> evidence -> analysis -> critic -> adaptive_planner
                                     ^                          |
                                     +---- (loop: widen) -------+
                                                                |
                                                          decision -> END

`src/core/graph.py` provides a drop-in StateGraph with the LangGraph API, so
this file is the only place to touch if you switch to real LangGraph.
"""

from src.agents import adaptive, analysis, critic, decision, evidence, planner
from src.core.graph import END, StateGraph


def build_graph():
    graph = StateGraph(dict)
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
    graph.add_edge("decision_agent", END)
    return graph.compile()


def run_pipeline(hypothesis) -> dict:
    """Convenience wrapper: hypothesis in, VerdictReport out."""
    final_state = build_graph().invoke({"hypothesis": hypothesis, "loops": 0,
                                        "trace": []})
    return final_state["report"]
