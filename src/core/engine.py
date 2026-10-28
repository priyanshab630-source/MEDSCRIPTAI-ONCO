"""Orchestration engine selection.

`USE_LANGGRAPH=1` runs the real LangGraph `StateGraph`; otherwise the built-in
shim in `src/core/graph.py` is used. Both expose the same API, so the agent
files and the graph wiring are identical either way - only the engine changes.
"""

import config

_ENGINE = None


def build_state_graph(state_schema=None):
    """Returns (graph, END sentinel, engine name)."""
    global _ENGINE
    schema = state_schema or dict

    if config.USE_LANGGRAPH:
        try:
            from langgraph.graph import END, StateGraph

            _ENGINE = "langgraph"
            return StateGraph(schema), END, _ENGINE
        except ImportError:
            print("[engine] langgraph is not installed - falling back to the "
                  "built-in StateGraph shim (pip install langgraph)")

    from src.core.graph import END, StateGraph

    _ENGINE = "builtin"
    return StateGraph(schema), END, _ENGINE


def engine_name() -> str:
    """What the next build_state_graph() call will use."""
    if config.USE_LANGGRAPH:
        try:
            import langgraph  # noqa: F401

            return "langgraph"
        except ImportError:
            return "builtin (langgraph not installed)"
    return "builtin"
