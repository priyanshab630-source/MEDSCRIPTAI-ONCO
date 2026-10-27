"""A tiny StateGraph runner with the same surface as LangGraph.

Why not just import langgraph? So the prototype runs with zero heavy
dependencies. If you `pip install langgraph` and want the real engine, the
node functions in `src/agents/workflow.py` are already LangGraph-compatible:
replace this import with `from langgraph.graph import StateGraph, END`.
"""

END = "__end__"


class CompiledGraph:
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, state, max_steps: int = 30):
        node = self.graph.entry
        steps = 0
        while node and node != END and steps < max_steps:
            state = self.graph.nodes[node](state)
            steps += 1
            node = self._next(node, state)
        return state

    def _next(self, node, state):
        if node in self.graph.conditional:
            router, mapping = self.graph.conditional[node]
            target = router(state)
            return mapping.get(target, target) if mapping else target
        return self.graph.edges.get(node, END)


class StateGraph:
    """Minimal add_node / add_edge / add_conditional_edges / compile API."""

    def __init__(self, state_schema=None):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = {}
        self.conditional = {}
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, src, dst):
        self.edges[src] = dst

    def add_conditional_edges(self, src, router, mapping=None):
        self.conditional[src] = (router, mapping)

    def set_entry_point(self, name):
        self.entry = name

    def compile(self):
        return CompiledGraph(self)
