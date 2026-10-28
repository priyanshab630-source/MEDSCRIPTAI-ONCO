"""Central configuration for MEDSCRIPTAI-ONCO.

Every setting is optional: with no environment variables at all the prototype
still runs, using the local knowledge-graph snapshot and rule-based reasoning
instead of an LLM. Copy `.env.example` to `.env` and fill what you have.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# ---------- LLM (Sarvam AI by default, any OpenAI-compatible endpoint works) ----------
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.sarvam.ai/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "sarvam-m")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "45"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))

# ---------- Knowledge graph (Neo4j optional, JSON snapshot is the fallback) ----------
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ---------- Orchestration engine ----------
# USE_LANGGRAPH=1 runs the real LangGraph StateGraph; otherwise the dependency-free
# shim in src/core/graph.py is used. Same nodes, same edges, same result.
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "0") in ("1", "true", "yes")

# ---------- Observability ----------
# Local JSONL traces are always written; LangSmith / Langfuse are opt-in.
TRACE_DIR = Path(os.getenv("TRACE_DIR", str(ROOT / "traces")))
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "1") in ("1", "true", "yes")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "medscriptai-onco")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ---------- Decision policy ----------
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.55"))
SUPPORT_THRESHOLD = float(os.getenv("SUPPORT_THRESHOLD", "0.30"))
MAX_ADAPTIVE_LOOPS = int(os.getenv("MAX_ADAPTIVE_LOOPS", "2"))
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "4"))

# Weights for the 6-dimension uncertainty vector (must sum to 1.0).
UNCERTAINTY_WEIGHTS = {
    "evidence_coverage": 0.25,
    "assay_consistency": 0.15,
    "computational_agreement": 0.20,
    "graph_support": 0.15,
    "literature_conflict": 0.15,
    "model_reliability": 0.10,
}
