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

# ---------- Decision policy ----------
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.55"))
SUPPORT_THRESHOLD = float(os.getenv("SUPPORT_THRESHOLD", "0.30"))
MAX_ADAPTIVE_LOOPS = int(os.getenv("MAX_ADAPTIVE_LOOPS", "2"))

# Weights for the 6-dimension uncertainty vector (must sum to 1.0).
UNCERTAINTY_WEIGHTS = {
    "evidence_coverage": 0.25,
    "assay_consistency": 0.15,
    "computational_agreement": 0.20,
    "graph_support": 0.15,
    "literature_conflict": 0.15,
    "model_reliability": 0.10,
}
