"""Small dataclasses shared by every agent.

Keeping the shapes here means each agent file only holds reasoning logic.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Hypothesis:
    """The drug-target claim the framework has to validate or falsify."""

    drug: str
    target: str
    disease: str = ""
    smiles: str = ""

    def key(self) -> str:
        return f"{self.drug}->{self.target}"


@dataclass
class EvidenceItem:
    """One piece of evidence: a literature snippet, a KG edge or an assay row."""

    source: str          # "literature" | "knowledge_graph" | "assay"
    claim: str
    polarity: int        # +1 supports, -1 contradicts, 0 neutral
    weight: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComputeResult:
    """One computational signal (QSAR / GNN / docking)."""

    name: str
    score: float         # normalised to [0, 1], higher = more supportive
    detail: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UncertaintyVector:
    """Six decomposed dimensions instead of one collapsed confidence score.

    Every component is an *uncertainty contribution* in [0, 1]:
    0 = fully resolved, 1 = nothing known / badly conflicted.
    """

    evidence_coverage: float = 1.0
    assay_consistency: float = 1.0
    computational_agreement: float = 1.0
    graph_support: float = 1.0
    literature_conflict: float = 1.0
    model_reliability: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)

    def overall(self, weights: dict) -> float:
        total = sum(weights.values()) or 1.0
        return round(sum(getattr(self, k) * w for k, w in weights.items()) / total, 4)

    def dominant(self) -> str:
        """Which dimension is driving the uncertainty most."""
        d = self.as_dict()
        return max(d, key=d.get)


@dataclass
class VerdictReport:
    """What the Decision agent returns to the caller."""

    verdict: str                       # SUPPORTED | CONTRADICTED | UNCERTAIN | ABSTAIN
    overall_uncertainty: float
    uncertainty: dict
    dominant_dimension: str
    rationale: str
    support_score: float = 0.0
    evidence: list = field(default_factory=list)
    computations: list = field(default_factory=list)
    abstain_reasons: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    tool_mode: str = "deterministic"
    engine: str = "builtin"

    def to_dict(self) -> dict:
        return asdict(self)
