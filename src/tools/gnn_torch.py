"""Trainable message-passing GNN (PyTorch).

This is the *trained* counterpart to `src/tools/gnn.py`. Same graph, same
aggregation, but the weights are learned by gradient descent and persisted to
`models/gnn_logp.pt` by `training/train_gnn.py`.

What it is trained on matters, so the report says it: the checkpoint predicts
**MolLogP** (a cheap RDKit property) on a 33-molecule proxy library. That makes
the embedding real and learned, but it does *not* make the affinity readout a
validated predictor - the readout on top of the embedding is still rule-based,
and `model_reliability` in the uncertainty vector still carries a penalty.

    python training/train_gnn.py --epochs 300
"""

import json

import numpy as np

import config
from src.tools import gnn as numpy_gnn
from src.tools import smiles as smiles_tool

HIDDEN = 32
CHECKPOINT = config.ROOT / "models" / "gnn_logp.pt"
METADATA = config.ROOT / "models" / "gnn_logp.json"


def available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def features(smiles: str):
    """(atom features, normalised adjacency) as float32 arrays."""
    mol = smiles_tool.parse(smiles)
    x = numpy_gnn.featurize(mol)
    adj = numpy_gnn.adjacency(mol)
    return x.astype(np.float32), adj.astype(np.float32)


def build_model(hidden: int = HIDDEN):
    """A two-layer GNN with mean pooling and a scalar head."""
    import torch

    in_dim = numpy_gnn.FEATURE_DIM

    class MolecularGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_1 = torch.nn.Linear(in_dim, hidden)
            self.neigh_1 = torch.nn.Linear(in_dim, hidden)
            self.self_2 = torch.nn.Linear(hidden, hidden)
            self.neigh_2 = torch.nn.Linear(hidden, hidden)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden // 2, 1),
            )

        def forward(self, x, adj):
            h = torch.tanh(self.self_1(x) + adj @ self.neigh_1(x))
            h = torch.tanh(self.self_2(h) + adj @ self.neigh_2(h))
            return self.head(h.mean(dim=0)).squeeze(-1)

        def embed(self, x, adj):
            h = torch.tanh(self.self_1(x) + adj @ self.neigh_1(x))
            h = torch.tanh(self.self_2(h) + adj @ self.neigh_2(h))
            return h.mean(dim=0)

    return MolecularGNN()


def load(path=None):
    """Load the trained model, or None when unavailable/untrained."""
    path = path or CHECKPOINT
    if not available() or not path.exists():
        return None
    import torch

    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def embed(smiles: str, model) -> np.ndarray:
    """Pooled learned embedding for one molecule."""
    import torch

    x, adj = features(smiles)
    if x.shape[0] == 0:
        return np.zeros(HIDDEN, dtype=np.float32)
    with torch.no_grad():
        vector = model.embed(torch.from_numpy(x), torch.from_numpy(adj))
    return vector.numpy()


def predict_property(smiles: str, model=None) -> float:
    """The trained target: predicted MolLogP."""
    import torch

    model = model or load()
    if model is None:
        raise RuntimeError("no trained checkpoint - run training/train_gnn.py")
    x, adj = features(smiles)
    with torch.no_grad():
        return float(model(torch.from_numpy(x), torch.from_numpy(adj)))


def metadata() -> dict:
    if not METADATA.exists():
        return {}
    with open(METADATA, encoding="utf-8") as handle:
        return json.load(handle)


def describe() -> str:
    """One line for the report: trained or not, and on what."""
    if load() is None:
        return "untrained (no checkpoint)"
    info = metadata()
    return (f"trained on proxy task {info.get('target', 'MolLogP')} "
            f"({info.get('molecules', '?')} molecules, val MAE "
            f"{info.get('val_mae', '?')})")
