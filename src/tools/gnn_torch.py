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
import pathlib
import sys

import numpy as np

import config
from src.tools import gnn as numpy_gnn
from src.tools import smiles as smiles_tool

HIDDEN = 32
CHECKPOINT = (pathlib.Path(config.GNN_CHECKPOINT) if config.GNN_CHECKPOINT
              else config.ROOT / "models" / "gnn.pt")
METADATA = config.ROOT / "models" / "gnn.json"
LEGACY_CHECKPOINT = config.ROOT / "models" / "gnn_logp.pt"   # proxy-task checkpoint


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
            self.dropout = torch.nn.Dropout(0.15)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(hidden // 2, 1),
            )

        def forward(self, x, adj):
            h = torch.tanh(self.self_1(x) + adj @ self.neigh_1(x))
            h = torch.tanh(self.self_2(h) + adj @ self.neigh_2(h))
            return self.head(self.dropout(h).mean(dim=0)).squeeze(-1)

        def embed(self, x, adj):
            h = torch.tanh(self.self_1(x) + adj @ self.neigh_1(x))
            h = torch.tanh(self.self_2(h) + adj @ self.neigh_2(h))
            return h.mean(dim=0)

    return MolecularGNN()


def load(path=None):
    """Load the trained model, or None when unavailable/untrained.

    Prefers the current checkpoint (trained on a real dataset); falls back to the
    legacy proxy-task weights so older clones still work.
    """
    path = path or CHECKPOINT
    if not path.exists() and LEGACY_CHECKPOINT.exists() and path == CHECKPOINT:
        path = LEGACY_CHECKPOINT
    if not available() or not path.exists():
        return None
    import torch

    model = build_model()
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    except RuntimeError as exc:
        # Architecture changed since the checkpoint was written: retrain.
        print(f"[gnn] checkpoint incompatible with the current model ({exc}); "
              "run training/train_gnn.py", file=sys.stderr)
        return None
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


def predict_distribution(smiles: str, n_samples: int = 25, model=None,
                         seed: int = 0xC0FFEE):
    """Monte-Carlo dropout: (mean, std) over `n_samples` stochastic passes.

    The spread is epistemic uncertainty - how much the network disagrees with
    itself about a molecule it has not really seen. It is measured, not assumed.
    """
    try:
        import torch
    except ImportError:
        return None                       # no torch: no MC dropout, and no crash

    model = model or load()
    if model is None:
        return None

    x, adj = features(smiles)
    if x.shape[0] == 0:
        return None
    tensor_x, tensor_adj = torch.from_numpy(x), torch.from_numpy(adj)

    # Seeded so the same molecule always reports the same uncertainty: a
    # framework that asks for traceability cannot have a stochastic verdict.
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    model.train()                     # dropout ON: this is what makes it MC-dropout
    predictions = []
    try:
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(float(model(tensor_x, tensor_adj)))
    finally:
        model.eval()
        torch.set_rng_state(rng_state)

    mean = sum(predictions) / len(predictions)
    variance = sum((value - mean) ** 2 for value in predictions) / len(predictions)
    return mean, variance ** 0.5


def epistemic_uncertainty(smiles: str, n_samples: int = 25):
    """MC-dropout spread scaled by the training target spread -> [0, 1] or None."""
    distribution = predict_distribution(smiles, n_samples)
    if distribution is None:
        return None
    _, std = distribution
    scale = metadata().get("target_std") or 1.0
    return round(min(1.0, std / max(scale, 1e-6)), 4)


def metadata_for(checkpoint=None) -> pathlib.Path:
    """Metadata lives next to its checkpoint: gnn.pt -> gnn.json."""
    checkpoint = pathlib.Path(checkpoint or CHECKPOINT)
    return checkpoint.with_suffix(".json")


def metadata(checkpoint=None) -> dict:
    path = metadata_for(checkpoint)
    if not path.exists() and LEGACY_CHECKPOINT.with_suffix(".json").exists():
        path = LEGACY_CHECKPOINT.with_suffix(".json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def describe() -> str:
    """One line for the report: trained or not, and on what."""
    if load() is None:
        return "untrained (no checkpoint)"
    info = metadata()
    dataset = info.get("dataset") or "proxy task"
    metric = info.get("val_mae") or (info.get("val") or {}).get("mae")
    detail = f"val MAE {metric}" if metric is not None else f"val {info.get('val')}"
    return (f"trained on {dataset} ({info.get('molecules', '?')} molecules, "
            f"{detail}); MC-dropout uncertainty enabled")
