"""Molecular GNN: message passing over the atom graph, numpy only.

This is the *wiring* of a graph neural network, not a trained model. Atom
features -> two rounds of neighbourhood aggregation -> mean pooling ->
profile-conditioned readout. The readout weights are hand-set and explainable;
the message-passing weights are seeded so runs are reproducible.

To use a trained checkpoint, keep `embed()` and replace `predict_binding()`
with a loaded torch model - the rest of the pipeline does not change.
"""

import hashlib

import numpy as np

from src.tools import qsar
from src.tools import smiles as smiles_tool

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
FEATURE_DIM = len(ELEMENTS) + 1 + 3 + 1 + 1   # one-hot (+"other") + degree + aromatic + ring

HIDDEN = 16
LAYERS = 2
SEED = 13


def _element_one_hot(symbol):
    vector = np.zeros(len(ELEMENTS) + 1)
    key = symbol.capitalize()
    index = ELEMENTS.index(key) if key in ELEMENTS else len(ELEMENTS)
    vector[index] = 1.0
    return vector


def featurize(mol: dict) -> np.ndarray:
    """Atom feature matrix of shape (n_atoms, FEATURE_DIM)."""
    degree = np.zeros(len(mol["atoms"]))
    ring_atoms = set()
    for a, b, _ in mol["bonds"]:
        degree[a] += 1
        degree[b] += 1

    # Atoms that appear in more than one ring bond are treated as ring atoms.
    seen = {}
    for a, b, _ in mol["bonds"]:
        seen[a] = seen.get(a, 0) + 1
        seen[b] = seen.get(b, 0) + 1
    for atom, count in seen.items():
        if count > 2:
            ring_atoms.add(atom)

    rows = []
    for i, atom in enumerate(mol["atoms"]):
        deg = np.zeros(3)
        deg[min(int(degree[i]), 2)] = 1.0
        rows.append(np.concatenate([
            _element_one_hot(atom["symbol"]),
            deg,
            [1.0 if atom["aromatic"] else 0.0],
            [1.0 if i in ring_atoms else 0.0],
        ]))
    return np.array(rows, dtype=float) if rows else np.zeros((0, FEATURE_DIM))


def _seeded_matrix(seed_text: str, rows: int, cols: int, scale: float = 0.3) -> np.ndarray:
    """Deterministic pseudo-random weights, reproducible across runs."""
    digest = hashlib.sha256(seed_text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.normal(0.0, scale, size=(rows, cols))


def adjacency(mol: dict) -> np.ndarray:
    n = len(mol["atoms"])
    adj = np.eye(n)                       # self loops so atoms keep their own signal
    for a, b, order in mol["bonds"]:
        adj[a, b] = adj[b, a] = order
    degree = adj.sum(axis=1, keepdims=True)
    degree[degree == 0] = 1.0
    return adj / degree                   # row-normalised aggregation


def embed(smiles: str) -> np.ndarray:
    """Run LAYERS of message passing and mean-pool to one vector."""
    mol = smiles_tool.parse(smiles)
    x = featurize(mol)
    if x.shape[0] == 0:
        return np.zeros(HIDDEN)

    norm_adj = adjacency(mol)
    in_dim = FEATURE_DIM
    for layer in range(LAYERS):
        w_self = _seeded_matrix(f"self-{layer}", in_dim, HIDDEN)
        w_neigh = _seeded_matrix(f"neigh-{layer}", in_dim, HIDDEN)
        x = np.tanh(x @ w_self + norm_adj @ (x @ w_neigh))
        in_dim = HIDDEN
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x.mean(axis=0)


def predict_binding(smiles: str, profile: dict) -> tuple:
    """Returns (score in [0,1], explanation string).

    Uses the trained torch checkpoint when one exists (`training/train_gnn.py`);
    otherwise falls back to the seeded, untrained NumPy path. Either way the
    readout on top of the embedding is rule-based, and the report says so.
    """
    descriptors = qsar.compute(smiles)
    if "error" in descriptors:
        return 0.0, descriptors["error"]

    embedding, source = None, "untrained numpy message passing"
    try:                                          # local import avoids a cycle
        from src.tools import gnn_torch

        model = gnn_torch.load()
        if model is not None:
            embedding = gnn_torch.embed(smiles, model)
            source = "trained checkpoint - " + gnn_torch.describe()
    except Exception as exc:                      # noqa: BLE001 - torch optional
        source = f"untrained numpy message passing ({type(exc).__name__})"

    if embedding is None:
        embedding = embed(smiles)

    # Readout part 1: how well the molecule's bulk properties sit inside the
    # target's reference window (explainable, hand-set weights).
    descriptor_terms, notes = [], []
    for field, span in (("mw", 500.0), ("logp", 5.0), ("hbd", 5.0),
                        ("hba", 10.0), ("tpsa", 140.0)):
        value = descriptors[field] / span
        window = profile.get(field)
        if window:
            low, high = window[0] / span, window[1] / span
            inside = low <= value <= high
            descriptor_terms.append(1.0 if inside else -0.6)
            notes.append(f"{field} {'fits' if inside else 'misses'} pocket window")
        else:
            descriptor_terms.append(0.0)

    # Readout part 2: projection of the pooled graph embedding.
    head = _seeded_matrix("readout", len(embedding), 1, scale=0.15).ravel()
    graph_term = float(np.dot(np.asarray(embedding, dtype=float), head))

    logit = 1.2 * sum(descriptor_terms) / max(len(descriptor_terms), 1) + graph_term
    score = float(1.0 / (1.0 + np.exp(-2.0 * logit)))
    detail = f"{source}; readout = profile fit + graph projection. " + "; ".join(notes[:3])
    return round(score, 4), detail
