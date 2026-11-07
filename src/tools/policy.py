"""Learned tool-selection policy (reinforcement learning).

The deterministic ReAct policy always walks the same priority list. This replaces
it with a policy trained by REINFORCE against the real environment: the agent
picks a tool, the tool actually runs, and the episode is scored on the evidence
it gathered relative to the steps it spent.

State  : 11 features (see `encode`) - what is known and what is still unused
Action : which of the five tools to call next
Reward : +1.0 for reaching both a computational and an evidence signal,
         +0.3 * mean computational score, -0.1 per tool call, -0.5 if no evidence

Trained by `training/train_policy.py`, checkpoint at `models/tool_policy.pt`.
Without a checkpoint the agent keeps using the deterministic policy, so this is a
pure upgrade rather than a dependency.
"""

import json

import config

TOOLS = ["rdkit_descriptors", "docking", "gnn_binding", "kg_lookup",
         "literature_search"]
FEATURE_DIM = 11
CHECKPOINT = config.ROOT / "models" / "tool_policy.pt"
METADATA = config.ROOT / "models" / "tool_policy.json"
HIDDEN = 32


def available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def encode(transcript: list, has_smiles: bool, max_steps: int) -> list:
    """The 11-feature state vector."""
    used = [step["tool"] for step in transcript]
    compute_tools = {"rdkit_descriptors", "docking", "gnn_binding"}
    evidence_tools = {"kg_lookup", "literature_search"}

    compute_scores = [step.get("score", 0.0) for step in transcript
                      if step["tool"] in compute_tools and step.get("score") is not None]
    mean_score = sum(compute_scores) / len(compute_scores) if compute_scores else 0.0

    features = [
        1.0 if has_smiles else 0.0,
        len(transcript) / max(max_steps, 1),
        len(used) / len(TOOLS),
        1.0 if any(tool in compute_tools for tool in used) else 0.0,
        1.0 if any(tool in evidence_tools for tool in used) else 0.0,
        mean_score,
    ]
    features += [1.0 if tool in used else 0.0 for tool in TOOLS]
    return features


def build(hidden: int = HIDDEN):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, len(TOOLS)),
    )


def save(model, stats: dict = None) -> None:
    import torch

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    if stats is not None:
        with open(METADATA, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)


def load(path=None):
    """Load the trained policy, or None when unavailable/incompatible."""
    path = path or CHECKPOINT
    if not available() or not path.exists():
        return None
    import torch

    model = build()
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    except RuntimeError:
        return None
    model.eval()
    return model


def mask_for(transcript: list) -> list:
    """1.0 for tools not yet used, 0.0 otherwise."""
    used = {step["tool"] for step in transcript}
    return [0.0 if tool in used else 1.0 for tool in TOOLS]


def distribution(model, features: list, mask: list):
    """Masked softmax over the available tools."""
    import torch

    logits = model(torch.tensor([features], dtype=torch.float32))
    mask_tensor = torch.tensor([mask], dtype=torch.float32)
    logits = logits.masked_fill(mask_tensor == 0, float("-inf"))
    return torch.distributions.Categorical(logits=logits)


def select(transcript: list, has_smiles: bool, max_steps: int, model=None,
           greedy: bool = True):
    """Pick the next tool with the trained policy; None if it cannot decide."""
    model = model or load()
    if model is None:
        return None
    mask = mask_for(transcript)
    if not any(mask):
        return None
    features = encode(transcript, has_smiles, max_steps)
    distribution_ = distribution(model, features, mask)
    index = int(distribution_.probs.argmax()) if greedy else int(distribution_.sample())
    return TOOLS[index]


def metadata() -> dict:
    if not METADATA.exists():
        return {}
    with open(METADATA, encoding="utf-8") as handle:
        return json.load(handle)


def describe() -> str:
    if load() is None:
        return "deterministic priority order (no trained policy)"
    info = metadata()
    return (f"REINFORCE policy ({info.get('episodes', '?')} episodes, mean return "
            f"{info.get('mean_return', '?')})")
