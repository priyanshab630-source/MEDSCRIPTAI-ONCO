"""Train the tool-selection policy with REINFORCE.

    pip install torch rdkit
    python training/train_policy.py --episodes 300

The environment is the real one: every action runs an actual tool (RDKit
descriptors, conformer docking, the GNN, the knowledge graph, the literature
corpus), and the reward measures how much evidence the episode gathered per step
it spent. There is no simulator and no cached reward table.

Reward
    +1.0  both a computational and an evidence signal were obtained
    +0.3 * mean computational score
    -0.1  per tool call
    -0.5  no evidence signal at all
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def hypotheses() -> list:
    """Every drug-target pair in the seed tables, plus deliberate mismatches."""
    from src.core.schemas import Hypothesis

    drugs = json.loads((config.DATA_DIR / "drugs.json").read_text(encoding="utf-8"))
    targets = json.loads((config.DATA_DIR / "targets.json").read_text(encoding="utf-8"))

    pairs = []
    for name, record in drugs.items():
        pairs.append(Hypothesis(drug=name, target=record["target"],
                                disease="KRAS-mutant NSCLC",
                                smiles=record["smiles"]))
    # mismatches: a real drug against the wrong target
    for name, record in drugs.items():
        for target in targets:
            if target != record["target"]:
                pairs.append(Hypothesis(drug=name, target=target,
                                        smiles=record["smiles"]))
    return pairs


def profile_for(target: str) -> dict:
    targets = json.loads((config.DATA_DIR / "targets.json").read_text(encoding="utf-8"))
    return targets.get(target, {})


def run_episode(policy_model, hypothesis, max_steps: int):
    """One episode: sample tools, execute them, score the trajectory."""
    import torch

    from src.tools import policy, react_agent

    transcript, log_probs, masks = [], [], []
    while len(transcript) < max_steps:
        mask = policy.mask_for(transcript)
        if not any(mask):
            break
        features = policy.encode(transcript, bool(hypothesis.smiles), max_steps)
        distribution = policy.distribution(policy_model, features, mask)
        index = int(distribution.sample())
        log_probs.append(distribution.log_prob(torch.tensor(index)))
        masks.append(mask)

        step = react_agent._run_tool(policy.TOOLS[index], hypothesis,
                                     profile_for(hypothesis.target))
        transcript.append(step)

    return transcript, log_probs


def reward_of(transcript: list) -> float:
    compute_tools = {"rdkit_descriptors", "docking", "gnn_binding"}
    evidence_tools = {"kg_lookup", "literature_search"}
    tools = [step["tool"] for step in transcript]

    has_compute = any(tool in compute_tools for tool in tools)
    has_evidence = any(tool in evidence_tools for tool in tools)
    scores = [step.get("score", 0.0) for step in transcript
              if step["tool"] in compute_tools and step.get("score") is not None]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    reward = 0.3 * mean_score - 0.1 * len(transcript)
    if has_compute and has_evidence:
        reward += 1.0
    if not has_evidence:
        reward -= 0.5
    return round(reward, 4)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="REINFORCE for tool selection")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch is required: pip install torch")
        return 1

    from src.tools import policy

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = policy.build()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    baseline = 0.0
    returns = []
    pairs = hypotheses()

    for episode in range(1, args.episodes + 1):
        hypothesis = random.choice(pairs)
        transcript, log_probs = run_episode(model, hypothesis, args.max_steps)
        reward = reward_of(transcript)
        returns.append(reward)
        baseline = 0.9 * baseline + 0.1 * reward       # moving-average baseline

        if log_probs:
            loss = -torch.stack(log_probs).sum() * (reward - baseline)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if episode % max(1, args.episodes // 5) == 0:
            recent = returns[-25:]
            print(f"episode {episode:4d} | reward {reward:+.2f} | "
                  f"mean(last 25) {sum(recent) / len(recent):+.3f} | "
                  f"tools {[s['tool'] for s in transcript]}")

    stats = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "mean_return": round(sum(returns[-50:]) / min(len(returns), 50), 4),
        "tools": policy.TOOLS,
        "feature_dim": policy.FEATURE_DIM,
    }
    policy.save(model, stats)
    print(f"mean return over the last 50 episodes: {stats['mean_return']}")
    print(f"saved {policy.CHECKPOINT} and {policy.METADATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
