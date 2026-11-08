"""Train the message-passing GNN on a real dataset.

    pip install torch rdkit
    python training/train_gnn.py --dataset esol --epochs 60
    python training/train_gnn.py --dataset bace --epochs 60      # classification

Datasets come from `datasets/manifest.json`, so every run records where its data
came from, who to cite and the file checksum. Bundled: ESOL (1128 measured
solubilities, Delaney 2004) and BACE (1513 inhibition labels, MoleculeNet).
`--dataset toy` falls back to the 33-molecule smoke library used earlier.

What this trains: the graph encoder and a scalar head. For ESOL the head predicts
measured log solubility; for BACE it predicts the inhibitor class (BCE loss).
The same encoder is what `src/tools/gnn.py` uses for its binding readout, so a
better-trained encoder is a better embedding - it is still not an affinity model.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.tools import datasets  # noqa: E402

DEFAULT_OUT = config.ROOT / "models" / "gnn.pt"


def toy_rows() -> list:
    """The original 33-molecule smoke library, kept for a fast sanity run."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    smiles_list = json.loads(
        (config.DATA_DIR / "gnn_pretrain_smiles.json").read_text(encoding="utf-8"))
    rows = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            rows.append({"smiles": smiles,
                         "target": float(Descriptors.MolLogP(molecule))})
    return rows


def load_rows(name: str) -> tuple:
    """(rows, provenance dict)."""
    if name == "toy":
        rows = toy_rows()
        random.Random(13).shuffle(rows)
        provenance = {"name": "toy", "title": "33-molecule smoke library",
                      "task": "regression", "rows": len(rows),
                      "target_label": "MolLogP", "target_units": "logP",
                      "source": "RDKit-computed MolLogP over data/gnn_pretrain_smiles.json",
                      "citation": "none - synthetic smoke test",
                      "sha256": ""}
        return rows, provenance

    entry = datasets.get(name)
    verify = datasets.verify(name)
    if not verify["ok"]:
        raise SystemExit(f"{name}: {verify['reason']}. "
                         f"Run: python scripts/fetch_datasets.py --fetch {name}")

    rows = datasets.load(name)
    provenance = {
        "name": entry["name"], "title": entry.get("title"),
        "task": entry.get("task", "regression"),
        "rows": len(rows),
        "target_label": entry.get("target_label", entry["target_column"]),
        "target_units": entry.get("target_units", ""),
        "source": entry.get("source"), "citation": entry.get("citation"),
        "url": entry.get("url"), "license": entry.get("license"),
        "sha256": entry.get("sha256", ""),
    }
    return rows, provenance


def auroc(labels: list, scores: list) -> float:
    """Rank-based AUROC (Mann-Whitney U), no sklearn needed."""
    positives = [s for s, l in zip(scores, labels) if l == 1]
    negatives = [s for s, l in zip(scores, labels) if l == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return round(wins / (len(positives) * len(negatives)), 4)


def evaluate(model, batches, task: str, mean: float, std: float) -> dict:
    import torch

    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, adj, target in batches:
            output = model(x, adj)
            predictions.append(float(output))
            targets.append(float(target))

    if task == "classification":
        labels = [1 if value > 0.5 else 0 for value in targets]
        probabilities = [1.0 / (1.0 + math.exp(-value)) for value in predictions]
        accuracy = sum(1 for p, l in zip(probabilities, labels)
                       if (p >= 0.5) == (l == 1)) / len(labels)
        return {"accuracy": round(accuracy, 4),
                "auroc": auroc(labels, probabilities),
                "n": len(labels)}

    # regression: undo the standardisation before reporting errors
    actual = [value * std + mean for value in predictions]
    truth = [value * std + mean for value in targets]
    mae = sum(abs(a - t) for a, t in zip(actual, truth)) / len(actual)
    rmse = math.sqrt(sum((a - t) ** 2 for a, t in zip(actual, truth)) / len(actual))
    mean_truth = sum(truth) / len(truth)
    ss_tot = sum((t - mean_truth) ** 2 for t in truth)
    ss_res = sum((a - t) ** 2 for a, t in zip(actual, truth))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else 0.0,
            "n": len(actual)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the GNN on a real dataset")
    parser.add_argument("--dataset", default="esol",
                        help="esol | bace | qm7 | toy (see datasets/manifest.json)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import torch
    except ImportError:
        print("torch is required: pip install torch")
        return 1

    from src.tools import gnn_torch

    rows, provenance = load_rows(args.dataset)
    task = provenance["task"]
    print(f"dataset {provenance['name']}: {len(rows)} rows, task={task}")
    print(f"  source: {provenance.get('source')}")

    train_rows, val_rows, test_rows = datasets.split(rows)
    print(f"  split: train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)}")

    targets = [row["target"] for row in train_rows]
    mean = sum(targets) / len(targets)
    std = (sum((value - mean) ** 2 for value in targets) / len(targets)) ** 0.5 or 1.0

    def make_batches(subset):
        batches = []
        for row in subset:
            x, adj = gnn_torch.features(row["smiles"])
            if x.shape[0] == 0:
                continue
            normalised = (row["target"] - mean) / std if task == "regression" \
                else row["target"]
            batches.append((torch.from_numpy(x), torch.from_numpy(adj),
                            torch.tensor(float(normalised), dtype=torch.float32)))
        return batches

    train_batches = make_batches(train_rows)
    val_batches = make_batches(val_rows)
    test_batches = make_batches(test_rows)
    if not train_batches:
        print("no trainable molecules")
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = gnn_torch.build_model(args.hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() if task == "classification" \
        else torch.nn.MSELoss()

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_batches)
        total, steps = 0.0, 0
        optimiser.zero_grad()
        for index, (x, adj, target) in enumerate(train_batches, start=1):
            loss = loss_fn(model(x, adj), target)
            (loss / args.batch_size).backward()
            total += float(loss.detach())
            steps += 1
            if index % args.batch_size == 0:
                optimiser.step()
                optimiser.zero_grad()
        optimiser.step()
        optimiser.zero_grad()

        if epoch % max(1, args.epochs // 6) == 0 or epoch == 1:
            validation = evaluate(model, val_batches, task, mean, std)
            metric = (f"val acc {validation['accuracy']:.3f} auroc {validation['auroc']:.3f}"
                      if task == "classification"
                      else f"val MAE {validation['mae']:.3f} R2 {validation['r2']:.3f}")
            print(f"epoch {epoch:3d}/{args.epochs} | train loss {total / steps:.4f} | {metric}")

    test_metrics = evaluate(model, test_batches, task, mean, std)
    validation_metrics = evaluate(model, val_batches, task, mean, std)
    print(f"\ntest ({test_metrics['n']} held-out molecules): {test_metrics}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    metadata = {
        "dataset": provenance["name"],
        "title": provenance.get("title"),
        "source": provenance.get("source"),
        "citation": provenance.get("citation"),
        "url": provenance.get("url"),
        "dataset_sha256": provenance.get("sha256", ""),
        "task": task,
        "target": provenance.get("target_label"),
        "target_units": provenance.get("target_units", ""),
        "molecules": len(rows),
        "split": {"train": len(train_rows), "val": len(val_rows),
                  "test": len(test_rows)},
        "epochs": args.epochs,
        "hidden": args.hidden,
        "seed": args.seed,
        "target_mean": round(mean, 4),
        "target_std": round(std, 4),
        "val": validation_metrics,
        "test": test_metrics,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - started, 1),
    }
    if task == "regression":
        metadata["val_mae"] = validation_metrics["mae"]      # backwards compatible
    metadata_path = gnn_torch.metadata_for(args.out)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                             encoding="utf-8")
    print(f"saved {args.out} and {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
