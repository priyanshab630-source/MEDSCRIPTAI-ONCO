"""Train the message-passing GNN on a proxy property task.

    pip install torch rdkit
    python training/train_gnn.py --epochs 300

The proxy task is MolLogP on `data/gnn_pretrain_smiles.json` (33 molecules). It
exists to prove the training loop, the checkpoint format and the inference path -
not to produce a binding predictor. Swap `--target` for a real assay column once
you have one, and the rest of the pipeline does not change.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def make_dataset(target: str):
    """[(smiles, value)] for every library molecule RDKit can parse."""
    with open(config.DATA_DIR / "gnn_pretrain_smiles.json", encoding="utf-8") as handle:
        smiles_list = json.load(handle)

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    rows = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        rows.append((smiles, float(Descriptors.MolLogP(molecule))))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the GNN on a proxy property")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(config.ROOT / "models" / "gnn_logp.pt"))
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import torch
    except ImportError:
        print("torch is required: pip install torch")
        return 1

    from src.tools import gnn_torch

    rows = make_dataset("MolLogP")
    if len(rows) < 10:
        print("not enough molecules to train on")
        return 1

    random.Random(args.seed).shuffle(rows)
    split = max(1, int(len(rows) * (1 - args.val_fraction)))
    train_rows, val_rows = rows[:split], rows[split:]

    # Standardise the target so the loss is well conditioned.
    mean = sum(value for _, value in train_rows) / len(train_rows)
    std = (sum((value - mean) ** 2 for _, value in train_rows) / len(train_rows)) ** 0.5 or 1.0

    def batch(rows_subset):
        tensors = [(torch.from_numpy(gnn_torch.features(s)[0]),
                    torch.from_numpy(gnn_torch.features(s)[1]),
                    torch.tensor((v - mean) / std, dtype=torch.float32))
                   for s, v in rows_subset]
        return tensors

    model = gnn_torch.build_model(args.hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    train_batches = batch(train_rows)
    val_batches = batch(val_rows)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, adj, target in train_batches:
            optimiser.zero_grad()
            loss = loss_fn(model(x, adj), target)
            loss.backward()
            optimiser.step()
            total += float(loss)
        if epoch % max(1, args.epochs // 5) == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val = sum(abs(float(model(x, adj)) - float(target))
                          for x, adj, target in val_batches) / max(len(val_batches), 1)
            print(f"epoch {epoch:4d} | train loss {total / len(train_batches):.4f} "
                  f"| val MAE (z-scored) {val:.4f}")

    # Report the error in logP units, which is what anyone will ask for.
    model.eval()
    with torch.no_grad():
        mae_logp = sum(abs((float(model(x, adj)) * std + mean) - (float(target) * std + mean))
                       for x, adj, target in val_batches) / max(len(val_batches), 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    with open(gnn_torch.METADATA, "w", encoding="utf-8") as handle:
        json.dump({
            "target": "MolLogP",
            "molecules": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "epochs": args.epochs,
            "hidden": args.hidden,
            "val_mae": round(mae_logp, 4),
            "target_mean": round(mean, 4),
            "target_std": round(std, 4),
        }, handle, indent=2)

    print(f"val MAE {mae_logp:.3f} logP units over {len(val_rows)} held-out molecules")
    print(f"saved {args.out} and {gnn_torch.METADATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
