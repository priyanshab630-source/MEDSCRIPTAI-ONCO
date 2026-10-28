"""Preference tuning (DPO) on the pairs built by training/build_dataset.py.

Stage two of the plan: after SFT teaches the output shape, DPO pushes the model
away from the collapsed-confidence answer and towards the structured verdict
(with abstention when the evidence is thin).

    pip install transformers datasets peft accelerate trl
    python training/dpo_train.py --data training/data/dpo.jsonl \
        --model training/checkpoints/sft --out training/checkpoints/dpo
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = "transformers, datasets, peft, trl"


def load_rows(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sequence_logprob(model, input_ids, attention_mask) -> "torch.Tensor":
    """Sum of token log-probabilities under the model."""
    import torch

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (picked * attention_mask[:, 1:]).sum(dim=-1)


def train_tiny(rows: list, args) -> int:
    """Run a real DPO objective on a tiny local model.

    The loss is the reference implementation from the DPO paper:

        -log sigmoid( beta * [ (logpi(y_w) - logref(y_w))
                             - (logpi(y_l) - logref(y_l)) ] )

    The reference model is a frozen copy of the starting weights. Small, slow and
    CPU-only, but it is the actual objective rather than a placeholder.
    """
    try:
        import torch
    except ImportError:
        print("torch is required: pip install torch")
        return 1

    from training.tiny_model import build_tiny

    corpus = [f"{row['prompt']} {row['chosen']} {row['rejected']}" for row in rows]
    tokenizer, model = build_tiny(corpus)
    reference = copy.deepcopy(model)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    print(f"tiny mode: local {sum(p.numel() for p in model.parameters())} "
          f"parameter model, DPO beta={args.beta}")

    def encode(text: str):
        return tokenizer(text, return_tensors="pt", truncation=True, max_length=192)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = args.max_steps if args.max_steps > 0 else len(rows)
    model.train()
    for step in range(steps):
        row = rows[step % len(rows)]
        chosen = encode(f"{row['prompt']} {row['chosen']}")
        rejected = encode(f"{row['prompt']} {row['rejected']}")

        pi_chosen = _sequence_logprob(model, chosen["input_ids"], chosen["attention_mask"])
        pi_rejected = _sequence_logprob(model, rejected["input_ids"], rejected["attention_mask"])
        with torch.no_grad():
            ref_chosen = _sequence_logprob(reference, chosen["input_ids"], chosen["attention_mask"])
            ref_rejected = _sequence_logprob(reference, rejected["input_ids"], rejected["attention_mask"])

        margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        loss = -torch.nn.functional.logsigmoid(args.beta * margin)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        print(f"step {step + 1}/{steps} | loss {float(loss.detach()):.4f} | "
              f"margin {float(margin):.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Preference-tuned tiny model saved to {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DPO the SFT adapter on verdict pairs")
    parser.add_argument("--data", default="training/data/dpo.jsonl")
    parser.add_argument("--model", default="training/checkpoints/sft")
    parser.add_argument("--out", default="training/checkpoints/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--tiny", action="store_true",
                        help="build a local 2-layer model (no download, CPU, seconds)")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.data))
    print(f"Loaded {len(rows)} preference pair(s) from {args.data}")

    if args.tiny:
        return train_tiny(rows, args)

    try:
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        print(f"Missing training dependency ({exc}).\n"
              f"pip install {REQUIRED}")
        return 1

    dataset = Dataset.from_list(rows)
    if True:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    config = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.tiny else 4,
        learning_rate=args.lr,
        logging_steps=1 if args.tiny else 10,
        save_strategy="no" if args.tiny else "epoch",
        max_length=192 if args.tiny else 1024,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=config, train_dataset=dataset, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"Preference-tuned model saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
