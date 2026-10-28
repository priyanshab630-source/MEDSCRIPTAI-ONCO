"""Preference tuning (DPO) on the pairs built by training/build_dataset.py.

Stage two of the plan: after SFT teaches the output shape, DPO pushes the model
away from the collapsed-confidence answer and towards the structured verdict
(with abstention when the evidence is thin).

    pip install transformers datasets peft accelerate trl
    python training/dpo_train.py --data training/data/dpo.jsonl \
        --model training/checkpoints/sft --out training/checkpoints/dpo
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED = "transformers, datasets, peft, trl"


def load_rows(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DPO the SFT adapter on verdict pairs")
    parser.add_argument("--data", default="training/data/dpo.jsonl")
    parser.add_argument("--model", default="training/checkpoints/sft")
    parser.add_argument("--out", default="training/checkpoints/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args(argv)

    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        print(f"Missing training dependency ({exc}).\n"
              f"pip install {REQUIRED}")
        return 1

    rows = load_rows(Path(args.data))
    dataset = Dataset.from_list(rows)
    print(f"Loaded {len(dataset)} preference pair(s) from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    config = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        max_length=1024,
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
