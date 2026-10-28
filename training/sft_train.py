"""Supervised fine-tune on the SFT set built by training/build_dataset.py.

Not run in CI (needs a GPU and a multi-GB base model). The point of shipping it
is that the data format, the LoRA setup and the evaluation hook are all decided,
so the step is a command rather than a design question.

    pip install transformers datasets peft accelerate trl
    python training/sft_train.py --data training/data/sft.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct --out training/checkpoints/sft

Swap --model for any instruction-tuned base you have access to; nothing here is
specific to one provider.
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED = "transformers, datasets, peft, trl"


def load_rows(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_text(row: dict) -> dict:
    """Flatten chat messages into the single text field trainers expect."""
    parts = []
    for message in row["messages"]:
        parts.append(f"<|{message['role']}|>\n{message['content']}")
    return {"text": "\n".join(parts)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SFT a small model on verdict traces")
    parser.add_argument("--data", default="training/data/sft.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="training/checkpoints/sft")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args(argv)

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(f"Missing training dependency ({exc}).\n"
              f"pip install {REQUIRED}")
        return 1

    rows = [to_text(row) for row in load_rows(Path(args.data))]
    dataset = Dataset.from_list(rows)
    print(f"Loaded {len(dataset)} SFT example(s) from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        max_length=1024,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset,
        peft_config=lora, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
