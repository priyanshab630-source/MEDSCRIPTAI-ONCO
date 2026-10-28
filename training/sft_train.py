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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def train_tiny(rows: list, args) -> int:
    """A real optimisation loop on a tiny local model - no downloads, no GPU.

    This is the plumbing test: tokenise the SFT rows, take gradient steps on the
    language-modelling loss, save the model. It proves the dataset is trainable;
    it does not claim the resulting weights are useful.
    """
    try:
        import torch
    except ImportError:
        print("torch is required: pip install torch")
        return 1

    from training.tiny_model import build_tiny

    texts = [row["text"] for row in rows]
    tokenizer, model = build_tiny(texts)
    print(f"tiny mode: local {sum(p.numel() for p in model.parameters())} "
          "parameter model, full fine-tune")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = args.max_steps if args.max_steps > 0 else len(texts)
    model.train()
    for step in range(steps):
        text = texts[step % len(texts)]
        encoded = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=192, padding=False)
        if encoded["input_ids"].shape[1] < 2:
            continue
        output = model(**encoded, labels=encoded["input_ids"])
        output.loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        print(f"step {step + 1}/{steps} | loss {float(output.loss.detach()):.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Tiny model saved to {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SFT a small model on verdict traces")
    parser.add_argument("--data", default="training/data/sft.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="training/checkpoints/sft")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--tiny", action="store_true",
                        help="build a local 2-layer model (no download, CPU, seconds)")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args(argv)

    rows = [to_text(row) for row in load_rows(Path(args.data))]
    print(f"Loaded {len(rows)} SFT example(s) from {args.data}")

    if args.tiny:
        return train_tiny(rows, args)

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(f"Missing training dependency ({exc}).\n"
              f"pip install {REQUIRED}")
        return 1

    dataset = Dataset.from_list(rows)
    lora = None
    if True:
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
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
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=1 if args.tiny else 10,
        save_strategy="no" if args.tiny else "epoch",
        gradient_checkpointing=not args.tiny,
        max_length=192 if args.tiny else 1024,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset,
        peft_config=lora, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"{'Tiny model' if args.tiny else 'Adapter'} saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
