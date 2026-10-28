# Fine-tuning: SFT then DPO

The framework's value is the *shape* of its answer — four-way verdict, decomposed
uncertainty, explicit abstention — so that is exactly what the training data
teaches. Nothing here is run in CI: both scripts need a GPU and a multi-GB base
model. The data pipeline and the reward design are tested.

## Step 1 — collect reports

```bash
python main.py --drug SHP099 --target SHP2 --disease "KRAS-mutant NSCLC" --json > reports/shp099_nsclc.json
python main.py --drug RMC-4550 --target SHP2 --disease "KRAS-mutant CRC" --json > reports/rmc4550_crc.json
python main.py --drug CompoundX --target SHP2 --json > reports/unknown.json
```

Run the whole demo matrix, including the `ABSTAIN` case — abstention is the
behaviour most worth teaching, and it is the one a base model never produces.

## Step 2 — build the datasets

```bash
python training/build_dataset.py reports/*.json --out training/data
```

| File | Shape | Purpose |
|---|---|---|
| `sft.jsonl` | `messages: [system, user, assistant]` | teach the output format |
| `dpo.jsonl` | `prompt`, `chosen`, `rejected` | prefer structure over confidence |

`chosen` is the real structured verdict. `rejected` is the collapsed failure
mode the framework exists to avoid: a single confidence number with no
uncertainty decomposition and no abstention.

## Step 3 — train

```bash
pip install transformers datasets peft accelerate trl

python training/sft_train.py --data training/data/sft.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct --out training/checkpoints/sft

python training/dpo_train.py --data training/data/dpo.jsonl \
    --model training/checkpoints/sft --out training/checkpoints/dpo
```

Both scripts take any instruction-tuned base model; LoRA keeps the memory
footprint inside a single consumer GPU.

## Step 4 — evaluate (the part that matters)

Fine-tuning is only worth it if it changes behaviour on the cases that count:

- **Abstention precision** — does the tuned model say `ABSTAIN` on hypotheses
  with no evidence, instead of inventing a verdict?
- **Uncertainty fidelity** — are the six dimensions populated with defensible
  values, or filled with plausible noise?
- **Format stability** — does it still emit valid JSON after DPO?

`python main.py --demo` remains the regression suite; add the tuned model behind
`src/tools/llm.py` and re-run it.
