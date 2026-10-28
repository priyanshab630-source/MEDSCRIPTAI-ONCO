"""A tiny, fully local model + tokenizer for pipeline smoke tests.

`--tiny` mode exists so the fine-tuning scripts can be executed end to end on a
CPU in seconds, with no model download and no GPU. It is a *plumbing* test: a
2-layer, 32-dimension GPT-2 with a word-level vocabulary built from the training
text. It proves the data format, the collator, the trainer wiring and the save
path - not that fine-tuning improves anything.

For real training, drop `--tiny` and pass `--model <instruction-tuned checkpoint>`.
"""

from pathlib import Path


def _vocabulary(texts: list) -> list:
    """Word-level vocabulary from the training text, plus special tokens."""
    words = set()
    for text in texts:
        words.update(text.split())
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for word in sorted(words):
        vocab.setdefault(word, len(vocab))
    return vocab


def build_tiny(texts: list, max_length: int = 192):
    """Returns (tokenizer, model) - both small enough for a laptop CPU."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = _vocabulary(texts)
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
        eos_token="<eos>", model_max_length=max_length,
    )

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=max_length,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer, GPT2LMHeadModel(config)


def is_tiny(model_name: str) -> bool:
    return model_name in ("tiny", "") or not Path(model_name).exists()
