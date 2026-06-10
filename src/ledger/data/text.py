"""Language-model data path: tokenizer + packed token stream + sources.

Two sources behind one interface:
  - "pseudo": deterministic pseudo-English, generated locally (no network) for CPU smoke tests.
  - "fineweb": real FineWeb-Edu via HuggingFace `datasets` (used on Modal, where network exists).

Default tokenizer is byte-level (vocab=256, no downloads, identical locally and on Modal).
Massive activations are architecture-driven, so byte-level is a valid substrate for the
mechanistic study; GPT-2 BPE is available for the standard, paper-facing run.
"""
from __future__ import annotations

import random
from typing import List

import torch


class ByteTokenizer:
    """UTF-8 byte tokenizer. vocab=256, no external files."""
    vocab_size = 256
    eot = 10  # newline as a cheap document separator

    def encode(self, s: str) -> List[int]:
        return list(s.encode("utf-8", errors="ignore"))

    def decode(self, ids) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def build_tokenizer(name: str = "bytes"):
    if name == "bytes":
        return ByteTokenizer()
    if name == "gpt2":
        import tiktoken  # downloads the BPE vocab once (needs network; fine on Modal)
        enc = tiktoken.get_encoding("gpt2")

        class _GPT2:
            vocab_size = enc.n_vocab
            eot = enc.eot_token

            def encode(self, s: str):
                return enc.encode_ordinary(s)

            def decode(self, ids):
                return enc.decode(list(ids))

        return _GPT2()
    raise ValueError(f"unknown tokenizer {name!r}")


_WORDS = (
    "the of and to in a is that it for on with as are was be by this from at or an but not "
    "they he she we you i which their there has have had will would can could one all about "
    "time people year work school water light world life part number system program question"
).split()


def sample_pseudo_text(n_chars: int = 200_000, seed: int = 0) -> str:
    """Deterministic pseudo-English: real-ish word/sentence structure, no network. Smoke only."""
    rng = random.Random(seed)
    out, total = [], 0
    while total < n_chars:
        sent = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(5, 16)))
        sent = sent.capitalize() + ".\n" if rng.random() < 0.15 else sent.capitalize() + ". "
        out.append(sent)
        total += len(sent)
    return "".join(out)[:n_chars]


def load_token_ids(source: str, tokenizer, max_tokens: int, seed: int = 0) -> torch.Tensor:
    """Return a 1-D LongTensor of token ids from the chosen source."""
    if source == "pseudo":
        text = sample_pseudo_text(n_chars=max_tokens * 2, seed=seed)  # bytes ~ chars for ascii
        ids = tokenizer.encode(text)[:max_tokens]
        return torch.tensor(ids, dtype=torch.long)
    if source == "fineweb":
        from datasets import load_dataset  # guarded: only available on Modal image
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        ids: List[int] = []
        for ex in ds:
            ids.extend(tokenizer.encode(ex["text"]))
            ids.append(tokenizer.eot)
            if len(ids) >= max_tokens:
                break
        return torch.tensor(ids[:max_tokens], dtype=torch.long)
    raise ValueError(f"unknown data source {source!r}")


class TokenDataset:
    """Packed token array; yields random contiguous (x, y=next-token) windows (nanoGPT-style)."""

    def __init__(self, ids: torch.Tensor, block_size: int):
        assert ids.dim() == 1 and len(ids) > block_size + 1, "need a 1-D token stream longer than block_size"
        self.ids = ids
        self.block_size = block_size

    def batch(self, batch_size: int, device: str, generator: torch.Generator):
        hi = len(self.ids) - self.block_size - 1
        ix = torch.randint(0, hi, (batch_size,), generator=generator)
        x = torch.stack([self.ids[i: i + self.block_size] for i in ix])
        y = torch.stack([self.ids[i + 1: i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)
