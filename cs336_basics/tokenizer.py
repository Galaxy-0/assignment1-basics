from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import ast
import math
import re
from typing import Any

import regex as gpt_regex


_BASE_GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _compile_gpt2_pattern() -> gpt_regex.Pattern[str]:
    return gpt_regex.compile(_BASE_GPT2_PATTERN)


def _build_special_split_pattern(special_tokens: list[str] | None) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    # Longer specials first to handle overlaps correctly
    parts = sorted((re.escape(s) for s in special_tokens), key=len, reverse=True)
    if not parts:
        return None
    return re.compile("(" + "|".join(parts) + ")")


def _byte_pair_merge(piece: bytes, ranks: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    """Greedy BPE merge for a single UTF-8 byte sequence.

    This is a straightforward implementation that recomputes adjacent pair ranks
    after each merge (O(n^2) worst-case). It's sufficient for our test sizes.
    """
    n = len(piece)
    if n == 0:
        return []
    if n == 1:
        return [piece]

    # Start with a list of 1-byte tokens
    toks: list[bytes] = [bytes([b]) for b in piece]
    if len(toks) == 1:
        return toks

    inf = math.inf

    def pairs_with_ranks(tokens: list[bytes]) -> list[tuple[float, int]]:
        return [
            (ranks.get((tokens[i], tokens[i + 1]), inf), i)
            for i in range(len(tokens) - 1)
        ]

    pairs = pairs_with_ranks(toks)
    while True:
        if not pairs:
            break
        best_rank, best_i = min(pairs, key=lambda x: x[0])
        if best_rank is inf:
            break
        # Merge best pair and recompute neighborhood
        merged = toks[best_i] + toks[best_i + 1]
        toks[best_i : best_i + 2] = [merged]
        if len(toks) == 1:
            pairs = []
            break
        # Recompute all ranks (simple and safe)
        pairs = pairs_with_ranks(toks)

    return toks


@dataclass
class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    def __post_init__(self) -> None:
        # Build id<->bytes maps
        self._id_to_bytes: dict[int, bytes] = dict(self.vocab)
        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in self._id_to_bytes.items()}

        # Ensure special tokens are present in vocab (append if missing)
        self._special_tokens: list[str] = list(self.special_tokens or [])
        next_id = max(self._id_to_bytes.keys(), default=-1) + 1
        for s in self._special_tokens:
            b = s.encode("utf-8")
            if b not in self._bytes_to_id:
                self._id_to_bytes[next_id] = b
                self._bytes_to_id[b] = next_id
                next_id += 1

        # Build merge ranks
        self._ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

        # Compile patterns
        self._base_pattern = _compile_gpt2_pattern()
        self._special_split = _build_special_split_pattern(self._special_tokens)
        self._special_set = set(self._special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | bytes | Path,
        merges_filepath: str | bytes | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct from serialized files produced by our training code/CLI.

        Expected formats:
        - vocab file: lines of the form "<id>\t<bytes_repr>" (repr of bytes object)
        - merges file: lines of the form "<bytes_repr_left> <bytes_repr_right>"
        """
        vocab: dict[int, bytes] = {}
        with Path(vocab_filepath).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    id_str, bytes_repr = line.split("\t", 1)
                    i = int(id_str)
                    b = ast.literal_eval(bytes_repr)
                    if isinstance(b, (bytes, bytearray)):
                        vocab[i] = bytes(b)
                except Exception:
                    continue

        merges: list[tuple[bytes, bytes]] = []
        with Path(merges_filepath).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    left_repr, right_repr = line.split(" ", 1)
                    left = ast.literal_eval(left_repr)
                    right = ast.literal_eval(right_repr)
                    if isinstance(left, (bytes, bytearray)) and isinstance(right, (bytes, bytearray)):
                        merges.append((bytes(left), bytes(right)))
                except Exception:
                    continue

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_span(self, text: str) -> list[int]:
        # Pre-tokenize using GPT-2 base pattern, then BPE each piece
        ids: list[int] = []
        for tok in self._base_pattern.findall(text):
            b = tok.encode("utf-8")
            # BPE merge
            pieces = _byte_pair_merge(b, self._ranks)
            for p in pieces:
                tid = self._bytes_to_id.get(p)
                if tid is None:
                    # Fallback: split to single bytes
                    for bt in p:
                        tid_b = self._bytes_to_id.get(bytes([bt]))
                        if tid_b is None:
                            raise KeyError(f"Byte {bt} not in vocabulary.")
                        ids.append(tid_b)
                else:
                    ids.append(tid)
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self._special_split or not self._special_set:
            return self._encode_span(text)

        out: list[int] = []
        parts = self._special_split.split(text)
        for part in parts:
            if part == "":
                continue
            if part in self._special_set:
                tid = self._bytes_to_id[part.encode("utf-8")]
                out.append(tid)
            else:
                out.extend(self._encode_span(part))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # Yield lazily for memory efficiency
            for tid in self.encode(chunk):
                yield tid

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        b = b"".join(self._id_to_bytes[i] for i in ids)
        # Use replacement for invalid utf-8 sequences
        return b.decode("utf-8", errors="replace")


__all__ = ["Tokenizer"]

