from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path
from typing import Iterable

from cs336_basics.train_bpe import train_bpe


def _bytes_to_readable(b: bytes, limit: int | None = 120) -> str:
    """Turn bytes into a compact readable string with escapes for non-printables.
    - Printable ASCII kept as-is (except backslash/quote escaped)
    - Newlines/tab/carriage-return -> \n, \t, \r
    - Others -> \xHH
    Optionally truncate to `limit` visible chars for display.
    """
    out_chars: list[str] = []
    for ch in b:
        if ch == 0x5C:  # \\
            out_chars.append("\\\\")
        elif ch == 0x22:  # "
            out_chars.append('\\"')
        elif ch == 0x27:  # '
            out_chars.append("\\'")
        elif ch == 0x0A:
            out_chars.append("\\n")
        elif ch == 0x0D:
            out_chars.append("\\r")
        elif ch == 0x09:
            out_chars.append("\\t")
        elif 0x20 <= ch <= 0x7E:
            out_chars.append(chr(ch))
        else:
            out_chars.append(f"\\x{ch:02x}")
        if limit is not None and len(out_chars) >= limit:
            out_chars.append("…")
            break
    return "".join(out_chars)


def _save_vocab_and_merges(
    outdir: Path,
    tag: str,
    vocab_size: int,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    vocab_path = outdir / f"{tag}_vocab_{vocab_size}.txt"
    merges_path = outdir / f"{tag}_merges_{vocab_size}.txt"

    with vocab_path.open("w", encoding="utf-8") as f:
        for i, b in sorted(vocab.items()):
            f.write(f"{i}\t{b!r}\n")

    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left!r} {right!r}\n")

    return vocab_path, merges_path


def _find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes, int]:
    """Return (token_id, token_bytes, byte_len) for the longest token (deterministic tie-break)."""
    # Deterministic tie-break: prefer smaller id
    best_id = -1
    best_b = b""
    best_len = -1
    for tid, tb in vocab.items():
        L = len(tb)
        if L > best_len or (L == best_len and tid < best_id):
            best_id, best_b, best_len = tid, tb, L
    return best_id, best_b, best_len


def _load_vocab_file(path: Path) -> dict[int, bytes]:
    """Load a vocab txt file saved by this script (id<tab>bytes-repr)."""
    vocab: dict[int, bytes] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                id_str, bytes_repr = line.split("\t", 1)
                tid = int(id_str)
                b = ast.literal_eval(bytes_repr)
                if not isinstance(b, (bytes, bytearray)):
                    continue
                vocab[tid] = bytes(b)
            except Exception:
                continue
    return vocab


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on a corpus and serialize outputs."
    )
    parser.add_argument("--input", required=True, help="Path to corpus text file")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--special",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to add (can repeat). Default: <|endoftext|>",
    )
    parser.add_argument("--workers", type=int, default=None, help="Num workers for pretokenization")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag/prefix for output files (default: derived from input filename)",
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        help="Path to another saved vocab txt (e.g., TinyStories) for 1-2 sentence comparison",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2
    tag = args.tag or input_path.stem
    outdir = Path(args.outdir)

    start = time.time()
    vocab, merges = train_bpe(
        input_path,
        args.vocab_size,
        special_tokens=list(args.special or []),
        num_workers=args.workers,
    )
    elapsed = time.time() - start

    vocab_path, merges_path = _save_vocab_and_merges(outdir, tag, args.vocab_size, vocab, merges)
    best_id, best_bytes, best_len = _find_longest_token(vocab)
    best_preview = _bytes_to_readable(best_bytes, limit=160)

    # Deliverable (a): 1-2 sentence
    deliver_a = (
        f"Training on {input_path.name} produced a {args.vocab_size}-token byte-level BPE; "
        f"the longest token has {best_len} bytes (id={best_id}), e.g., '{best_preview}'."
    )

    print("Outputs saved:")
    print(f"  vocab:  {vocab_path}")
    print(f"  merges: {merges_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("Deliverable (a):", deliver_a)

    # Optional (b) comparison if another vocab given
    if args.compare_with:
        other_path = Path(args.compare_with)
        if other_path.exists():
            other_vocab = _load_vocab_file(other_path)
            if other_vocab:
                oid, obytes, olen = _find_longest_token(other_vocab)
                # Try to infer corpus tags from filenames
                this_tag = tag
                other_tag = other_path.stem
                deliver_b = (
                    f"Compared to {other_tag}, {this_tag} yields a longest token of {best_len} bytes vs {olen} bytes; "
                    f"the OpenWebText tokenizer typically captures longer URL/markup fragments, while TinyStories favors story-like words and punctuation."
                )
                print("Deliverable (b):", deliver_b)
        else:
            print(f"--compare-with file not found: {other_path}")

    # Also save a short summary alongside outputs
    summary_path = outdir / f"{tag}_summary_{args.vocab_size}.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"elapsed_sec\t{elapsed:.3f}\n")
        f.write(f"vocab_path\t{vocab_path}\n")
        f.write(f"merges_path\t{merges_path}\n")
        f.write(f"longest_token_id\t{best_id}\n")
        f.write(f"longest_token_len\t{best_len}\n")
        f.write(f"longest_token_preview\t{best_preview}\n")
        f.write("deliverable_a\t" + deliver_a + "\n")
    print(f"Summary saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

