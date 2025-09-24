from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import regex


from cs336_basics.pretokenization_example import find_chunk_boundaries

# GPT-2 pre-tokenization pattern
# Reference: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
_BASE_GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _compile_base_pattern() -> regex.Pattern[str]:
    """编译 GPT-2 基础预分词正则（不包含特殊 token）。"""
    return regex.compile(_BASE_GPT2_PATTERN)


def _split_on_specials(text: str, special_tokens: Iterable[str]) -> list[tuple[bool, str]]:
    """按特殊 token 切分文本，返回 (is_special, piece)。保留特殊 token 本身。"""
    specials = [s for s in special_tokens]
    if not specials:
        return [(False, text)]
    specials.sort(key=len, reverse=True)
    pat = regex.compile("(" + "|".join(regex.escape(s) for s in specials) + ")")
    parts = pat.split(text)
    out: list[tuple[bool, str]] = []
    for p in parts:
        if p == "":
            continue
        out.append((p in specials, p))
    return out


def _iter_pre_tokens(text: str, pattern: regex.Pattern[str]) -> Iterable[str]:
    """用指定正则遍历所有预分词。"""
    # regex.findall returns the full match string (no capture groups used)
    yield from pattern.findall(text)


def _initialize_pair_freq(
    token_counts: dict[tuple[int, ...], int]
) -> dict[tuple[int, int], int]:
    """统计初始 pair 频次。"""
    pair_freq: dict[tuple[int, int], int] = {}
    for token, freq in token_counts.items():
        if len(token) < 2:
            continue
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
    return pair_freq


def _apply_pair_freq_delta(
    pair_freq: dict[tuple[int, int], int],
    token: tuple[int, ...],
    freq: int,
    delta: int,
) -> None:
    """根据 token 调整 pair 频次。"""
    if len(token) < 2:
        return
    for i in range(len(token) - 1):
        pair = (token[i], token[i + 1])
        new_count = pair_freq.get(pair, 0) + delta * freq
        if new_count:
            pair_freq[pair] = new_count
        else:
            pair_freq.pop(pair, None)


def _count_tokens_in_chunk(
    corpus_path: str,
    start: int,
    end: int,
    base_pattern_str: str,
    base_pattern_flags: int,
    special_tokens: tuple[str, ...],
) -> tuple[Counter[str], Counter[str]]:
    """Worker helper: read a chunk from disk and return token & special frequencies.
    先按特殊 token 切分，再对普通片段用 GPT-2 基础正则分词。
    """
    if end <= start:
        return Counter(), Counter()

    with open(corpus_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    if not chunk:
        return Counter(), Counter()

    text = chunk.decode("utf-8", errors="ignore")
    base_pattern = regex.compile(base_pattern_str, base_pattern_flags)
    counter: Counter[str] = Counter()
    special_counter: Counter[str] = Counter()
    for is_special, piece in _split_on_specials(text, special_tokens):
        if is_special:
            special_counter[piece] += 1
        else:
            for tok in base_pattern.findall(piece):
                counter[tok] += 1
    return counter, special_counter


def _uniform_boundaries(file_size: int, desired_chunks: int) -> list[int]:
    if file_size <= 0 or desired_chunks <= 0:
        return [0, file_size]
    chunk_size = max(1, file_size // desired_chunks)
    boundaries = [0]
    for i in range(1, desired_chunks):
        boundaries.append(min(file_size, i * chunk_size))
    boundaries.append(file_size)
    return sorted(set(boundaries))


def _pretokenize_corpus(
    input_path: Path,
    pattern: regex.Pattern[str],
    special_tokens: list[str],
    num_workers: int | None,
) -> tuple[Counter[str], Counter[str]]:
    """Count pre-token frequencies, optionally in parallel using chunking."""

    corpus_path = str(input_path)
    special_tuple = tuple(special_tokens)
    special_set = set(special_tokens)

    file_size = input_path.stat().st_size if input_path.exists() else 0
    if num_workers is None:
        env_workers = os.environ.get("CS336_BPE_WORKERS")
        if env_workers and env_workers.isdigit():
            num_workers = int(env_workers)
        else:
            cpu_count = os.cpu_count() or 1
            num_workers = min(8, max(1, cpu_count))

    # Fall back to sequential path for small corpora or single worker
    if num_workers <= 1 or file_size < 128_000:  # 128 KB threshold
        text = input_path.read_text(encoding="utf-8", errors="ignore")
        segments = _split_on_specials(text, special_tokens)

        # No debug span dump in production

        counter: Counter[str] = Counter()
        special_counter: Counter[str] = Counter()
        for is_special, piece in segments:
            if is_special:
                special_counter[piece] += 1
            else:
                for tok in pattern.findall(piece):
                    counter[tok] += 1
        return counter, special_counter

    pattern_str = pattern.pattern
    pattern_flags = pattern.flags

    special_token_bytes = (
        special_tuple[0].encode("utf-8") if special_tuple else None
    )

    with open(input_path, "rb") as f:
        if special_token_bytes:
            boundaries = find_chunk_boundaries(
                f, num_workers, special_token_bytes
            )
        else:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            boundaries = _uniform_boundaries(size, num_workers)

    if len(boundaries) <= 1:
        return Counter(), Counter()

    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    effective_workers = min(num_workers, len(chunk_ranges))

    counter: Counter[str] = Counter()
    special_counter: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(
                _count_tokens_in_chunk,
                corpus_path,
                start,
                end,
                pattern_str,
                pattern_flags,
                special_tuple,
            )
            for start, end in chunk_ranges
        ]
        for future in futures:
            chunk_counter, chunk_specials = future.result()
            counter.update(chunk_counter)
            special_counter.update(chunk_specials)

    return counter, special_counter


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE：
      1. 读语料，构造正则
      2. 预分词 + 统计频次（Counter）
      3. 初始化 byte 词表，追加特殊 token
      4. 循环统计最高频 pair -> 合并 -> 更新词表/merges
    """
    corpus_path = Path(input_path)
    # Use base GPT-2 pattern; handle special tokens by splitting before tokenization
    pattern = _compile_base_pattern()

    # 2) Pretokenize and count frequencies of each pre-token string
    pre_counter, special_counter = _pretokenize_corpus(
        corpus_path, pattern, special_tokens, num_workers
    )

    # 3) Initialize byte vocab and add special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    seen_specials: set[str] = set()
    special_to_id: dict[str, int] = {}
    for s in special_tokens:
        if s in seen_specials:
            continue
        if len(vocab) >= vocab_size:
            break
        vocab[next_id] = s.encode("utf-8")
        special_to_id[s] = next_id
        next_id += 1
        seen_specials.add(s)

    # Represent each unique pre-token as a tuple of symbol IDs
    token_counts: dict[tuple[int, ...], int] = {}
    for tok_str, cnt in pre_counter.items():
        b = tok_str.encode("utf-8")
        token_tuple = tuple(b)
        if token_tuple:
            token_counts[token_tuple] = token_counts.get(token_tuple, 0) + cnt

    for special_tok, cnt in special_counter.items():
        if special_tok not in special_to_id:
            continue
        token_tuple = (special_to_id[special_tok],)
        token_counts[token_tuple] = token_counts.get(token_tuple, 0) + cnt

    merges: list[tuple[bytes, bytes]] = []
    pair_freq = _initialize_pair_freq(token_counts)

    # 4) Iteratively merge most frequent pairs
    while len(vocab) < vocab_size and pair_freq:
        best_pair, best_count = max(
            pair_freq.items(),
            key=lambda item: (
                item[1],
                vocab[item[0][0]],
                vocab[item[0][1]],
            ),
        )
        if best_count <= 0:
            break

        a_id, b_id = best_pair
        new_bytes = vocab[a_id] + vocab[b_id]
        new_id = next_id
        next_id += 1
        vocab[new_id] = new_bytes
        merges.append((vocab[a_id], vocab[b_id]))

        new_token_counts: defaultdict[tuple[int, ...], int] = defaultdict(int)
        for token, freq in token_counts.items():
            if len(token) < 2:
                new_token_counts[token] += freq
                continue

            i = 0
            changed = False
            new_token_list: list[int] = []
            while i < len(token):
                if (
                    i < len(token) - 1
                    and token[i] == a_id
                    and token[i + 1] == b_id
                ):
                    new_token_list.append(new_id)
                    i += 2
                    changed = True
                else:
                    new_token_list.append(token[i])
                    i += 1

            if not changed:
                new_token_counts[token] += freq
                continue

            _apply_pair_freq_delta(pair_freq, token, freq, delta=-1)
            new_token = tuple(new_token_list)
            _apply_pair_freq_delta(pair_freq, new_token, freq, delta=1)
            new_token_counts[new_token] += freq

        token_counts = new_token_counts
        pair_freq.pop(best_pair, None)

    return vocab, merges
