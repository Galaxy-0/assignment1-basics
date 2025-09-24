from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Iterable

import regex


# GPT-2 pre-tokenization pattern
# Reference: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
_BASE_GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _build_pattern_with_specials(special_tokens: Iterable[str]) -> regex.Pattern[str]:
    """构造包含特殊token的预分词正则。"""
    specials = [s for s in special_tokens]
    if specials:
        # Sort by length desc so longer specials take precedence
        specials.sort(key=len, reverse=True)
        escaped = [regex.escape(s) for s in specials]
        pattern_str = "|".join(escaped + [_BASE_GPT2_PATTERN])
    else:
        pattern_str = _BASE_GPT2_PATTERN
    return regex.compile(pattern_str)


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


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE：
      1. 读语料，构造正则
      2. 预分词 + 统计频次（Counter）
      3. 初始化 byte 词表，追加特殊 token
      4. 循环统计最高频 pair -> 合并 -> 更新词表/merges
    """
    # 1) Read corpus and build regex pattern with specials
    text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    pattern = _build_pattern_with_specials(special_tokens)

    # 2) Pretokenize and count frequencies of each pre-token string
    special_set = set(special_tokens)
    pre_counter: Counter[str] = Counter()
    for token in _iter_pre_tokens(text, pattern):
        if token in special_set:
            continue
        pre_counter[token] += 1

    # 3) Initialize byte vocab and add special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    seen_specials: set[str] = set()
    for s in special_tokens:
        if s in seen_specials:
            continue
        if len(vocab) >= vocab_size:
            break
        vocab[next_id] = s.encode("utf-8")
        next_id += 1
        seen_specials.add(s)

    # Represent each unique pre-token as a tuple of symbol IDs
    token_counts: dict[tuple[int, ...], int] = {}
    for tok_str, cnt in pre_counter.items():
        b = tok_str.encode("utf-8")
        token_tuple = tuple(b)
        if token_tuple:
            token_counts[token_tuple] = token_counts.get(token_tuple, 0) + cnt

    merges: list[tuple[bytes, bytes]] = []
    pair_freq = _initialize_pair_freq(token_counts)

    # 4) Iteratively merge most frequent pairs
    while len(vocab) < vocab_size and pair_freq:
        best_pair, best_count = max(
            pair_freq.items(),
            key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]),
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
