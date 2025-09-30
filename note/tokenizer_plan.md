# Tokenizer 实现方案（设计与步骤）

目标
- 实现一个字节级 BPE Tokenizer，接口如下：
  - `__init__(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None)`
  - `@classmethod from_files(vocab_filepath, merges_filepath, special_tokens=None)`
  - `encode(text: str) -> list[int]`
  - `encode_iterable(iterable: Iterable[str]) -> Iterator[int]`
  - `decode(ids: list[int]) -> str`
- 需求：与 GPT‑2/tiktoken 行为一致（在测试语料上），支持用户提供的 `special_tokens`（若不在 vocab 中则追加）。

核心设计
- 词表映射：
  - `id_to_bytes: dict[int, bytes]` 与 `bytes_to_id: dict[bytes, int]`（构造时从传入 `vocab` 建立）。
  - 若传入了 `special_tokens`，将其 UTF‑8 编码后检查是否在 `bytes_to_id`；没有则追加到 `vocab` 尾部。
- BPE merges 排名：
  - `ranks: dict[tuple[bytes, bytes], int]`，按 `merges` 列表顺序赋 rank（越小优先级越高）。
- 预分词：
  - 使用 GPT‑2 基础正则（与训练一致）：
    `"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"`
  - 在预分词前，优先按 `special_tokens` 拆分文本（使用按长度降序的正则 alternation，确保重叠时长者优先匹配）。
- BPE 合并（编码阶段）：
  - 对每个非特殊片段：先按 GPT‑2 正则切分，再将每个分片转为 UTF‑8 字节串，执行字节级 BPE 合并。
  - 合并算法：简单、正确的贪心实现（每次选择 rank 最小的相邻对进行合并，合并后重新计算相邻对的排名；O(n^2) 最坏，但测试规模可接受）。
  - 将合并后的每个 bytes 片段映射为 token id；若某片段不在 `vocab`，回退为逐字节输出（单字节一定在词表中）。
- 解码：
  - 将 ids 转回 bytes 拼接后，用 UTF‑8 `errors='replace'` 解码得到字符串（保证遇到无效 UTF‑8 时使用 U+FFFD）。
- 流式编码 `encode_iterable`：
  - 逐块（如逐行）调用 `encode` 并 `yield`，避免一次性加载整个大文件，满足内存要求。
- 文件加载 `from_files`：
  - 兼容我们 BPE 训练脚本/CLI 的落盘格式：
    - vocab：每行 `"<id>\t<bytes_repr>"`
    - merges：每行 `"<bytes_repr_left> <bytes_repr_right>"`
  - 使用 `ast.literal_eval` 解析 bytes 的 repr。

重叠特殊 token 的处理
- 例如同时存在 `<|endoftext|>` 与 `<|endoftext|><|endoftext|>`：
  - 构造拆分正则时先按长度降序排序，确保同起点时优先匹配更长的特殊 token（测试覆盖）。

正确性与与 tiktoken 的对齐
- 预分词模式一致（GPT‑2 基础正则）。
- 合并顺序通过 `ranks` 保证与 merges 一致。
- 特殊 token 在编码阶段不会被拆分；解码时按字节自然还原为对应字符串。

性能与限制
- 合并实现为简洁的 O(n^2) 贪心法；对测试文本与常规输入开销可接受。
- `encode_iterable` 为流式，内存占用受控；`encode` 会在大文本上占用较多内存，符合测试预期。

与测试对应关系
- `tests/adapters.py::get_tokenizer` 适配器返回我们实现的 `Tokenizer` 实例。
- `tests/test_tokenizer.py` 的以下点被覆盖：
  - 空字符串、单字符、Unicode 文本、文件往返对比（roundtrip）。
  - 与 tiktoken 的 ID 匹配（在 GPT‑2 vocab/merges 上）。
  - 特殊 token 与重叠特殊 token 的稳定保留。
  - `encode_iterable` 的流式编码与内存限制场景。

如何运行测试
- 先确保适配器已连接（本仓库已在 `tests/adapters.py:get_tokenizer` 连接）。
- 运行：`uv run pytest tests/test_tokenizer.py`。

文件位置
- 实现：`cs336_basics/tokenizer.py`
- 适配器：`tests/adapters.py:get_tokenizer`

备注
- 本实现不依赖 GPT‑2 的 bytes↔unicode 映射；测试夹具已将 GPT‑2 vocab/merges 转换为 bytes 表示，因此可直接使用 UTF‑8 处理文本与特殊 token。
