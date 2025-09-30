# OpenWebText 32k BPE 训练与交付方案（计划与思路）

目标与交付
- 训练一个 byte-level BPE 分词器（vocab_size=32,000）于 OpenWebText 文本（如 `data/owt_valid.txt`）。
- 按要求序列化 `vocab` 与 `merges` 至磁盘，并给出：
  - (a) 1–2 句话：报告“最长 token”及其合理性。
  - (b) 1–2 句话：与 TinyStories 训练得到的分词器对比。

资源约束（题目要求）
- 时间：≤ 12 小时（CPU 即可，无 GPU）。
- 内存：≤ 100 GB（当前实现通常远低于此）。

现有实现（cs336_basics/train_bpe.py）概览
- 预分词：
  - 先对文本按特殊 token 切分（若传入，默认 `<|endoftext|>`），保证不跨文档边界；否则走“均匀切块”。
  - 对非特殊片段使用 GPT‑2 基础正则进行预分词（使用 `regex` 库）。
  - 大文件时支持多进程进行预分词（在 `_pretokenize_corpus` 内决定，阈值约 1MB；macOS/Notebook 下建议从脚本调用避免 spawn 问题）。
- 词表初始化：
  - 256 个字节 token + 追加 `special_tokens`（若提供且有空间）。
- 训练循环：
  - 统计 pair 频次（`_initialize_pair_freq`）。
  - 每轮选择最高频 pair 合并；并对受影响的 token 更新 pair 频次（增量更新 `_apply_pair_freq_delta`）。
  - tie‑break：在频次相同情况下按 `(count, left_bytes, right_bytes)` 字节序做确定性选择。

时间与瓶颈
- 预分词阶段：可通过 `num_workers` 并行，通常几十秒以内（对 300MB 量级）。
- 主要耗时在合并主循环（纯 Python 单线程，每轮：选最大 pair + 重写 `token_counts`）。
- 基于 30MB 样本的本机校准外推：vocab≈32k 的 OWT 全量训练预估 9–12 小时（详见对话中的推算）。

执行方案（不立刻跑重任务也能就绪）
1) 使用一键 CLI 跑法（已加入脚本）
- 命令模块：`cs336_basics/cli_train_bpe.py`
- 功能：训练、落盘、计算“最长 token”、可选与 TinyStories 的 vocab 做一句话对比，并输出简要 summary。
- 示例命令：
  - 训练 OWT 32k 并落盘（建议 6–8 进程）：
    - `uv run python -m cs336_basics.cli_train_bpe --input data/owt_valid.txt --vocab-size 32000 --special "<|endoftext|>" --workers 8 --tag owt_valid`
  - 训练完成后文件位于：
    - `outputs/owt_valid_vocab_32000.txt`
    - `outputs/owt_valid_merges_32000.txt`
    - `outputs/owt_valid_summary_32000.txt`（含最长 token、用时等）
  - 与 TinyStories 做一句话对比（可选）：
    - 在已有 TinyStories 的 `..._vocab_*.txt` 情况下，追加参数：
    - `--compare-with outputs/tinystories_vocab_10000.txt`

2) 先做小样本“节拍校准”，再决定是否全量跑
- 目的：更精确地评估本机时间，以便决定是否夜间跑全量。
- 步骤：
  - 抽取 30MB 子集：`python - <<'PY'\nfrom pathlib import Path\nsrc=Path('data/owt_valid.txt'); dst=Path('data/owt_valid_30M.sample')\nN=30*1024*1024\nwith src.open('rb') as f, dst.open('wb') as g: g.write(f.read(N))\nprint('done', dst.exists(), dst.stat().st_size)\nPY`
  - 计时跑一次小 vocab（如 2048）：
    - `uv run python - <<'PY'\nimport time\nfrom cs336_basics.train_bpe import train_bpe\nstart=time.time(); train_bpe('data/owt_valid_30M.sample', 2048, special_tokens=['<|endoftext|>'], num_workers=8)\nprint('elapsed_s=', round(time.time()-start,3))\nPY`
  - 通过“每次合并平均耗时 × 需要的合并步数 × 数据量倍率”，外推到 32k 全量用时。

3) 交付语句模板（跑完后直接套用/微调）
- (a) OpenWebText：
  - “Training on OpenWebText produced a 32k byte‑level BPE; the longest token has X bytes (e.g., '...'). It mostly captures long URL/markup fragments, which makes sense for web text.”
- (b) 与 TinyStories 对比：
  - “Compared to TinyStories, OpenWebText yields longer tokens and more URL/markup punctuation merges, while TinyStories favors narrative words and simpler punctuation due to its domain.”

保证正确性的要点
- 特殊 token：即便 OWT 语料中没有 `<|endoftext|>`，也不影响；预分词会走“均匀切块”。
- 确定性：频次相同的 tie‑break 使用字节序，保证可重复。
- 预分词并行：仅加速预分词阶段；主循环仍在 Python 端单线程（影响总耗时的主要部分）。
- Notebook 注意：在 Jupyter 中多进程可能有 spawn 问题；建议使用模块方式运行 CLI（`python -m cs336_basics.cli_train_bpe`）。

若需要进一步提速（可选的改进方向，按收益排序）
1) 使用 max‑heap 维护“最佳 pair”，避免每轮线性 `max(pair_freq.items())` 扫描（需在每次更新时同步 heap 的计数）。
2) 维护 pair→token 的反向索引，只重写包含该 pair 的 token，而不是扫描全部 `token_counts`。
3) 预分词 worker 初始化中预编译正则，降低进程启动开销（大语料/多进程时收益有限但稳健）。
4) 观察性策略（不改变语义）：增加合并进度的日志节拍，便于估算剩余时间（对性能影响可控）；或分阶段落盘 checkpoint（若允许中断续跑，则需扩展代码）。

内存与稳定性
- `token_counts` 与 `pair_freq` 是主要内存消费者；32k 规模在 300MB 量级语料上通常在数 GB 以内（远低于 100GB 上限）。
- 不建议在训练中“丢弃低频 token”来省内存（会改变合并路径，影响可复现性与测试一致性）。

对比 TinyStories 的流程（用于(b)）
- TinyStories 训练可先用同脚本/或已有 notebook 生成 vocab；再用 `--compare-with` 进行一句话摘要对比。
- 预期差异：
  - OWT：较多 URL、HTML、emoji 及标点组合，最长 token 往往很长；
  - TinyStories：偏叙事文本，短词频高，最长 token 相对短，分布更集中在词/子词。

总结
- 本方案不要求你立刻跑全量；先做小样本节拍校准，再决定是否夜间跑全量（预计 9–12 小时，满足 ≤12h 约束）。
- 训练与落盘一条命令即可，跑完终端与 `outputs/*_summary_32000.txt` 会给出交付所需的一两句话素材与“最长 token”。

