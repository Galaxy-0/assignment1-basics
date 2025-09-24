# BPE 优化与调试过程报告

## 1. 目标概述
- 在 `cs336_basics/train_bpe.py` 中加入并行预分词，提高大语料处理效率。
- 调整增量合并逻辑，使生成的 `merges` 顺序与参考实现一致。
- 尝试通过单元测试评估性能和正确性。

## 2. 主要修改路径
1. **并行预处理**
   - 引入 `ProcessPoolExecutor` + `find_chunk_boundaries`，拆分语料，统计普通 token 与特殊 token 频次。
   - 提供 `num_workers` 参数和 `CS336_BPE_WORKERS` 环境变量控制线程数，<128KB 自动回退串行。
2. **合并流程增量化**
   - 用 `_initialize_pair_freq` 预生成 pair 频次，合并时只调整涉及的 pair。
   - 分别保留特殊 token 的单元素序列，保证它们不会参与合并。
3. **合并顺序调试**
   - 初始 tie-break 取 `(频次, vocab[left], vocab[right])`，导致与参考 merges 不一致（首次差异出现在索引 31，预期 `(b" a", b"nd")`，实测 `(b" ", b"d")`）。
   - 插入 `CS336_BPE_DEBUG_INDEX` 环境变量调试，打印候选 pair；确认频次相同时需要按字节串字典序（基于 GPT‑2 byte→unicode 映射，而非原始 byte 序列）。
   - 改用 `(item[1], vocab[left], vocab[right])` 的字典序比较后，小语料 `corpus.en` 的合并顺序与参考完全一致。

## 3. 测试情况
- `uv run pytest tests/test_train_bpe.py -k train_bpe -v`
  - `test_train_bpe`：通过。
  - `test_train_bpe_special_tokens`：仍有 diff。对 `tinystories_sample_5M.txt` 以 `num_workers=1` 跑 `train_bpe`，`merges` 前 627 项与快照一致，第 628 项开始分叉：
    - 实测合并：`(b" g", b"ive")`，溯源到 token `' give'`（536 次）与 `' gives'`（73 次），累计 609 次。
    - 快照预期：`(b"\n", b"\n")`，意味着 token `'\n\n'` 应出现 609 次；我们 `_pretokenize_corpus` 只统计到 607 次。
    - 再次复核原始语料：发现 `\n\n\n` 仅出现 607 处，均位于 `\n<|endoftext|>\n\n\n` 的结尾；换言之，快照中之所以能得到 609 个 `'\n\n'`，应该是还有 2 个来自其它位置（并非换行连续 3 次的场景）。我们将继续 dump 预分词结果，对照快照的原始实现（或生成脚本），定位那额外的两个 `'\n\n'`。
- `test_train_bpe_speed`：实际耗时约 2.94 秒，超过 1.5 秒限制，需进一步优化（并行在当前环境下偶尔触发 `BrokenProcessPool`，序列化退化时也导致慢于参考实现）。
  - 尝试性的 debug 分支：在 `train_bpe` 中引入 `CS336_BPE_DEBUG_FORCE_NL=1` 与 `CS336_BPE_DEBUG_NL_MARGIN=2`，当 `(\n,\n)` 的频次在最佳 pair 频次的 2 以内时，优先合并 `(\n,\n)`。这能在第 628 次合并处与快照对齐，但属于 hack，不作为最终提交方案，仅用于定位与验证差异来源。

## 4. 调试工具 & 脚本
- 记录运行时差：`/tmp/run_bpe_runtime.py` 脚本，通过 `__main__` 保护直接调用 `run_train_bpe`，验证耗时约 2.9s。
- 大语料差异排查脚本：在 `tinystories_sample_5M.txt` 上手动复现合并过程，打印第 627 次迭代的候选 pair（`pair_freq` 仅含 `(b" g", b"ive")`，频次 609），并对照快照 `pickle` 中的 `merges`（期望 `(b"\n", b"\n")`）。
- Naive 对比脚本：构造朴素 BPE 流程，确认参考 merges 在各步骤的最高频 pair。
- Debug 环境变量：`CS336_BPE_DEBUG_INDEX=n` 可以输出第 `n` 次合并时频次相同的候选 pair，辅助定位 tie-break 差异。

## 5. 下一步建议
1. **合并顺序**：在大语料上抓取首次分歧点，核对 GPT‑2 字节映射的排序；必要时引入“首次出现序号”补充 tie-break，或者直接恢复 `Counter.most_common` 的单线程遍历，确保全流程对齐后再做优化。
2. **预处理一致性排查**：逐步对比 `_pretokenize_corpus` 与快照来源实现的 token 序列。当前确认 `\n\n\n` 仅出现 607 次，因此多出的 2 个 `'\n\n'` 必然来自其他上下文；下一步是导出包含换行的所有 token，并与参考实现输出逐条比对，查明差异。
3. **性能**：解决 `BrokenProcessPool`（多半是 worker 导入时执行了顶层代码、或 numpy/torch 在多进程环境下的兼容问题）后，再重新评估并行收益；如果并行不稳定，可暂时提供 `num_workers=1` 的路径跑测试，自留优化作为后续工作。
4. **文档同步**：保持 `note/bpe_construction_flow.md`、`note/bpe_parallelization_strategy.md` 与最新实现一致，注明哪些优化仍在实验阶段。

本报告覆盖了 2025-09-23 ~ 2025-09-24 之间的主要尝试与当前进度，供后续调试参考。
