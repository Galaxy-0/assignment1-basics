# BPE 并行化预处理策略

## 背景
当前 `cs336_basics/train_bpe.py` 的实现采用串行流程：一次性读取完整语料 → 预分词 → 统计频次 → 迭代训练。随着语料和 `vocab_size` 增大，预分词和初始频次统计会占用显著时间。仓库提供的 `cs336_basics/pretokenization_example.py` 展示了如何通过特殊 token 边界拆分大文件，这为我们并行化前置步骤提供了模板。

## 基本思路
1. **分块语料**：使用 `find_chunk_boundaries` 按 `<|endoftext|>` 边界定位多个区间，确保任何特殊 token 不被拆断。
2. **并行预分词/统计**：将每个 `(start, end)` 区间分配给独立的 worker 进程或线程，在局部执行 GPT‑2 正则预分词，并生成局部的词频 `Counter`（排除特殊 token）。
3. **合并局部统计**：汇总所有 worker 的词频计数，得到全局的 `pre_counter` 或 `token_counts`，为后续 BPE 迭代提供输入。
4. **串行 BPE 迭代**：保持现有的增量式 `train_bpe` 主循环逻辑，确保合并顺序和数值稳定性。

## 细化步骤
### 1. 利用 `find_chunk_boundaries`
- 函数签名：`find_chunk_boundaries(file, desired_num_chunks, split_special_token)`。
- 读取大文件时，先根据文件长度均匀估算分界，再按 4KB `mini_chunk` 向后读取寻找 `<|endoftext|>`。
- 输出的边界列表可以直接用于切分文件：`zip(boundaries[:-1], boundaries[1:])`。

### 2. 并行执行预分词
- 在主进程中创建 `ProcessPoolExecutor` 或 `multiprocessing.Pool`。
- 每个 worker：
  - 用 `seek` + `read` 获取自己的字节片段，并 decode 成字符串（忽略错误）。
  - 调用和训练一致的 GPT‑2 正则，得到预分词结果。
  - 过滤 `<|endoftext|>` 等特殊 token，返回局部 `Counter[str]`。
- 主进程收集并将这些 Counter 叠加 (`+=`)，得到全局 `pre_counter`。

### 3. 构建全局 `token_counts`
- 与当前实现一致：把字符串 `tok_str` `encode('utf-8')` 成 `tuple[int, ...]`。
- 如需进一步优化，可在 worker 侧直接返回“局部 token_counts”或“局部 pair_freq”，减少主进程工作量。

### 4. 保持单线程 BPE 迭代
- 合并频次 → 选择最高频 pair → 更新词表/merges → 增量维护 pair 频次，这部分仍由主进程执行。
- 串行迭代可保证合并顺序 determinism，避免并发加锁的复杂性。

## 并行化后的潜在收益
- **预分词速度**：几乎线性随 CPU 核数增长。对数十 MB 语料，可将初始处理时间从数十秒降至几秒。
- **缓存友好**：每个 worker 处理连续的字节块，避免了在大文件上反复 seek/scan。
- **扩展性**：更容易插入中间统计（如直接统计局部 pair 频次）。

## 注意事项
- **特殊 token 完整性**：务必使用 `find_chunk_boundaries` 输出的边界，禁止在特殊 token 中间切分块。
- **文件句柄**：如果 worker 无法共享同一 file object，可传递路径，让 worker 打开 `with open(..., 'rb')` 再读取对应区间。
- **内存占用**：对极大语料，可考虑流式合并 Counter，或使用 `Counter.update` 逐块叠加，避免一次性堆积太多数据。
- **序列一致性**：并行阶段产生的 `pre_counter` 和串行计算结果应完全相同；建议在开发时对比两种流程输出，确认无差异后再启用并行。

## 总结
通过 `pretokenization_example.py` 的分块技巧，我们可以把 BPE 的高耗时部分（预分词和初始统计）切分后并行执行，最大化 CPU 利用率；而实际的合并循环仍保持串行，从而兼顾正确性与性能。下一步可以在 `train_bpe.py` 中封装一个 `parallel_pretokenize_and_count` 辅助函数，将这套策略整合进正式代码，同时保留串行回退路径，便于在不同环境下切换。
