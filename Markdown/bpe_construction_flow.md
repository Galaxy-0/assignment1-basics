# 从零构建 BPE 的渐进式流程

这份笔记用“骨架 → 细节”的方式梳理 Byte Pair Encoding (BPE) 的训练过程，帮助在阅读 `train_bpe.ipynb` 时保持全局视角。

## Level 0：一张流程图
```
[语料加载]
      ↓
[初始分词：字符 / 字节]
      ↓
[统计全部相邻 token pair 的频次]
      ↓
[挑选频次最高的 pair]
      ↓
[合并 pair → 生成新 token]
      ↓
[更新语料中的 token 序列]
      ↓
[记录合并步骤到 merges]
      ↓
[达到限制?]
   ├─ 否 → 回到“统计 pair”
   └─ 是 → 输出 merges + vocab
```
目标：在 `max_merges`（或目标词表大小）约束下，得到 merges 列表和最终 vocab。

## Level 1：朴素伪代码骨架
```
def train_bpe(corpus, max_merges):
    tokens = tokenize_to_chars(corpus)
    merges = []

    for _ in range(max_merges):
        pair_counts = count_adjacent_pairs(tokens)
        if not pair_counts:
            break
        best_pair = argmax(pair_counts)
        tokens = merge_pair(tokens, best_pair)
        merges.append(best_pair)

    vocab = build_vocab(tokens, merges)
    return merges, vocab
```
- `tokenize_to_chars`：把文本拆成最细粒度的符号序列。
- `count_adjacent_pairs`：遍历语料，统计所有相邻符号对。
- `merge_pair`：把选中 pair 替换成新 token，并更新所有序列。
- `build_vocab`：把初始符号和新 token 汇总成词表。

先用极小语料（例如 `"low low lower"`）验证收敛过程，打印每轮 tokens 变化，打牢直觉。

## Level 2：Notebook 映射指南
- 对照 `train_bpe.ipynb`，为上面的每个函数找到对应的代码单元。
- 每个单元写一句“输入 → 输出”的描述，确认它在流程中的位置。
- Notebook 中出现的额外结构（批量处理、性能优化、可视化等），在掌握主干后再补充理解。

## Level 3：验证与扩展
- **玩具测试**：挑 2-3 条短句手算 2-3 次合并，和程序输出比对。
- **调试输出**：在朴素实现中打印中间变量（tokens、pair_counts、best_pair）。
- **性能考量**：确认何时需要用到更高效的数据结构（比如基于 `Counter` 或 `numpy` 的实现）。
- **语料细节**：结合 notebook，补充特殊 token、字节级转换、序列化等工程细节。

把这份“骨架”手册和 notebook 对照阅读，可以先掌握从 0 到 1 的主流程，再平滑过渡到 notebook 里的完整实现。
