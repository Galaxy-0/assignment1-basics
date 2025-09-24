阶段 1：熟悉项目结构

  - 阅读 README.md 和作业 PDF，弄清各模块的目标。
  - 浏览 cs336_basics/ 下的各个 .py 文件，了解已经提
  供的函数签名和注释。
  - 打开 tests/ 目录下的测试文件，特别是 tests/
  test_nn_utils.py、tests/test_tokenizer.py、tests/
  test_model.py 等，看看测试期望你的函数返回什么。

  阶段 2：实现基础数学工具（优先级高）

  - 关注 cs336_basics/nn_utils.py。实现 softmax、
  cross_entropy、gradient_clipping 等基础函数。
  - 每实现一个函数，立刻运行对应测试：uv run pytest
  tests/test_nn_utils.py -k softmax 等。
  - 确保这些基础函数通过测试，稍后模型组件要用到
  它们。

  阶段 3：优化器与学习率调度

  - 打开 cs336_basics/optimizer.py，实现 AdamW 类及
  get_lr_cosine_schedule。
  - 对应地运行 uv run pytest tests/
  test_optimizer.py。

  阶段 4：数据处理模块

  - 实现 cs336_basics/data.py 中的 get_batch（用于从
  数据集中取一批序列）。
  - 跑测试：uv run pytest tests/test_data.py.

  阶段 5：分词器与BPE训练

  - 文件 cs336_basics/tokenizer.py：实现 Tokenizer 的
  编码、解码逻辑，以及与 tiktoken 对齐的细节。
  - 文件 cs336_basics/train_bpe.py：实现 BPE 训练流程
  （统计对、合并规则、特殊 token 处理等）。
  - 这些部分相对复杂，建议：
      1. 阅读测试 tests/test_tokenizer.py 和 tests/
  test_train_bpe.py，了解测试用例。
      2. 先实现最简单的编码/解码，再加特殊 token
  支持。
      3. 每写完一段功能就跑 uv run pytest tests/
  test_tokenizer.py -k roundtrip 或速度测试。
  - 注意性能测试 test_train_bpe_speed，实现时要考虑效
  率（尽量用字典或计数器）。

  阶段 6：神经网络组件

  - 在 cs336_basics/model.py 中，从最底层算子开始
  实现：
      1. Linear
      2. Embedding
      3. SiLU / SwiGLU
      4. scaled_dot_product_attention（含 4D 版本）
      5. MultiheadSelfAttention（同时关注带 RoPE 的
  版本）
      6. RMSNorm
      7. TransformerBlock
      8. TransformerLM
  - 同时实现 cs336_basics/rope.py 中的 RoPE 函数（如
  果分离到独立文件）。
  - 每完成一个模块就跑对应单元测试，例如 uv run
  pytest tests/test_model.py::test_linear。
  - 对照测试中的参数（如张量形状、dtype、mask 逻
  辑），确保实现符合要求。
  - tests/test_model.py 涵盖所有组件，最好分项目运
  行，最后再跑一次全部。

  阶段 7：模型序列化 / 训练辅助

  - cs336_basics/serialization.py 中的
  save_checkpoint、load_checkpoint 等函数。
  - 跑 uv run pytest tests/test_serialization.py。

  阶段 8：综合测试与提交准备

  - 确认前面各阶段测试都通过后，运行完整测试：uv
  run pytest。
  - 借助 make_submission.sh 生成提交包，确认命令执行
  无误。
  - 使用 git status 查看是否还有未提交的调试文件（如
  notebook 或临时脚本）。

  练习方式建议

  1. 每次改动不多的功能，就 git add -p 分段提交，防止
  把一次性太多改动混在一起。
  2. 多利用注释和测试判断参数形状。遇到不确定的地
  方，可以在 Python 交互式 shell 里 import torch 做小
  实验。
  3. 阅读测试里对 NotImplementedError 的预期，有些测
  试会写详细注释告诉你想要什么样的行为。

  遇到困难时

  - 注意测试报错里给出的对比信息，常会指出实际值和期
  望值的差异。
  - 可以 print 或使用断点调试（pdb）检查中间变量。
  - 如果某个模块逻辑复杂，考虑先写伪代码或在纸上画
  流程。

  按这个路线循序渐进实现，你会对 Transformer 组件和
  分词体系有较系统的理解。需要具体函数的详细步骤或示
  例，随时告诉我，我可以按函数拆开讲。


  # 第一部分 BPE

  - 数据来源：tests/fixtures/
  tinystories_sample_5M.txt 和 corpus.en 是指定的迷你
  数据集，先确认路径、编码（UTF‑8）。
  - 预分词阶段
      1. 读取整个文本到内存（这份样本只有几 MB，可以
  承受）。
      2. 用 GPT‑2 的预分词正则切分；注意要把课程指定
  的特殊 token 放在正则前面匹配，这样它们不会被拆开。
      3. 统计每个预分词出现次数，但对特殊 token 本身
  不要计入统计，以免它们参与 merge。
  - 初始化词表
      1. 先建立基础 256 个 byte token，ID 对应字
  节值。
      2. 把特殊 token 追加到词表末尾（顺序按传入列
  表），同时检查总 vocab 大小别超出限制。
      3. 词表大到上限就直接返回，不需要再训练。
  - BPE 训练循环
      1. 把每个预分词转换为 “字节 ID 元组”（例如
  b"the" → (116, 104, 101)），用频次保存。
      2. 循环直到词表达到上限或没有可合并对：
          - 统计所有 token 中相邻 pair 的出现次数
  （用 Counter）。
          - 选频次最高的 pair；若没有 pair，退出。
          - 新 token 的字节串就是两个旧 token 字节拼
  接，把 (tok_a, tok_b) 记到 merges 列表。
          - 词表中分配新的 ID，保存新字节串。
          - 遍历所有 token，把选中的 pair 替换成新
  ID（比如 tuple -> list -> tuple 或直接遍历创建新
  tuple）。注意：替换时要成段替换（一次 pair → 一个
  ID），避免重叠错误。
          - 更新词表大小计数，继续下一轮。
  - 特殊 token 保护
      - 由于预分词阶段已经避免特殊 token 出现在
  word_freq 中，后面自然不会被合并；但还是要确保
  词表里特殊 token 的字节串不会参与 pair 统计（即
  word_freq 里不出现它们）。
  - 性能考虑
      - 样本数据不大，可一次读入；不过要避免在循环里
  重复创建庞大结构。
      - pair_counts.most_common(1) 可以快速拿到最高
  频 pair。
      - 更新 token 时直接构建新 tuple（不可变结构便于
  做 dict key），并累加频次。
      - 每轮遍历所有 token 是 O(词表大小 × token 长
  度)，在这个小数据集上是可接受的，跑一次速度测试确认
  小于 1.5 秒。
      - 如果还慢，可考虑把词频储存在列表或数组，以减
  少字典操作开销。
  - 自测步骤
      1. 先在极小语料上写一个测试脚本验证输出格式
  （vocab 映射、merges 列表条数等）。
      2. 完成后运行 uv run pytest tests/
  test_train_bpe.py -k train_bpe。
      3. 如果 special_tokens 相关测试没过，检查正则是
  否匹配到了 <| 这样的片段。
      4. 保证训练循环在 vocab 达到上限或 pair_counts
  空时结束，避免死循环。

  按照这个思路，你可以先写框架，再逐步填充每个子步骤
  的代码。如果某个环节（例如正则构造、统计逻辑）遇到
  问题，可以把你的代码片段贴出（别期待完整答案），我
  能帮你分析 bug 或提供进一步的线索。