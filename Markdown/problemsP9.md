Problem (train_bpe): BPE Tokenizer Training (15 points)  
Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. 

Your BPE training function should handle (at least) the following input parameters:  input_path: str Path to a text file with BPE tokenizer training data.  vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).  special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.  Your BPE training function should return the resulting vocabulary and merges:  vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).  merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.  To test your BPE training function against our provided tests, you will first need to implement the test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py. Your implementation should be able to pass all tests. Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex package in Python is, if anything, even faster.

任务（train_bpe）：BPE 分词器训练（15 分）
  交付内容：编写一个函数，给定训练文本文件路径，训练
  一个（字节级）BPE 分词器。

  你的 BPE 训练函数至少要处理以下输入参数：

  - input_path: str 指向 BPE 分词器训练数据的文本
  文件。
  - vocab_size: int 正整数，表示最终词表的最大大小
  （包含初始字节词表、合并后产生的新词项以及任何特
  殊 token）。
  - special_tokens: list[str] 要加入词表的字符串列
  表。这些特殊 token 不会影响 BPE 训练流程。

  函数应返回训练得到的词表和合并规则：

  - vocab: dict[int, bytes] 分词器词表，键是词表中的
  token ID（int），值是对应的字节串。
  - merges: list[tuple[bytes, bytes]] BPE 合并序列。
  列表中的每个元素是一个字节对 (<token1>, <token2>)，
  表示 <token1> 与 <token2> 被合并。合并列表需要按创
  建顺序排列。

  要用我们提供的测试验证你的 BPE 训练函数，首
  先要实现测试适配器（tests/adapters.py 中的
  run_train_bpe）。然后运行 uv run pytest tests/
  test_train_bpe.py。你的实现需要通过所有测试。

  可选（可能耗时较多）：你可以用系统语言实现训练
  核心部分，例如 C++（可考虑 cppyy）或 Rust（使用
  PyO3）。如果这样做，要注意哪些操作需要复制，哪些可
  以直接读取 Python 内存，并确保提供构建说明，或保证
  只用 pyproject.toml 就能构建。另外要注意，GPT-2 的
  正则在多数引擎中支持不好，也常常很慢。我们已验证
  Oniguruma 在这方面较快并支持负向前瞻，而 Python 的
  regex 包甚至更快。