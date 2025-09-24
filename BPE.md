# 字节级 BPE 训练算法：完整流程

本文档详细说明了从零实现字节级 BPE（Byte Pair Encoding）训练函数的完整步骤，涵盖输入输出约定、数据结构、预分词策略、主训练循环、边界与性能考量，以及与测试对齐的注意点。

## 目标与输入/输出

### 输入
- `input_path: str | Path`：UTF-8 文本语料路径。
- `vocab_size: int`：最终词表容量上限（= 256 字节初始 + 合并产生的新 token + 特殊 token）。
- `special_tokens: list[str]`：要加入词表的特殊 token，它们不参与训练统计。

### 为什么提到"256"？
- 不是说 `vocab_size` 固定为 256；256 指"初始字节词表"的大小（8 位字节共有 2^8=256 种取值）。
- 你传入的 `vocab_size` 控制的是最终容量上限：`256（初始字节）+ merges（训练产生）+ special_tokens` 总数不得超过它。
- 如果 `vocab_size < 256 + len(special_tokens)`，训练在逻辑上不可行（连基础字节和特殊标记都放不下），应当报错或直接早停。
- 选择更大的 `vocab_size` 会带来更短的输入序列（更多合并），但也会增大嵌入矩阵和 softmax 的计算与显存开销，需要在压缩率与计算成本之间权衡。
- 使用 256 作为初始集合的动机：覆盖任何 UTF‑8 文本，无 OOV；若少于 256，将无法表达某些字节模式，导致不可逆或失败。

### 输出
- `vocab: dict[int, bytes]`：token id -> 原始字节串。初始含 256 个单字节映射，随后追加合并产物与特殊 token。
- `merges: list[tuple[bytes, bytes]]`：按顺序记录每次合并的左右字节串 `(left_bytes, right_bytes)`。

## 总体思路

1. 读入语料为字符串，构造"GPT-2 预分词正则"（需把 `special_tokens` 放在前面优先匹配）。
2. 用该正则切分出预分词字符串，统计其频次（`Counter`）。对特殊 token 本身，不计入频次（避免参与合并）。
3. 将每个预分词转成 UTF-8 字节序列，并存为 `tuple[int, ...]`（便于做字典键），得到 `word_freq: {token_tuple: count}`。
4. 初始化 `vocab`：包含 256 个字节 token（id 0..255），然后依次加入 `special_tokens` 的字节串（若未超出上限）。
5. 进入训练循环：
   - 统计所有 token 中相邻 pair 的出现次数（按 token 频次加权）。
   - 取最高频 pair，作为下一次合并目标；若不存在，提前结束。
   - 新 token 的字节串 = 左右子串字节拼接；记录到 `merges`；在 `vocab` 中分配新 id。
   - 将所有 token 序列中的该 pair 左到右、不重叠地替换为新 id，合并相同序列的频次。
   - 若词表达到 `vocab_size`，或无可合并对，则结束。
6. 返回 `(vocab, merges)`。

## 数据结构与表示

- 预分词输出：字符串序列（后续统一 `.encode('utf-8')`）。
- `word_freq: dict[tuple[int, ...], int]`：每个 token 序列（以字节 id 构成的元组）对应的频次。
- `pair_counts: Counter[tuple[int, int]]`：所有相邻对（如 `(97,98)`）的出现次数（累计各 token 的频次）。
- `vocab: dict[int, bytes]`：初始 `{i: bytes([i]) for i in range(256)}`，随后追加新 token 与特殊 token。
- `merges: list[tuple[bytes, bytes]]`：记录左、右子串的原始字节串（而非 id），便于重建 tokenizer。

## 预分词（GPT-2 正则 + 特殊 token 保护）

### 基础 GPT-2 正则模式（字符串）
```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```
需用第三方 `regex` 包（支持 Unicode 类别与负向前瞻）。

### 特殊 token 处理
将 `special_tokens` 预先 `regex.escape` 后，以 `(?:tok1|tok2|...)|<GPT-2基础模式>` 的形式拼接，确保特殊 token 优先整体匹配。

`pattern.findall(text)` 得到预分词列表；对在 `special_tokens` 集合内的分片，不计入 `word_freq` 统计。

## 详细步骤分解

### 第一步：读取和预处理语料

#### 1.1 读取文件
- 打开文件，使用 UTF-8 编码读取
- 将整个文件内容读入内存作为一个字符串

#### 1.2 构建预分词正则表达式
1. 定义基础 GPT-2 正则模式字符串
2. 处理特殊标记：
   - 对每个 special_token 进行 regex.escape() 转义
   - 用 "|" 连接所有转义后的特殊标记
   - 如果有特殊标记，构造: "(?:" + 特殊标记模式 + ")|" + GPT-2基础模式
3. 编译正则表达式

#### 1.3 执行预分词
- 使用 pattern.findall(text) 获取所有匹配的片段
- 得到一个字符串列表，每个元素是一个预分词单元

例子: "Hello world!" → ["Hello", " world", "!"]

### 第二步：统计预分词频次并转换为字节序列

#### 2.1 统计预分词频次
1. 创建一个 Counter 对象或空字典
2. 遍历每个预分词:
   - 如果该预分词在 special_tokens 集合中，跳过（不统计）
   - 否则，将该预分词的计数 +1

例子: ["Hello", " world", "Hello", "!"] → {"Hello": 2, " world": 1, "!": 1}

#### 2.2 将字符串转换为字节序列
1. 创建新字典 word_freq = {}
2. 对每个 (word, count) 对:
   - 将 word 转为字节: word_bytes = word.encode('utf-8')
   - 将字节序列转为整数元组: token_tuple = tuple(word_bytes)
   - 存储: word_freq[token_tuple] = count

例子:
- "Hi" → b'Hi' → (72, 105)
- "你好" → b'\xe4\xbd\xa0\xe5\xa5\xbd' → (228, 189, 160, 229, 165, 189)

### 第三步：初始化词表

#### 3.1 创建基础字节词表
- 创建空字典 vocab = {}
- 对于 i 从 0 到 255: vocab[i] = bytes([i])

#### 3.2 添加特殊标记到词表
1. 设置 next_token_id = 256
2. 对每个 special_token:
   - 检查是否还有空间
   - 将特殊标记转为字节并添加到词表
   - 递增 ID

#### 3.3 初始化合并列表
创建空列表: merges = []

### 第四步：主训练循环（核心）

#### 4.1 统计所有相邻对的频次
对 word_freq 中的每个 (token_tuple, freq):
- 如果 len(token_tuple) < 2: 跳过
- 否则，遍历相邻对并累加频次

#### 4.2 选择最高频的相邻对
- 如果 pair_counts 为空，训练结束
- 找最大频次的 pair: best_pair = max(pair_counts, key=pair_counts.get)

#### 4.3 创建新 token 并更新词表
1. 从词表获取左右字节串
2. 合并字节串: new_token_bytes = left_bytes + right_bytes
3. 记录合并规则: merges.append((left_bytes, right_bytes))
4. 添加新 token 到词表

#### 4.4 执行全量替换（左到右、不重叠）

替换算法的关键：
```python
i = 0
while i < len(token_tuple):
    if i+1 < len(token_tuple) and (token_tuple[i], token_tuple[i+1]) == best_pair:
        # 替换为新 token
        result.append(new_token_id)
        i += 2  # 跳过两个已合并的 token
    else:
        # 保留当前 token
        result.append(token_tuple[i])
        i += 1
```

#### 4.5 更新循环控制变量
- 递增 token ID: next_token_id += 1
- 检查终止条件: 如果 next_token_id >= vocab_size，词表已满，退出循环

### 第五步：返回结果
返回元组: (vocab, merges)

## 完整流程示例

### 输入数据
```
文本: "aa bb aa"
vocab_size: 258
special_tokens: []
```

### 执行过程

**初始状态:**
- 预分词: ["aa", " ", "bb", " ", "aa"]
- 统计频次: {"aa": 2, " ": 2, "bb": 1}
- 转为字节:
  - "aa" → (97, 97): 频次 2
  - " " → (32,): 频次 2
  - "bb" → (98, 98): 频次 1

**第一轮:**
1. 统计相邻对: (97, 97) 出现 2 次，(98, 98) 出现 1 次
2. 最高频: (97, 97)
3. 创建新 token: vocab[256] = b'aa'
4. 替换: (97, 97) → (256,)

**第二轮:**
1. 统计相邻对: (98, 98) 出现 1 次
2. 最高频: (98, 98)
3. 创建新 token: vocab[257] = b'bb'
4. 替换: (98, 98) → (257,)

**终止:**
next_token_id = 258 >= vocab_size = 258，返回结果

## 关键实现细节和注意事项

### 1. 数据结构选择
**为什么用 tuple[int, ...] 作为键？**
- tuple 是不可变的，可以作为字典键
- 整数比较比字节串比较更快
- 便于索引和切片操作

### 2. 左到右不重叠替换的细节
错误示例（会重叠）：
```python
for i in range(len(token_tuple)-1):
    if (token_tuple[i], token_tuple[i+1]) == best_pair:
        # 替换但不跳过，会导致重叠
```

正确示例（不重叠）：
```python
i = 0
while i < len(token_tuple):
    if i+1 < len(token_tuple) and (token_tuple[i], token_tuple[i+1]) == best_pair:
        # 替换并跳过两个位置
        i += 2
    else:
        # 保留当前，移动一个位置
        i += 1
```

### 3. 特殊标记处理要点
- 必须在正则表达式前面，确保优先完整匹配
- 统计频次时必须排除，避免被拆分合并
- 添加到词表时占用 ID 空间，影响 vocab_size 计算

### 4. 性能优化技巧
- 使用 Counter 而不是手动维护字典
- 批量处理而不是逐个处理
- 避免不必要的类型转换（如反复 bytes ↔ tuple）
- 合并后的 word_freq 可以用 defaultdict(int) 避免检查键存在

### 5. 边界情况处理
- 空文件：直接返回只有 256 个字节的初始词表
- vocab_size < 256：逻辑错误，应报错或早停
- 所有 token 长度为 1：无相邻对，提前结束
- 特殊标记超过剩余空间：只添加能放下的部分

## 背景与参数说明

### 为什么做"字节级"BPE
- **通用性**：任何 Unicode 文本都能被 UTF-8 编码成 0–255 的字节序列，初始 256 个字节 token 能覆盖所有输入，无 OOV。
- **可逆性与稳定性**：在字节层面合并与还原是逐字节确定的，不依赖语言学规则，跨语言一致。
- **跨域适配**：不同语种/符号（含 emoji）无需为字符集特判；常见字节序列会被合并成更长 token，提高压缩率。
- **权衡**：非 ASCII 字符一个字符通常占多字节，初始会变长，但 BPE 通过统计常见序列逐步缩短。

### 为什么需要预分词（GPT-2 正则）
- **约束合并范围**：先把文本切成更大的"词/符号/空白"单元，避免跨词/跨空白的合并。
- **一致性**：测试与业界实现使用相同的正则，确保结果可比、可复现。
- **性能**：预分词减少候选 pair 的组合爆炸，统计更高效。
- **特殊 token 保护**：把特殊 token 放在正则前面优先匹配，保证它们不被拆分。

### 参数含义与影响

#### input_path（语料来源）
- 决定统计到的高频片段，从而决定 merges 的形状
- 小语料运行更快、合并更"窄"；大语料更全面但耗时
- 需确保 UTF-8 解码一致

#### vocab_size（词表上限）
- 越大，能合并的 token 越多，序列更短；嵌入矩阵更大、内存占用更高
- 过小：压缩不足；过大：收益递减且占用资源
- 定义包含：256 个初始字节 + 特殊 token + 训练产生的新 token

#### special_tokens（特殊标记）
- 用于边界/控制（如 `<|bos|>`、`<|eot|>`、系统提示等）
- 必须作为"原子"存在，不能被拆分或参与合并
- 仅加入词表，不参与频次统计与 merges

### 为什么 merges 是"有序的字节对列表"
- BPE 的应用阶段依赖合并顺序：早先产生的合并拥有更高优先级
- 返回字节对（而不是 id 对）可移植性更强
- 顺序决定了分词的确定性：同样文本在同样 merges 下会得到一致的分词结果

### 左到右、不重叠替换的原因
- 原始 BPE 算法的关键约束：在某一轮，只允许把"本轮选中的 pair"各自合并一次
- 确保每一轮的计数与替换是对齐的，训练具有可解释的"逐步构造"特性

### 性能门槛与复杂度
- 每轮：统计相邻对 O(L)，全量替换 O(L)；总计 O(R·L)
- 在测试用小语料下，朴素实现即可达标（< 1.5s）
- 常数优化：使用 `Counter` 与不可变 `tuple` 做键

### 可复现性
给定固定语料、参数与实现细节（尤其是预分词与替换规则），训练输出的 `merges` 顺序是确定的；这对下游对齐与测试通过至关重要。

## 常见坑与注意事项

1. 忘记把 `special_tokens` 提前匹配或从统计中移除，导致 `merges` 出现 `<|` 等片段
2. 词表上限计算错误：记得包含 256 字节 + 新 token + 特殊 token
3. 替换逻辑重叠（未 `i += 2`），或顺序错误（非左到右）
4. 用字符串层面合并而非字节层面合并
5. 使用标准库 `re` 而非 `regex` 包，导致不支持 Unicode 类别

## 与测试对齐的要点

- 使用 `regex` 包与给定 GPT-2 正则；普通 `re` 可能太慢或不支持
- 预分词阶段需要把 `special_tokens` 放在正则前缀位置优先匹配，并从统计中剔除
- `run_train_bpe` 测试会读取 `tests/fixtures` 下的小语料，比较 `merges` 与 `vocab` 的集合
- 性能测试目标：在该小语料上 < 1.5 秒