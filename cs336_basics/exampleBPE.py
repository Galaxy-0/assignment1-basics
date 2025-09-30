本项目来源于和鲸社区，使用转载需要标注来源
作者: 天海一直在
来源: https://www.heywhale.com/mw/project/689709e023583639fc675b5c
import os
import heapq
import regex
import time
import random
import multiprocessing
from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict, Any, Union
import mmap
import re
from collections import defaultdict

# GPT-2预分词模式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def load_and_sample_data(file_path: str, sample_size: int = 22000, special_token: str = "<|endoftext|>") -> str:
    """内存映射方式加载并采样文档"""
    try:
        with open(file_path, "r+", encoding='utf-8', errors='ignore') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                documents = []
                start = 0
                while start < len(mm):
                    end = mm.find(special_token.encode('utf-8'), start)
                    if end == -1:
                        doc = mm[start:].decode('utf-8', errors='replace').strip()
                        if doc:
                            documents.append(doc)
                        break
                    
                    doc = mm[start:end].decode('utf-8', errors='replace').strip()
                    if doc:
                        documents.append(doc)
                    start = end + len(special_token)
                
                if len(documents) > sample_size:
                    documents = random.sample(documents, sample_size)
                
                return special_token.join(documents)
    except Exception as e:
        raise IOError(f"加载数据集失败: {e}")

def gpt2_bytes_to_unicode_local() -> Dict[int, str]:
    """字节到Unicode映射"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)) 
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def pre_tokenize_document(doc: str, bytes_to_unicode_map: Dict[int, str]) -> List[List[str]]:
    """预分词处理单个文档"""
    tokens = regex.findall(GPT2_SPLIT_PATTERN, doc, flags=regex.UNICODE)
    sequences = []
    for token in tokens:
        token_unicode = ''.join(bytes_to_unicode_map[b] for b in token.encode('utf-8'))
        sequences.append(list(token_unicode))
    return sequences

def parallel_pre_tokenize(documents: List[str], num_processes: int, bytes_to_unicode_map: Dict[int, str]) -> List[List[str]]:
    """并行预分词优化"""
    if num_processes <= 1:
        return [seq for doc in documents for seq in pre_tokenize_document(doc, bytes_to_unicode_map)]
    
    with multiprocessing.Pool(
        num_processes,
        initializer=init_worker,
        initargs=(bytes_to_unicode_map,)
    ) as pool:
        results = list(tqdm(
            pool.imap(pre_tokenize_worker, documents, chunksize=50),
            total=len(documents),
            desc="预分词",
            mininterval=1
        ))
    return [seq for doc_sequences in results for seq in doc_sequences]

# 全局变量用于多进程
global_worker_byte_map = None
def init_worker(byte_map: Dict[int, str]):
    global global_worker_byte_map
    global_worker_byte_map = byte_map

def pre_tokenize_worker(doc: str) -> List[List[str]]:
    return pre_tokenize_document(doc, global_worker_byte_map)

class BPEIndex:
    """高效索引结构用于BPE合并"""
    def __init__(self, sequences: List[List[str]]):
        self.sequences = sequences # 存储所有文本序列
        self.pair_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int) # 统计字节对频率
        self.pair_positions: DefaultDict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list) # 记录字节对位置
        self.heap = []  # 最大堆（存最高频字节对）
        self.heap_entries: Dict[Tuple[str, str], Any] = {} # 堆条目快速访问
         
        # 初始化索引 一次性统计所有相邻字节对的出现位置和频率——将不可行的O(N²)问题转化为可处理的O(N log N)
        for seq_idx, seq in enumerate(sequences):
            for pos in range(len(seq) - 1):
                pair = (seq[pos], seq[pos + 1])
                self.pair_counts[pair] += 1
                self.pair_positions[pair].append((seq_idx, pos))
        
        # 构建堆 将高频字节对（>1次）加入最大堆，让 get_most_frequent() 能 O(1) 获取最高频对。
        for pair, count in self.pair_counts.items():
            if count > 1:  # 只添加计数大于1的pair
                entry = [-count, pair]
                heapq.heappush(self.heap, entry)
                self.heap_entries[pair] = entry
    
    def get_most_frequent(self) -> Tuple[str, str]:
        """快速返回当前最高频字节对（跳过已被合并的无效条目）"""
        while self.heap:
            neg_count, pair = self.heap[0]
            # 检查pair是否仍然有效
            if pair not in self.heap_entries:
                heapq.heappop(self.heap)
                continue
                
            current_count = self.pair_counts.get(pair, 0)
            
            # 检查计数是否匹配且大于1
            if -neg_count == current_count and current_count > 1:
                return pair
            # 否则移除无效条目
            heapq.heappop(self.heap)
            if pair in self.heap_entries:  # 确保条目存在
                del self.heap_entries[pair]
        return None
    
    def merge_pair(self, pair: Tuple[str, str], new_token: str) -> int:
        """合并字符对并更新索引"""
        if pair not in self.pair_positions or not self.pair_positions[pair]:
            return 0
        
        # 按序列和位置分组
        positions_by_seq = defaultdict(list)
        for seq_idx, pos in self.pair_positions[pair]:
            positions_by_seq[seq_idx].append(pos)
        
        merge_count = 0
        for seq_idx, positions in positions_by_seq.items():
            seq = self.sequences[seq_idx]
            # 按位置倒序排序
            positions.sort(reverse=True)
            last_merged_pos = -2
            
            for pos in positions:
                # 检查是否已被前面的合并影响
                if pos >= len(seq) - 1 or pos <= last_merged_pos:
                    continue
                if seq[pos] != pair[0] or seq[pos + 1] != pair[1]:
                    continue
                
                # 执行合并
                seq[pos] = new_token
                del seq[pos + 1]
                merge_count += 1
                last_merged_pos = pos
                
                # 更新左侧pair
                if pos > 0:
                    left_pair = (seq[pos - 1], pair[0])
                    self._update_pair_count(left_pair, -1)
                    
                    new_left_pair = (seq[pos - 1], new_token)
                    self._update_pair_count(new_left_pair, 1)
                    self._add_position(new_left_pair, seq_idx, pos - 1)
                
                # 更新右侧pair
                if pos < len(seq) - 1:
                    right_pair = (pair[1], seq[pos + 1])
                    self._update_pair_count(right_pair, -1)
                    
                    new_right_pair = (new_token, seq[pos + 1])
                    self._update_pair_count(new_right_pair, 1)
                    self._add_position(new_right_pair, seq_idx, pos)
        
        # 清理已合并的pair
        if pair in self.pair_counts:
            del self.pair_counts[pair]
        if pair in self.pair_positions:
            del self.pair_positions[pair]
        if pair in self.heap_entries:
            # 标记为无效，稍后清理
            self.heap_entries[pair] = None
        
        return merge_count
    
    def _update_pair_count(self, pair: Tuple[str, str], delta: int):
        """更新字符对计数"""
        if delta == 0:
            return
        
        # 确保pair存在于字典中
        if pair not in self.pair_counts:
            self.pair_counts[pair] = 0
            
        new_count = self.pair_counts[pair] + delta
        self.pair_counts[pair] = new_count
        
        # 确保计数不为负
        if new_count < 0:
            new_count = 0
            self.pair_counts[pair] = 0
        
        if pair in self.heap_entries and self.heap_entries[pair] is not None:
            # 更新堆条目
            self.heap_entries[pair][0] = -new_count
            heapq.heapify(self.heap)
        elif new_count > 1:  # 只添加计数大于1的pair
            # 新建堆条目
            entry = [-new_count, pair]
            heapq.heappush(self.heap, entry)
            self.heap_entries[pair] = entry
    
    def _add_position(self, pair: Tuple[str, str], seq_idx: int, pos: int):
        """添加新位置到索引"""
        self.pair_positions[pair].append((seq_idx, pos))

def run_train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str] = ["<|endoftext|>"],
    num_processes: int = 8,
    sample_size: int = 22000,
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 参数验证
    base_vocab_size = 256 + len(special_tokens)
    if vocab_size < base_vocab_size:
        raise ValueError(f"vocab_size至少需{base_vocab_size}")
    
    # 1. 字节到Unicode映射
    bytes_to_unicode_map = gpt2_bytes_to_unicode_local()
    unicode_to_bytes_map = {v: bytes([k]) for k, v in bytes_to_unicode_map.items()}
    
    # 2. 初始化词汇表
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    existing_bytes = set(vocab.values())
    
    # 3. 添加特殊token
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in existing_bytes and len(vocab) < vocab_size:
            vocab[next_token_id] = st_bytes
            existing_bytes.add(st_bytes)
            next_token_id += 1
    
    # 4. 加载并采样数据
    print(f"📖 从 {input_path} 加载并采样 {sample_size} 个文档...")
    text = load_and_sample_data(input_path, sample_size, special_tokens[0])
    
    # 5. 分割文档
    escaped_tokens = [re.escape(st) for st in special_tokens]   ## 返回 "<\|endoftext\|>"
    split_pattern = "|".join(escaped_tokens) 
    documents = [part for part in re.split(split_pattern, text) if part]
    
    # 6. 并行预分词
    sequences = parallel_pre_tokenize(documents, num_processes, bytes_to_unicode_map)
    print(f"✅ 预分词完成，得到 {len(sequences):,} 个token序列")
    
    # 7. 初始化索引结构
    print("🔧 构建BPE索引...")
    bpe_index = BPEIndex(sequences)
    merges = []
    vocab_progress = len(vocab)
    total_merges = vocab_size - vocab_progress
    
    # 8. BPE训练主循环
    print(f"🔄 开始BPE训练，目标合并数: {total_merges:,}")
    progress_bar = tqdm(total=total_merges, desc="训练BPE", unit="合并", mininterval=0.5)
    
    while vocab_progress < vocab_size:
        best_pair = bpe_index.get_most_frequent()
        if best_pair is None:
            print("\n⚠️ 没有更多有效的字符对可供合并，提前结束训练")
            break
        
        # 创建新token
        new_token_str = best_pair[0] + best_pair[1]
        p1_bytes = unicode_to_bytes_map[best_pair[0]]
        p2_bytes = unicode_to_bytes_map[best_pair[1]]
        new_token_bytes = p1_bytes + p2_bytes
        
        # 执行合并
        merge_count = bpe_index.merge_pair(best_pair, new_token_str)
        if merge_count == 0:
            continue
        
        # 更新词汇表
        if new_token_bytes not in existing_bytes:
            vocab[next_token_id] = new_token_bytes
            existing_bytes.add(new_token_bytes)
            merges.append((p1_bytes, p2_bytes))
            next_token_id += 1
            vocab_progress += 1
            progress_bar.update(1)
        
        # 更新映射表
        unicode_to_bytes_map[new_token_str] = new_token_bytes
    
    progress_bar.close()
    return vocab, merges

def evaluate_tokenizer(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], test_text: str):
    """简单评估分词器效果"""
    print("\n🔍 分词器评估")
    sample_text = test_text[:200] + "..." if len(test_text) > 200 else test_text
    print(f"样例文本: {sample_text}")
    
    # 简单统计
    unique_tokens = set(vocab.values())
    print(f"词汇表大小: {len(vocab):,}")
    print(f"唯一token数: {len(unique_tokens):,}")
    print(f"合并操作数: {len(merges):,}")

if __name__ == "__main__":
    # 配置参数
    config = {
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>", "<pad>", "<unk>"],
        "num_processes": 8,
        "sample_size": 22000,  # 初始采样22,000文档
    }
    
    # 数据集路径
    train_path = "/home/mw/input/cs336_129682968/TinyStoriesV2-GPT4-train.txt"
    valid_path = "/home/mw/input/cs336_129682968/TinyStoriesV2-GPT4-valid.txt"
    
    # 检查文件是否存在
    if not Path(train_path).exists():
        raise FileNotFoundError(f"训练集文件 {train_path} 不存在")
    if not Path(valid_path).exists():
        raise FileNotFoundError(f"验证集文件 {valid_path} 不存在")
    
    # 训练模型
    print("🚀 开始训练")
    start_time = time.time()
    
    train_vocab, train_merges = run_train_bpe(train_path, **config)
    
    print(f"\n✅ 训练完成! 耗时: {time.time() - start_time:.2f}秒")
    
    # 小规模验证 (使用验证集的10%)
    print("\n🔬 小规模验证")
    valid_config = config.copy()
    valid_config["sample_size"] = int(2)  # 验证集使用500文档 (10%)
    
    valid_vocab, valid_merges = run_train_bpe(valid_path, **valid_config)
    
    # 分析结果
    print("\n📊 训练结果")
    print(f"训练词汇表大小: {len(train_vocab):,}")
    print(f"训练合并操作数: {len(train_merges):,}")
    print(f"验证词汇表大小: {len(valid_vocab):,}")
    print(f"验证合并操作数: {len(valid_merges):,}")
    
    # 比较词汇表重叠率
    train_tokens = set(train_vocab.values())
    valid_tokens = set(valid_vocab.values())
    overlap = train_tokens & valid_tokens
    print(f"\n📈 词汇表重叠率: {len(overlap)/len(train_tokens):.1%}")
    
    # 加载验证集样例进行评估
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_text = f.read(1000)  # 读取前1000字符用于评估
    evaluate_tokenizer(train_vocab, train_merges, valid_text)

    import json  # 需要导入json模块

    # 在main函数末尾添加以下代码（在内存分析之前）
    def save_vocab_and_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], vocab_path: str, merges_path: str):
        """保存词汇表和合并列表到文件"""
        # 1. 保存词汇表 (JSON格式)
        vocab_str = {idx: token.decode('utf-8', errors='replace') for idx, token in vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)
        
        # 2. 保存合并列表 (文本格式)
        with open(merges_path, 'w', encoding='utf-8') as f:
            for merge in merges:
                part1 = merge[0].decode('utf-8', errors='replace')
                part2 = merge[1].decode('utf-8', errors='replace')
                f.write(f"{part1} {part2}\n")

    # 在main函数中调用保存功能（在训练完成后）
    output_dir = "/home/mw/project"  # 修改为您的输出目录
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "gpt2_vocab.json")
    merges_path = os.path.join(output_dir, "gpt2_merges.txt")

    save_vocab_and_merges(train_vocab, train_merges, vocab_path, merges_path)
    print(f"✅ 词汇表已保存至: {vocab_path}")
    print(f"✅ 合并列表已保存至: {merges_path}")

    # 内存分析
    import psutil
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 3)  # GB
    print(f"💾 峰值内存使用: {mem_usage:.2f} GB")