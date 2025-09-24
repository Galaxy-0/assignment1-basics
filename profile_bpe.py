import cProfile
import pstats
from pathlib import Path
from cs336_basics.train_bpe import train_bpe

def run_train():
    test_file = Path("tests/fixtures/corpus.en")
    vocab, merges = train_bpe(
        input_path=test_file,
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )
    return vocab, merges

# Profile the function
profiler = cProfile.Profile()
profiler.enable()

vocab, merges = run_train()

profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')

print("="*60)
print("TOP 20 FUNCTIONS BY CUMULATIVE TIME:")
print("="*60)
stats.print_stats(20)

print("\n" + "="*60)
print("TOP 10 FUNCTIONS BY TIME SPENT IN FUNCTION:")
print("="*60)
stats.sort_stats('time')
stats.print_stats(10)

print(f"\nResults: {len(vocab)} vocab items, {len(merges)} merges")