# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running Tests
```bash
uv run pytest                    # Run all tests
uv run pytest -v               # Run with verbose output
uv run pytest tests/test_model.py  # Run specific test file
```

### Running Code
```bash
uv run <python_file_path>       # Run any Python file with automatic environment
```

### Creating Submission
```bash
./make_submission.sh            # Creates submission zip after running tests
```

## Architecture Overview

This is CS336 Spring 2025 Assignment 1 focused on implementing fundamental transformer components from scratch. The codebase is structured as a Python package using `uv` for dependency management.

### Core Structure

- **`cs336_basics/`**: Main package directory
  - `__init__.py`: Package initialization and version info
  - Contains core implementation files (to be implemented by students)

- **`tests/`**: Comprehensive test suite
  - `adapters.py`: Critical adapter functions that connect student implementations to tests
  - `test_*.py`: Individual test modules for different components
  - `_snapshots/`: NumPy snapshots for deterministic testing
  - `fixtures/`: Test data and reference files

### Key Implementation Pattern

The assignment uses an adapter pattern where student implementations are connected to tests through `tests/adapters.py`. Each `run_*` function in the adapters file:

1. Takes weights/parameters and input data
2. Should instantiate and run student's corresponding implementation  
3. Returns the output for comparison with reference snapshots

Example adapter functions to implement:
- `run_linear()` - Linear layer forward pass
- `run_embedding()` - Embedding lookup  
- `run_scaled_dot_product_attention()` - Attention mechanism
- `run_multihead_self_attention()` - Multi-head attention
- `run_transformer_block()` - Complete transformer block
- `run_transformer_lm()` - Full language model

### Component Hierarchy

The transformer components build hierarchically:
1. **Basic layers**: Linear, Embedding, RMSNorm, SiLU
2. **Attention components**: Scaled dot-product attention → Multi-head attention → RoPE integration
3. **Feed-forward**: SwiGLU activation function
4. **Full blocks**: Transformer block (attention + FFN with residuals and norms)
5. **Complete model**: Transformer language model

### Testing Framework

- Uses pytest with NumPy snapshot testing for deterministic results
- Tests compare student outputs with pre-computed reference snapshots
- All tests initially fail with `NotImplementedError` until implementations are connected via adapters
- Supports both individual component testing and full model evaluation

### Data and Tokenization

- Includes BPE tokenizer implementation and training
- Test fixtures include sample corpora and reference tokenizer files
- Supports TinyStories dataset for language modeling experiments

### Dependencies

Key dependencies managed through `uv`:
- PyTorch 2.6.0+ (2.2.2 for Intel Macs)
- `einops` and `einx` for tensor operations
- `jaxtyping` for type hints
- `tiktoken` for tokenization utilities