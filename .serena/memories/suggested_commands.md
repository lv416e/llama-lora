# Essential Development Commands

## Environment Setup
```bash
# Install dependencies
uv sync

# Login to Hugging Face (required for gated models)
huggingface-cli login
```

## Code Quality
```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format

# Run tests
pytest tests/
pytest tests/ --cov=src/llama_lora  # with coverage
```

## Training & Inference
```bash
# Check baseline model performance
uv run python -m llama_lora.baseline

# Train with default configuration
uv run python -m llama_lora.train

# Train with custom parameters
uv run python -m llama_lora.train training.lr=1e-5 model.seq_len=2048

# Quick smoke test (CPU-friendly)
uv run python -m llama_lora.train +experiment=quick_test

# Validate configuration
uv run python -m llama_lora.validate

# Run inference with trained adapter
uv run python -m llama_lora.infer "Your prompt here"

# Merge adapter into standalone model
uv run python -m llama_lora.merge
```

## Monitoring
```bash
# View training logs with TensorBoard
uv run tensorboard --logdir outputs/runs
```

## Git Workflow
```bash
# Common git commands (Darwin system)
git status
git add .
git commit -m "feat: description"
git push
```

## System Utils (Darwin/macOS)
- `ls` - list files
- `cd` - change directory  
- `grep` - search text
- `find` - find files
- `open` - open files/directories in Finder