# Suggested Commands for Development

## Package Management (UV)
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name

# Run Python module
uv run python -m llama_lora.train

# Run Python script
uv run python script.py
```

## Training Commands
```bash
# Train with default configuration
uv run python -m llama_lora.train

# Train with custom parameters
uv run python -m llama_lora.train training.lr=1e-5 model.seq_len=2048

# Quick test (small dataset, 1 epoch)
uv run python -m llama_lora.train +experiment=quick_test

# Full training with DoRA
uv run python -m llama_lora.train +experiment=default-dora

# Validate configuration
uv run python -m llama_lora.validate
uv run python -m llama_lora.validate +experiment=full_training
```

## Inference Commands
```bash
# Run inference with latest model
uv run python -m llama_lora.infer "Your prompt here"

# Run inference with specific run
uv run python -m llama_lora.infer "Prompt" inference.run_id=run_20240101_120000

# Merge adapter into base model
uv run python -m llama_lora.merge

# Test baseline model
uv run python -m llama_lora.baseline
```

## Testing Commands
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config_validation.py -v

# Run tests with coverage
pytest tests/ --cov=src/llama_lora

# Run tests with output
pytest tests/ -v -s
```

## Code Quality Commands
```bash
# Format code with Ruff
ruff format .

# Check code style
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type checking (if mypy configured)
mypy src/
```

## Monitoring Commands
```bash
# Start TensorBoard
uv run tensorboard --logdir outputs/runs

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version

# Monitor system resources (macOS)
top -o cpu

# Monitor system resources (Linux)
htop
```

## Git Commands
```bash
# Check status
git status

# Stage changes
git add -A

# Commit with message
git commit -m "feat: add new feature"

# View recent commits
git log --oneline -10

# Create branch
git checkout -b feature/new-feature

# Push to remote
git push origin feature/new-feature
```

## Hugging Face Commands
```bash
# Login to Hugging Face
huggingface-cli login

# Set token via environment
export HF_TOKEN="your_token_here"

# Download model manually
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

# Clear HF cache
rm -rf ~/.cache/huggingface/
```

## System Commands (Darwin/macOS)
```bash
# List files with details
ls -la

# Navigate directories
cd path/to/directory

# Find files
find . -name "*.py" -type f

# Search in files (using ripgrep - faster than grep)
rg "search_term" --type py

# Search with grep
grep -r "search_term" . --include="*.py"

# Check disk usage
du -sh outputs/

# Check available disk space
df -h

# Process management
ps aux | grep python
kill -9 PID

# Environment variables
echo $PATH
export VAR_NAME="value"

# File permissions
chmod +x script.sh

# Archive/compress
tar -czf archive.tar.gz directory/
tar -xzf archive.tar.gz
```

## Quick Workflow Commands
```bash
# Complete training workflow
uv run python -m llama_lora.train +experiment=quick_test
uv run python -m llama_lora.infer "Test prompt"
uv run python -m llama_lora.merge

# Development cycle
ruff check --fix .
ruff format .
pytest tests/ -v
git add -A && git commit -m "fix: resolve issue"
```

## Troubleshooting Commands
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Reinstall dependencies
rm -rf .venv
uv sync

# Check Python version
python --version
uv python list

# Debug configuration
uv run python -m llama_lora.train --cfg job
uv run python -m llama_lora.train --help
```