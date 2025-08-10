# Essential Commands for LLaMA-LoRA Development

## 🚀 Core Development Workflow

### Environment Setup
```bash
# Navigate to project directory
cd /path/to/llama-lora

# Install dependencies with uv
uv sync

# Authenticate with Hugging Face (required for gated models)
huggingface-cli login
```

### Complete Training Pipeline
```bash
# 1. (Optional) Baseline evaluation
uv run python -m llama_lora.baseline

# 2. Train LoRA/DoRA adapter
uv run python -m llama_lora.train

# 3. Test fine-tuned model with inference
uv run python -m llama_lora.infer "Your test prompt here"

# 4. (Optional) Merge adapter into standalone model
uv run python -m llama_lora.merge
```

## ⚙️ Configuration & Validation

### Hydra Configuration System
```bash
# Validate configuration before training
uv run python -m llama_lora.validate

# Train with configuration overrides
uv run python -m llama_lora.train training.lr=1e-5 model.use_dora=true

# Use experiment presets
uv run python -m llama_lora.train +experiment=quick_test
uv run python -m llama_lora.train +experiment=full_training

# Override multiple parameters
uv run python -m llama_lora.train training.lr=2e-5 training.epochs=3 peft.r=16
```

### Advanced Inference Options
```bash
# Basic inference
uv run python -m llama_lora.infer "富士山の標高は？"

# Custom generation parameters
uv run python -m llama_lora.infer "Explain machine learning" \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.9
```

## 🧪 Testing & Quality Assurance

### Code Quality (Ruff)
```bash
# Check code quality
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Combined quality check
uv run ruff check --fix . && uv run ruff format .
```

### Testing Suite
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tokenizer_utils.py -v

# Run tests with coverage
pytest tests/ --cov=src/llama_lora

# Run single test function
pytest tests/test_config_validation.py::test_model_config_validation -v
```

## 📊 Monitoring & Debugging

### TensorBoard Monitoring
```bash
# Start TensorBoard (training logs)
uv run tensorboard --logdir outputs/runs

# View at http://localhost:6006
```

### Experiment Management
```bash
# Run multiple experiments
uv run python -m llama_lora.experiment

# Check output structure
tree outputs/
ls -la outputs/adapter/
ls -la outputs/merged/
```

## 🔧 Development & Debugging

### Package Management (uv)
```bash
# Update dependencies
uv sync

# Add new dependency
uv add package_name

# Remove dependency
uv remove package_name

# Update lock file
uv lock --upgrade
```

### Device & Environment Checks
```bash
# Check device availability
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
"

# Check memory usage (CUDA)
uv run python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB')
"
```

## 🍎 macOS-Specific Commands

### System Resource Monitoring
```bash
# Monitor system resources
top -o MEM
htop  # if installed

# Check disk space
df -h .
du -sh outputs/

# Open in Finder
open .
open outputs/
```

### File Operations
```bash
# Search for files
find . -name "*.py" -type f
find outputs/ -name "*adapter*" -type f

# Search in files
grep -r "DoRA" src/ --include="*.py"
grep -r "model_id" config/ --include="*.yaml"

# Count lines of code
find src/ -name "*.py" -exec wc -l {} + | tail -1
```

## 🚨 Troubleshooting Commands

### Memory Issues
```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Clear GPU memory (if CUDA)
uv run python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
"

# Check available memory
free -h  # Linux
vm_stat  # macOS
```

### Configuration Debugging
```bash
# Debug Hydra configuration
uv run python -m llama_lora.validate --cfg job

# Print effective configuration
uv run python -c "
import hydra
from omegaconf import OmegaConf
with hydra.initialize(config_path='config'):
    cfg = hydra.compose(config_name='config')
    print(OmegaConf.to_yaml(cfg))
"
```

### Process Management
```bash
# Kill stuck training processes
pkill -f "python.*llama_lora.train"
ps aux | grep python | grep llama_lora

# Monitor GPU usage (NVIDIA)
watch -n 1 nvidia-smi
```

## 📈 Quick Smoke Tests

### Fast End-to-End Test
```bash
# Complete pipeline with minimal resources
uv run python -m llama_lora.train +experiment=quick_test && \
uv run python -m llama_lora.infer "Test prompt" && \
uv run python -m llama_lora.merge

echo "✅ End-to-end pipeline completed successfully"
```

### Configuration Validation Test
```bash
# Test all experiment configurations
for config in config/experiment/*.yaml; do
    echo "Testing $(basename $config)"
    uv run python -m llama_lora.validate +experiment=$(basename $config .yaml)
done
```

## 🎯 Most Frequently Used Commands

### Daily Development
```bash
# Quality check before commit
uv run ruff check --fix . && uv run ruff format .

# Quick training iteration
uv run python -m llama_lora.train +experiment=quick_test

# Test inference
uv run python -m llama_lora.infer "Your prompt here"
```

### Before Production
```bash
# Full validation
uv run python -m llama_lora.validate

# Full test suite
pytest tests/ --cov=src/llama_lora

# Production training
uv run python -m llama_lora.train +experiment=full_training
```