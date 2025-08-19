# Technology Stack & Build System

## Core Technologies

### Python Ecosystem
- **Python**: >=3.12 required
- **Package Manager**: `uv` (primary), pip (fallback)
- **Dependency Management**: `pyproject.toml` with locked dependencies via `uv.lock`

### ML/AI Stack
- **PyTorch**: >=2.8.0 with torchvision
- **Transformers**: >=4.55.0 (Hugging Face)
- **PEFT**: >=0.17.0 (Parameter-Efficient Fine-Tuning)
- **Accelerate**: >=1.10.0 (distributed training)
- **Datasets**: >=4.0.0 (Hugging Face datasets)

### Configuration & Validation
- **Hydra**: >=1.3.2 (configuration management)
- **Pydantic**: >=2.11.7 (data validation)
- **OmegaConf**: Configuration composition and overrides

### Development Tools
- **Testing**: pytest >=8.4.1
- **Linting**: ruff >=0.12.8
- **Logging**: TensorBoard >=2.20.0, WandB >=0.21.1

## Common Commands

### Environment Setup
```bash
# Install dependencies (recommended)
uv sync

# Alternative with pip
pip install -e .
```

### Training & Inference
```bash
# Basic training with default config
uv run python -m llama_lora.train

# Training with config overrides
uv run python -m llama_lora.train training.lr=1e-5 model.seq_len=2048

# Using experiment presets
uv run python -m llama_lora.train +experiment=quick_test

# Inference with trained model
uv run python -m llama_lora.infer "Your prompt here"

# Merge adapter into standalone model
uv run python -m llama_lora.merge
```

### CLI Tools (after installation)
```bash
llama-lora-train
llama-lora-infer
llama-lora-merge
llama-lora-baseline
```

### Development & Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/llama_lora

# Linting
ruff check src/ tests/
ruff format src/ tests/

# Config validation
uv run python -m llama_lora.validate
```

### Monitoring
```bash
# TensorBoard (logs in outputs/runs)
uv run tensorboard --logdir outputs/runs

# GPU monitoring
watch -n 1 nvidia-smi
```

## Hardware Optimization

### CUDA/GPU Settings
- **TF32**: Automatically enabled for Ampere GPUs (A40/A100)
- **Flash Attention**: SDPA implementation with automatic fallback
- **Mixed Precision**: FP16 preferred for Ampere, BF16 for newer architectures
- **Quantization**: 4-bit/8-bit QLoRA support via bitsandbytes

### Memory Management
- **Gradient Checkpointing**: Enabled by default
- **Dynamic Padding**: Optimized batch processing
- **Fused Optimizers**: AdamW fused for better performance