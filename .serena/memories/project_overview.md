# LLaMA-LoRA Project Overview

## Project Purpose
A modern, production-ready framework for fine-tuning LLaMA models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation). The project emphasizes memory efficiency, automatic device optimization, and enterprise-grade error handling.

## Key Features
- **Advanced PEFT Support**: LoRA and DoRA fine-tuning with PEFT 0.17.0+
- **Multi-Device Optimization**: Automatic CUDA/MPS/CPU detection and optimization
- **Memory Efficiency**: Gradient checkpointing, dynamic padding, optimal batch sizing
- **Modern Configuration**: Hydra + Pydantic for type-safe, hierarchical configuration
- **Flash Attention**: Automatic FlashAttention2 with graceful fallback
- **Production Ready**: Comprehensive error handling, logging, and validation

## Target Model & Dataset
- **Primary Model**: meta-llama/Llama-3.2-1B-Instruct (gated model requiring HF authentication)
- **Dataset**: tatsu-lab/alpaca with configurable splits (default: 1% for quick testing)
- **Output Directory**: `./outputs/` (adapter, tokenizer, merged models, logs)

## Project Structure
```
llama-lora/
├── src/llama_lora/              # Main package
│   ├── train.py                 # Training pipeline with PEFT
│   ├── infer.py                 # Inference with adapter
│   ├── merge.py                 # Adapter merging to standalone model
│   ├── baseline.py              # Base model evaluation
│   ├── validate.py              # Configuration validation
│   ├── experiment.py            # Multi-experiment runner
│   └── utils/                   # Common utilities
│       ├── common.py            # Device, tokenizer, path management
│       └── exceptions.py        # Custom exception classes
├── config/                      # Hydra configuration
│   ├── config.yaml             # Base configuration
│   ├── schema.py               # Pydantic validation schemas
│   └── experiment/             # Experiment presets
├── tests/                       # Test suite
├── examples/                    # Jupyter notebooks
└── pyproject.toml              # Dependencies and entry points
```

## Core Workflows
1. **Training**: `uv run python -m llama_lora.train` → Creates LoRA/DoRA adapter
2. **Inference**: `uv run python -m llama_lora.infer "prompt"` → Uses base + adapter
3. **Merging**: `uv run python -m llama_lora.merge` → Creates standalone model
4. **Validation**: `uv run python -m llama_lora.validate` → Validates configuration

## Technical Highlights
- **Modern Python**: 3.12+ with uv package management
- **Type Safety**: Full Pydantic validation with Hydra integration
- **Memory Optimization**: Gradient checkpointing, mixed precision (device-dependent)
- **Device Flexibility**: Automatic CUDA/MPS/CPU detection and optimization
- **Professional Logging**: Structured logging with configurable backends (TensorBoard, Wandb)
- **Error Resilience**: Comprehensive exception handling with meaningful error messages