# LLaMA-LoRA Project Overview

## Purpose
A modern, optimized project for fine-tuning Llama models using PEFT (Parameter Efficient Fine-Tuning) with LoRA/DoRA adapters. Features automatic device mapping, Flash Attention support, and comprehensive error handling.

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: uv (modern Python package management)
- **Configuration**: Hydra + Pydantic for structured configuration
- **ML Libraries**: 
  - transformers (Hugging Face)
  - peft (Parameter Efficient Fine-Tuning)
  - torch (PyTorch)
  - accelerate (distributed training)
  - datasets (Hugging Face datasets)
- **Logging**: TensorBoard, WandB support
- **Testing**: pytest
- **Linting/Formatting**: ruff

## Key Features
- LoRA and DoRA adapter fine-tuning
- Automatic device mapping (CUDA/MPS/CPU)
- Flash Attention optimization
- Structured configuration with validation
- Comprehensive error handling
- Multi-format model export

## Current Status
- Recently migrated from scripts/ to src/llama_lora/ module structure
- Implementing Pydantic + Hydra configuration system
- Default model: meta-llama/Llama-3.2-1B-Instruct (gated model)
- Default dataset: tatsu-lab/alpaca