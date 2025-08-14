# LLaMA-LoRA Fine-Tuning Project

## Project Purpose
This project provides a modern, optimized framework for fine-tuning Llama models using PEFT (Parameter-Efficient Fine-Tuning) techniques, specifically LoRA (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation). It features automatic device mapping, Flash Attention support, and comprehensive error handling.

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: UV (modern Python package management)
- **ML Framework**: PyTorch 2.8.0+
- **Fine-tuning**: PEFT 0.17.0+ (LoRA/DoRA)
- **Model Loading**: Transformers 4.55.0+ (Hugging Face)
- **Training Optimization**: Accelerate 1.10.0+
- **Configuration**: Hydra-core 1.3.2+ with Pydantic 2.11.7+ for validation
- **Data Processing**: Datasets 4.0.0+
- **Logging**: TensorBoard 2.20.0+, WandB 0.21.1+
- **Testing**: Pytest 8.4.1+
- **Code Quality**: Ruff 0.12.8+

## Key Features
- Parameter-efficient fine-tuning with LoRA/DoRA
- Automatic GPU/CPU/MPS device detection and optimization
- Flash Attention support with automatic fallback
- Comprehensive error handling and recovery
- Structured configuration using Hydra and Pydantic
- Experiment tracking and reproducibility
- TensorBoard integration for monitoring
- Support for gated models (Llama 3.2 family)

## Model Requirements
- Default: meta-llama/Llama-3.2-1B-Instruct (gated model requiring HF authentication)
- VRAM: 8GB minimum for 1B model with LoRA
- System RAM: 16GB minimum (32GB+ recommended)