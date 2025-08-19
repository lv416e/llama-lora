# Product Overview

## LLaMA LoRA/DoRA Fine-Tuning Framework

A modern, optimized Python framework for fine-tuning LLaMA models using Parameter-Efficient Fine-Tuning (PEFT) techniques including LoRA (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation).

### Key Features

- **PEFT Support**: LoRA and DoRA adapters for memory-efficient fine-tuning
- **Hardware Optimization**: Automatic device detection (CUDA, MPS, CPU) with Flash Attention and mixed precision
- **Structured Experiments**: Hierarchical output organization with run tracking and metadata
- **Configuration Management**: Hydra-based configuration with Pydantic validation
- **Memory Efficiency**: Gradient checkpointing, quantization support (4-bit/8-bit QLoRA)
- **Production Ready**: Comprehensive error handling, logging, and testing

### Target Use Cases

- Fine-tuning LLaMA models on custom datasets
- Multilingual model adaptation (Japanese dataset support included)
- Research experiments with structured tracking
- Memory-constrained environments (1B-7B parameter models)

### Hardware Requirements

- **Minimum**: 8GB VRAM for 1B models with LoRA
- **Recommended**: 16GB+ VRAM, NVIDIA RTX 3090/4090 or A100
- **Apple Silicon**: M1/M2/M3 supported via MPS backend