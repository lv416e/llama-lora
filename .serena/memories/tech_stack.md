# Technology Stack

## Language & Runtime
- **Python**: 3.12+ (enforced via .python-version)
- **Package Manager**: uv (high-performance Python package manager)

## Core Dependencies (pyproject.toml)
```toml
[project.dependencies]
accelerate>=1.10.0          # Distributed training & device optimization
datasets>=4.0.0             # Dataset loading and processing
hydra-core>=1.3.2          # Hierarchical configuration management
jupyter>=1.1.1             # Interactive development environment
peft>=0.17.0               # Parameter-Efficient Fine-Tuning (LoRA/DoRA)
pydantic>=2.11.7           # Data validation and settings management
pytest>=8.4.1             # Testing framework
ruff>=0.12.8               # Fast linting and formatting
tensorboard>=2.20.0        # Training visualization
torch>=2.8.0               # Deep learning framework
transformers>=4.55.0       # Pre-trained transformer models
wandb>=0.21.1              # Experiment tracking (optional)
```

## Machine Learning Stack
### Core ML Framework
- **PyTorch 2.8.0+**: Deep learning framework with automatic differentiation
- **Transformers 4.55.0**: Hugging Face models and tokenizers
- **PEFT 0.17.0**: Latest LoRA/DoRA implementations with advanced features
- **Accelerate 1.10.0**: Device management, distributed training, mixed precision

### Model & Data Pipeline
- **Datasets 4.0.0**: Efficient dataset loading and preprocessing
- **Flash Attention**: Automatic FlashAttention2 with eager attention fallback
- **Mixed Precision**: Automatic fp16/bf16 selection based on device capabilities

## Configuration & Validation
- **Hydra Core 1.3.2**: Hierarchical configuration with overrides and composition
- **Pydantic 2.11.7**: Runtime type checking and data validation
- **OmegaConf**: Dynamic configuration merging and YAML support

## Development & Quality Tools
- **Ruff 0.12.8**: Fast linting and formatting (replaces black, flake8, isort)
- **Pytest 8.4.1**: Testing framework with fixtures and parametrization
- **Jupyter 1.1.1**: Interactive development and experimentation

## Device & Performance Optimization
### Supported Devices
- **CUDA**: NVIDIA GPUs with automatic mixed precision (fp16/bf16)
- **MPS**: Apple Silicon (M1/M2/M3) with Metal Performance Shaders
- **CPU**: Fallback with automatic optimization

### Memory Optimization Features
- **Gradient Checkpointing**: Trades compute for memory
- **Dynamic Padding**: Minimizes wasted computation
- **Batch Size Adaptation**: Automatic device-specific optimization
- **Tensor Core Optimization**: Padding to multiples of 8 for efficiency

## PEFT Configuration Details
```python
# LoRA/DoRA Target Modules (LLaMA-specific)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "up_proj", "down_proj"      # MLP layers
]

# Optimization Settings
- Rank (r): 8-32 (configurable)
- Alpha: 16-64 (configurable scaling)
- Dropout: 0.1 (regularization)
- DoRA: Optional weight decomposition
```

## Logging & Monitoring
- **TensorBoard**: Default training visualization
- **Wandb**: Optional experiment tracking
- **Structured Logging**: Configurable levels with library suppression

## Entry Points & CLI
```toml
[project.scripts]
llama-lora-train = "llama_lora.train:main"
llama-lora-infer = "llama_lora.infer:main"
llama-lora-merge = "llama_lora.merge:main"
llama-lora-baseline = "llama_lora.baseline:main"
```