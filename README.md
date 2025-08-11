# LoRA/DoRA Fine-Tuning for Llama Models

A modern, optimized project for fine-tuning Llama models using PEFT (LoRA/DoRA) with automatic device mapping, Flash Attention, and comprehensive error handling.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd llama-lora
    ```

2.  Install the dependencies (it is recommended to use a virtual environment):
    ```bash
    # Using uv
    uv sync
    
    # Or install with pip
    pip install -e .
    ```

## System Requirements

### Hardware Requirements
- **VRAM**: Minimum 8GB for 1B model with LoRA/DoRA (16GB+ recommended)
- **RAM**: Minimum 16GB system RAM (32GB+ recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3090/4090 or A100 recommended)
- **CPU**: Apple Silicon (M1/M2/M3) also supported with MPS

### VRAM Usage Guidelines
| Model Size | LoRA (fp16) | DoRA (fp16) | Full Fine-tune |
|------------|-------------|-------------|----------------|
| 1B params  | 4-6GB      | 6-8GB       | 12-16GB        |
| 3B params  | 8-12GB     | 12-16GB     | 24-32GB        |
| 7B params  | 16-24GB    | 24-32GB     | 48-64GB        |

*Note: Actual usage may vary based on sequence length, batch size, and optimization settings.*

## Usage

After installation, you can use the convenient command-line tools or run modules directly.

### 1. (Optional) Check Base Model Performance

Before fine-tuning, check the original model's performance. This uses the default configuration from `config/`.

```bash
# Using module
uv run python -m llama_lora.baseline

# Using command-line tool (after installation)
llama-lora-baseline
```

### 2. Run Fine-Tuning

Run the main training pipeline. This uses the Hydra configuration defined in the `config/` directory. You can override any setting from the command line.

```bash
# Using module - Run with default settings
uv run python -m llama_lora.train

# Using module - Override settings
uv run python -m llama_lora.train training.lr=1e-5 model.seq_len=2048

# Using command-line tool (after installation)
llama-lora-train
llama-lora-train training.lr=1e-5 model.seq_len=2048
```

### 3. Run Inference with Fine-Tuned Model

After training, test the fine-tuned model with a custom prompt.

```bash
uv run python -m llama_lora.infer "Your prompt here"
```

### 4. (Optional) Merge the Adapter

Create a standalone model by merging the adapter weights.

```bash
uv run python -m llama_lora.merge
```

### TensorBoard (Logging)

Training logs are recorded to `outputs/runs`.

View them with:

```bash
uv run tensorboard --logdir outputs/runs
```

## Important: Gated Model Access

This project uses `meta-llama/Llama-3.2-1B-Instruct`, which is a gated model on the Hugging Face Hub. To run the scripts, you must:

1.  Visit the [model's page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and request access.
2.  Log in to your Hugging Face account on your local machine:
    ```bash
    huggingface-cli login
    ```

## Quick Smoke Test

Use the small development preset to verify the end-to-end flow on CPU.

```bash
# 1) Train a tiny split (adapter + tokenizer saved under outputs/)
uv run python -m llama_lora.train +experiment=quick_test

# 2) Inference with the trained adapter
uv run python -m llama_lora.infer "日本語で自己紹介して"

# 3) Merge adapter into base to create a standalone model
uv run python -m llama_lora.merge

# The merged model is saved under `outputs/merged` and can be used
# as a standard Hugging Face model directory.
```

## Config Validation

Before running long jobs, validate your configuration:

```bash
uv run python -m llama_lora.validate
uv run python -m llama_lora.validate +experiment=full_training
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors
**Symptoms**: `CUDA out of memory` or similar GPU memory errors

**Solutions**:
```bash
# Reduce batch size
llama-lora-train training.batch_size=1

# Increase gradient accumulation to maintain effective batch size
llama-lora-train training.batch_size=1 training.gradient_accumulation_steps=16

# Use smaller sequence length
llama-lora-train model.seq_len=512

# Disable Flash Attention if causing issues
# (automatically handled by the code, but you can force CPU mode)
```

#### 2. Hugging Face Authentication Issues
**Symptoms**: `401 Client Error` or access denied to gated models

**Solutions**:
```bash
# Login to Hugging Face CLI
huggingface-cli login

# Or set token via environment variable
export HF_TOKEN="your_token_here"

# Request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

#### 3. Slow Training on CPU/MPS
**Symptoms**: Very slow training speeds on Apple Silicon or CPU

**Solutions**:
```bash
# Use smaller model for testing
llama-lora-train model.model_id="microsoft/DialoGPT-small"

# Reduce dataset size
llama-lora-train dataset.dataset_split="train[:0.1%]"

# Use development preset
llama-lora-train +experiment=quick_test
```

#### 4. Flash Attention Compatibility Issues
**Symptoms**: Import errors or CUDA version mismatches

**Solutions**:
- Install compatible Flash Attention version for your CUDA setup
- The code will automatically fall back to standard attention if Flash Attention fails
- Check CUDA version: `nvcc --version`

#### 5. Dataset Loading Errors
**Symptoms**: Dataset download failures or missing columns

**Solutions**:
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/datasets/

# Use different dataset split
llama-lora-train dataset.dataset_split="train[:1%]"

# Check internet connection and retry
```

### Performance Optimization Tips

1. **Use tensor-core optimized batch sizes** (multiples of 8)
2. **Enable dynamic padding** (automatically configured)
3. **Use mixed precision training** (fp16/bf16, automatically selected)
4. **Monitor GPU utilization** with `nvidia-smi` or `watch -n 1 nvidia-smi`
5. **Use gradient checkpointing** for memory efficiency (automatically enabled)

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Enable debug logging: add `--config-path=config --config-name=config` to see detailed logs
3. Run with minimal configuration first: `llama-lora-train +experiment=quick_test`

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tokenizer_utils.py -v

# Run tests with coverage
pytest tests/ --cov=src/llama_lora
```
