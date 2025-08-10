# LoRA/DoRA Fine-Tuning for Llama Models

A simple project to fine-tune Llama models using PEFT (LoRA/DoRA) from the Hugging Face ecosystem.

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
    ```

## Usage

The project's core logic is located in the `src/llama_lora/` package. You can run the different workflows as modules using `uv run`.

### 1. (Optional) Check Base Model Performance

Before fine-tuning, check the original model's performance. This uses the default configuration from `config/`.

```bash
uv run python -m llama_lora.baseline
```

### 2. Run Fine-Tuning

Run the main training pipeline. This uses the Hydra configuration defined in the `config/` directory. You can override any setting from the command line.

```bash
# Run with default settings
uv run python -m llama_lora.train

# Override settings
uv run python -m llama_lora.train training.lr=1e-5 model.seq_len=2048
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

Training logs are recorded to `out-llama-lora/runs`.

View them with:

```bash
uv run tensorboard --logdir out-llama-lora/runs
```

## Important: Gated Model Access

This project uses `meta-llama/Llama-3.2-1B-Instruct`, which is a gated model on the Hugging Face Hub. To run the scripts, you must:

1.  Visit the [model's page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and request access.
2.  Log in to your Hugging Face account on your local machine:
    ```bash
    huggingface-cli login
    ```
