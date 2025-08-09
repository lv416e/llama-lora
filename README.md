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
    uv pip install .
    ```

## Usage

This project is organized into several scripts inside the `scripts/` directory.

### 1. (Optional) Check Base Model Performance

Before fine-tuning, you can check the performance of the original, pre-trained model:

```bash
python scripts/baseline_inference.py
```

### 2. Run Fine-Tuning

This script will fine-tune the model based on the settings in `scripts/config.py`. The resulting adapter will be saved to the output directory (default: `./out-llama-lora/`).

```bash
python scripts/train.py
```

### 3. Run Inference with Fine-Tuned Model

After training is complete, you can test the fine-tuned model with a custom prompt:

```bash
python scripts/infer.py "Your prompt here"
```

### 4. (Optional) Merge the Adapter

To create a standalone model, you can merge the adapter weights into the base model:

```bash
python scripts/merge.py
```

## Important: Gated Model Access

This project uses `meta-llama/Llama-3.2-1B-Instruct`, which is a gated model on the Hugging Face Hub. To run the scripts, you must:

1.  Visit the [model's page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and request access.
2.  Log in to your Hugging Face account on your local machine:
    ```bash
    huggingface-cli login
    ```
