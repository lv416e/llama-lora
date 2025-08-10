"""Fine-tuning script for LLaMA models using LoRA/DoRA with PEFT.

This script provides a complete pipeline for fine-tuning LLaMA models using
Parameter-Efficient Fine-Tuning (PEFT) with LoRA or DoRA adapters. It includes
automatic device detection, mixed precision training, and memory optimization.
"""

import os
import random
import warnings
from typing import Any, Dict, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

import config

warnings.filterwarnings("ignore", category=UserWarning, message=".*tokenizers.*")


def _detect_device() -> str:
    """Detect the best available device for training.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _format_alpaca_dataset(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_length: int
) -> Dict[str, Any]:
    """Format dataset example into Alpaca instruction format.
    
    Args:
        example: Dataset example containing 'instruction', 'input', 'output' keys.
        tokenizer: Pre-trained tokenizer for encoding.
        max_length: Maximum sequence length.
        
    Returns:
        Dict containing tokenized inputs.
    """
    text = (
        f"### Instruction:\n{example['instruction']}\n"
        f"### Input:\n{example['input']}\n"
        f"### Response:\n{example['output']}"
    )
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )


def _get_precision_settings(device: str) -> tuple[bool, bool]:
    """Determine optimal precision settings for the given device.
    
    Args:
        device: Target device string.
        
    Returns:
        Tuple of (use_fp16, use_bf16) boolean flags.
    """
    if device != "cuda":
        return False, False
        
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    return use_fp16, use_bf16


def main() -> None:
    """Main training function for LoRA/DoRA fine-tuning.
    
    This function orchestrates the complete training pipeline including:
    - Device detection and setup
    - Dataset loading and preprocessing  
    - Model loading and PEFT configuration
    - Training with automatic mixed precision
    - Saving of trained artifacts
    """
    device = _detect_device()
    print(f"Using device: {device}")

    seed = getattr(config, "SEED", 42)
    _setup_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_dataset_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return _format_alpaca_dataset(example, tokenizer, config.SEQ_LEN)

    print(f"Loading dataset {config.DATASET_ID}...")
    dataset = load_dataset(config.DATASET_ID, split=config.DATASET_SPLIT)

    num_proc = max(1, (os.cpu_count() or 2) // 2)
    tokenized_dataset = dataset.map(
        format_dataset_fn,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    val_ratio = getattr(config, "VAL_RATIO", 0.1)
    split_dataset = tokenized_dataset.train_test_split(test_size=val_ratio, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Loading base model {config.MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID)

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print("Setting up PEFT with DoRA/LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.PEFT_R,
        lora_alpha=config.PEFT_LORA_ALPHA,
        lora_dropout=config.PEFT_LORA_DROPOUT,
        target_modules=config.PEFT_TARGET_MODULES,
        use_dora=config.USE_DORA,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    use_fp16, use_bf16 = _get_precision_settings(device)

    training_args = TrainingArguments(
        output_dir=config.BASE_OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH,
        gradient_accumulation_steps=config.ACCUM,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=False,
        logging_steps=10,
        logging_dir=getattr(config, "LOG_DIR", f"{config.BASE_OUTPUT_DIR}/runs"),
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=getattr(config, "EVAL_STEPS", 200),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        data_seed=seed,
        report_to=getattr(config, "REPORT_TO", "tensorboard"),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=getattr(config, "ES_PATIENCE", 3)
            )
        ],
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=True)
    print("Training finished.")

    print(f"Saving adapter to {config.ADAPTER_DIR}")
    model.save_pretrained(config.ADAPTER_DIR)

    print(f"Saving tokenizer to {config.TOKENIZER_DIR}")
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(config.TOKENIZER_DIR)

    print("\nAll done!")


if __name__ == "__main__":
    main()
