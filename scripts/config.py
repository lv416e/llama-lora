"""Configuration constants for LLaMA-LoRA fine-tuning pipeline.

This module contains all configuration parameters for model training, PEFT setup,
and output management. All constants are centralized here for easy modification
and consistency across the training workflow.
"""

from typing import List

MODEL_ID: str = "meta-llama/Llama-3.2-1B-Instruct"
"""Base model identifier for fine-tuning."""

DATASET_ID: str = "tatsu-lab/alpaca"
"""Dataset identifier from HuggingFace Hub."""

DATASET_SPLIT: str = "train[:1%]"
"""Dataset split specification for training."""

USE_DORA: bool = True
"""Enable Weight-Decomposed Low-Rank Adaptation (DoRA)."""

SEQ_LEN: int = 1024
"""Maximum sequence length for tokenization."""

LR: float = 2e-5
"""Learning rate for training."""

BATCH: int = 1
"""Per-device training batch size."""

ACCUM: int = 8
"""Gradient accumulation steps."""

EPOCHS: int = 1
"""Number of training epochs."""

PEFT_R: int = 16
"""LoRA rank parameter."""

PEFT_LORA_ALPHA: int = 32
"""LoRA alpha parameter for scaling."""

PEFT_LORA_DROPOUT: float = 0.05
"""Dropout rate for LoRA layers."""

PEFT_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj", 
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
"""Target modules for LoRA/DoRA application."""

BASE_OUTPUT_DIR: str = "./out-llama-lora"
"""Base output directory for all artifacts."""

ADAPTER_DIR: str = f"{BASE_OUTPUT_DIR}/adapter"
"""Directory for saving LoRA/DoRA adapter weights."""

MERGED_DIR: str = f"{BASE_OUTPUT_DIR}/merged"
"""Directory for saving merged model."""

TOKENIZER_DIR: str = f"{BASE_OUTPUT_DIR}/tokenizer"
"""Directory for saving tokenizer."""

REPORT_TO: str = "tensorboard"
"""Experiment tracking backend."""

LOG_DIR: str = f"{BASE_OUTPUT_DIR}/runs"
"""Directory for training logs."""

SEED: int = 42
"""Random seed for reproducibility."""

VAL_RATIO: float = 0.1
"""Validation set ratio."""

EVAL_STEPS: int = 200
"""Number of steps between evaluations."""

ES_PATIENCE: int = 3
"""Early stopping patience."""
