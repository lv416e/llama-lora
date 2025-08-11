"""Improved fine-tuning script for LLaMA models using LoRA/DoRA with PEFT and Hydra configuration.

This script provides a complete pipeline for fine-tuning LLaMA models using
Parameter-Efficient Fine-Tuning (PEFT) with LoRA or DoRA adapters. It includes
automatic device detection, mixed precision training, memory optimization,
proper error handling, and structured logging.
"""

import warnings
from typing import Any, Dict, Tuple

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .utils.common import (
    DeviceManager,
    SeedManager,
    TokenizerUtils,
    setup_logging,
    get_optimal_num_processes,
)
from .utils.storage import PathManager
from .utils.exceptions import (
    ModelLoadingError,
    DatasetError,
    TrainingError,
    ConfigurationError,
)

warnings.filterwarnings("ignore", category=UserWarning, message=".*tokenizers.*")

logger = setup_logging()


def validate_and_log_config(cfg: DictConfig) -> None:
    """Validate configuration values and log summary.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ConfigurationError: If configuration validation fails.
    """
    from config.schema import HydraConfig

    hydra_config = HydraConfig(
        model=cfg.model,
        dataset=cfg.dataset,
        training=cfg.training,
        peft=cfg.peft,
        output=cfg.output,
        logging=cfg.logging,
    )

    hydra_config.to_pydantic_config()

    logger.info("Configuration Summary:")
    logger.info("=" * 50)
    logger.info(f"Model: {cfg.model.model_id}")
    logger.info(f"Use DoRA: {cfg.model.use_dora}")
    logger.info(f"Dataset: {cfg.dataset.dataset_id} ({cfg.dataset.dataset_split})")
    logger.info(f"Learning Rate: {cfg.training.lr}")
    logger.info(f"Batch Size: {cfg.training.batch_size}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    logger.info(f"LoRA Rank: {cfg.peft.r}")
    logger.info(f"Output Dir: {cfg.output.base_output_dir}")
    logger.info("=" * 50)

    if cfg.training.lr <= 0:
        raise ConfigurationError(
            f"Learning rate must be positive, got {cfg.training.lr}"
        )

    if cfg.training.batch_size < 1:
        raise ConfigurationError(
            f"Batch size must be >= 1, got {cfg.training.batch_size}"
        )

    if cfg.peft.r < 1:
        raise ConfigurationError(f"LoRA rank must be >= 1, got {cfg.peft.r}")

    logger.info("Configuration validation successful!")


def load_and_setup_tokenizer(model_id: str) -> AutoTokenizer:
    """Load and configure tokenizer with error handling.

    Args:
        model_id: Model identifier.

    Returns:
        Configured tokenizer.

    Raises:
        ModelLoadingError: If tokenizer loading fails.
    """
    try:
        logger.info(f"Loading tokenizer from '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        return TokenizerUtils.setup_tokenizer(tokenizer)
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load tokenizer from {model_id}: {str(e)}"
        ) from e


def load_and_process_dataset(
    cfg: DictConfig, tokenizer: AutoTokenizer
) -> Tuple[Any, Any]:
    """Load and process dataset with error handling.

    Args:
        cfg: Configuration object.
        tokenizer: Configured tokenizer.

    Returns:
        Tuple of (train_dataset, eval_dataset).

    Raises:
        DatasetError: If dataset loading or processing fails.
    """
    try:
        logger.info(f"Loading dataset {cfg.dataset.dataset_id}...")
        dataset = load_dataset(cfg.dataset.dataset_id, split=cfg.dataset.dataset_split)

        def format_dataset_fn(example: Dict[str, Any]) -> Dict[str, Any]:
            return TokenizerUtils.format_alpaca_prompt(
                example, tokenizer, cfg.model.seq_len
            )

        logger.info("Processing dataset...")
        num_proc = get_optimal_num_processes()
        tokenized_dataset = dataset.map(
            format_dataset_fn,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )

        logger.info("Splitting dataset...")
        split_dataset = tokenized_dataset.train_test_split(
            test_size=cfg.dataset.val_ratio, seed=cfg.training.seed
        )

        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        logger.info(
            f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval samples"
        )
        return train_dataset, eval_dataset

    except Exception as e:
        raise DatasetError(f"Failed to load or process dataset: {str(e)}") from e


def load_and_setup_model(cfg: DictConfig) -> Any:
    """Load and configure model with PEFT setup using latest best practices.

    Args:
        cfg: Configuration object.

    Returns:
        Configured PEFT model.

    Raises:
        ModelLoadingError: If model loading fails.
    """
    try:
        logger.info(
            f"Loading base model '{cfg.model.model_id}' with ultimate optimization..."
        )

        # ① SDPA + TF32最適化を起動時強制（public docs準拠）
        import torch
        torch.set_float32_matmul_precision("high")  # TF32を許可
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32/CUDA optimizations enabled")

        # QLoRA対応：量子化設定
        quantization_config = None
        if hasattr(cfg, 'quantization'):
            try:
                from transformers import BitsAndBytesConfig
                
                if getattr(cfg.quantization, 'load_in_4bit', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=getattr(cfg.quantization, 'bnb_4bit_quant_type', 'nf4'),
                        bnb_4bit_compute_dtype=getattr(cfg.quantization, 'bnb_4bit_compute_dtype', 'bfloat16'),
                        bnb_4bit_use_double_quant=getattr(cfg.quantization, 'bnb_4bit_use_double_quant', True),
                    )
                    logger.info("4-bit QLoRA quantization enabled")
                elif getattr(cfg.quantization, 'load_in_8bit', False):
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("8-bit quantization enabled")
            except ImportError:
                logger.warning("bitsandbytes not available, skipping quantization")

        # モデル読み込み
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,  # A40でTensor Core最適化
            "attn_implementation": "sdpa",   # SDPA強制
            "trust_remote_code": True,
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_id,
            **model_kwargs
        )
        logger.info("SDPA attention implementation enabled with FP16")

        # QLoRA準備
        if quantization_config is not None:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")

        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        logger.info("Setting up LoRA/QLoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.peft.r,
            lora_alpha=cfg.peft.lora_alpha,
            lora_dropout=cfg.peft.lora_dropout,
            target_modules=list(cfg.peft.target_modules),
            use_dora=cfg.model.use_dora,
        )

        model = get_peft_model(model, peft_config)
        logger.info("Trainable parameters:")
        model.print_trainable_parameters()

        return model

    except Exception as e:
        raise ModelLoadingError(f"Failed to load or setup model: {str(e)}") from e Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
