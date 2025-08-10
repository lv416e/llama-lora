"""Improved fine-tuning script for LLaMA models using LoRA/DoRA with PEFT and Hydra configuration.

This script provides a complete pipeline for fine-tuning LLaMA models using
Parameter-Efficient Fine-Tuning (PEFT) with LoRA or DoRA adapters. It includes
automatic device detection, mixed precision training, memory optimization,
proper error handling, and structured logging.
"""

import warnings
from typing import Any, Dict

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
    PathManager,
    setup_logging,
    get_optimal_num_processes,
)
from .utils.exceptions import (
    ModelLoadingError,
    DatasetError,
    TrainingError,
    ConfigurationError,
)

# PydanticベースのHydra設定スキーマをConfigStoreに登録する

warnings.filterwarnings("ignore", category=UserWarning, message=".*tokenizers.*")

# Initialize logger
logger = setup_logging()


def validate_and_log_config(cfg: DictConfig) -> None:
    """Validate configuration values and log summary.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ConfigurationError: If configuration validation fails.
    """
    # Convert Hydra config to Pydantic for validation
    from config.schema import HydraConfig

    # Create HydraConfig instance from DictConfig
    hydra_config = HydraConfig(
        model=cfg.model,
        dataset=cfg.dataset,
        training=cfg.training,
        peft=cfg.peft,
        output=cfg.output,
        logging=cfg.logging,
    )

    # Convert to Pydantic for validation
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

    # Basic validation (Pydantic handles most validation automatically)
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
) -> tuple[Any, Any]:
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
            f"Loading base model '{cfg.model.model_id}' with auto optimization..."
        )

        # Latest best practices: auto device mapping, dtype, and Flash Attention
        # Try flash_attention_2 first, fall back to eager if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_id,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except ImportError:
            logger.warning("FlashAttention2 not available, falling back to eager attention")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_id,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="eager",
                trust_remote_code=True,
            )

        # Configure for training
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        logger.info("Setting up PEFT with LoRA/DoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.peft.r,
            lora_alpha=cfg.peft.lora_alpha,
            lora_dropout=cfg.peft.lora_dropout,
            target_modules=cfg.peft.target_modules,
            use_dora=cfg.model.use_dora,
        )

        model = get_peft_model(model, peft_config)
        logger.info("Trainable parameters:")
        model.print_trainable_parameters()

        return model

    except Exception as e:
        raise ModelLoadingError(f"Failed to load or setup model: {str(e)}") from e


def setup_training_arguments(cfg: DictConfig, device: str) -> TrainingArguments:
    """Setup training arguments with proper configuration.

    Args:
        cfg: Configuration object.
        device: Target device.

    Returns:
        Configured TrainingArguments.
    """
    use_fp16, use_bf16 = DeviceManager.setup_device_specific_settings(device)

    # Ensure output directory exists
    PathManager.ensure_directory(cfg.output.base_output_dir)

    return TrainingArguments(
        output_dir=cfg.output.base_output_dir,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.epochs,
        learning_rate=cfg.training.lr,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=False,
        logging_steps=10,
        logging_dir=cfg.output.log_dir,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg.training.seed,
        data_seed=cfg.training.seed,
        report_to=cfg.logging.report_to,
        optim="adamw_torch",
    )


def save_artifacts(cfg: DictConfig, model: Any, tokenizer: AutoTokenizer) -> None:
    """Save trained artifacts with error handling.

    Args:
        cfg: Configuration object.
        model: Trained model.
        tokenizer: Tokenizer.

    Raises:
        TrainingError: If saving fails.
    """
    try:
        logger.info(f"Saving adapter to {cfg.output.adapter_dir}")
        PathManager.ensure_directory(cfg.output.adapter_dir)
        model.save_pretrained(cfg.output.adapter_dir)

        logger.info(f"Saving tokenizer to {cfg.output.tokenizer_dir}")
        PathManager.ensure_directory(cfg.output.tokenizer_dir)
        tokenizer.save_pretrained(cfg.output.tokenizer_dir)

        logger.info("All artifacts saved successfully")

    except Exception as e:
        raise TrainingError(f"Failed to save artifacts: {str(e)}") from e


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for LoRA/DoRA fine-tuning.

    Args:
        cfg: Hydra configuration object containing all parameters.

    This function orchestrates the complete training pipeline including:
    - Configuration validation and logging
    - Device detection and setup
    - Dataset loading and preprocessing
    - Model loading and PEFT configuration
    - Training with automatic mixed precision
    - Saving of trained artifacts
    """
    try:
        # Validate and log configuration
        validate_and_log_config(cfg)

        # Setup device and seeds
        device = DeviceManager.detect_device()
        logger.info(f"Using device: {device}")
        SeedManager.setup_seed(cfg.training.seed)

        # Load and setup tokenizer
        tokenizer = load_and_setup_tokenizer(cfg.model.model_id)

        # Load and process dataset
        train_dataset, eval_dataset = load_and_process_dataset(cfg, tokenizer)

        # Setup data collator with dynamic padding for memory efficiency
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Optimize for tensor cores
            return_tensors="pt",
        )

        # Load and setup model
        model = load_and_setup_model(cfg)

        # Setup training arguments
        training_args = setup_training_arguments(cfg, device)

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.training.early_stopping_patience
                )
            ],
        )

        # Train
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished successfully")

        # Save artifacts
        save_artifacts(cfg, model, tokenizer)

        logger.info("Training pipeline completed successfully!")
        logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
