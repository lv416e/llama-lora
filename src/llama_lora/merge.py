"""Model merging script using Hydra for configuration.

This script merges LoRA/DoRA adapter weights with the base model to create
a standalone model. It uses Hydra to manage all model and path configurations,
with comprehensive error handling and structured logging.
"""

import warnings
from typing import Any

import hydra
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.common import PathManager, setup_logging
from .utils.exceptions import ModelLoadingError, AdapterError

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize logger
logger = setup_logging()


def validate_adapter_directory(adapter_dir: str) -> None:
    """Validate that the adapter directory exists.

    Args:
        adapter_dir: Path to adapter directory.

    Raises:
        AdapterError: If adapter directory does not exist.
    """
    if not PathManager.directory_exists(adapter_dir):
        error_msg = (
            f"Adapter directory not found at '{adapter_dir}'. "
            "Please run a training script to create an adapter first."
        )
        logger.error(error_msg)
        raise AdapterError(error_msg)


def load_base_model(model_id: str) -> Any:
    """Load base model with error handling.

    Args:
        model_id: Model identifier.

    Returns:
        Loaded base model.

    Raises:
        ModelLoadingError: If model loading fails.
    """
    try:
        logger.info(f"Loading base model '{model_id}'...")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load base model '{model_id}': {str(e)}"
        ) from e


def load_peft_adapter(base_model: Any, adapter_dir: str, device: str) -> Any:
    """Load PEFT adapter and merge with base model.

    Args:
        base_model: Base model to attach adapter to.
        adapter_dir: Path to adapter directory.
        device: Target device.

    Returns:
        PEFT model on specified device.

    Raises:
        AdapterError: If adapter loading fails.
    """
    try:
        logger.info(f"Loading PEFT adapter from '{adapter_dir}'...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        return peft_model.to(device)
    except Exception as e:
        raise AdapterError(
            f"Failed to load PEFT adapter from '{adapter_dir}': {str(e)}"
        ) from e


def merge_and_save_model(peft_model: Any, merged_dir: str) -> Any:
    """Merge adapter weights and save the model.

    Args:
        peft_model: PEFT model with adapter.
        merged_dir: Directory to save merged model.

    Returns:
        Merged model.

    Raises:
        ModelLoadingError: If merging or saving fails.
    """
    try:
        logger.info("Merging adapter weights into the base model...")
        merged_model = peft_model.merge_and_unload()
        logger.info("Merge complete.")

        # Ensure output directory exists
        PathManager.ensure_directory(merged_dir)

        logger.info(f"Saving merged model to '{merged_dir}'...")
        merged_model.save_pretrained(merged_dir)

        return merged_model

    except Exception as e:
        raise ModelLoadingError(f"Failed to merge or save model: {str(e)}") from e


def save_tokenizer(tokenizer_dir: str, model_id: str, merged_dir: str) -> None:
    """Load and save tokenizer to merged directory.

    Args:
        tokenizer_dir: Directory containing fine-tuned tokenizer.
        model_id: Original model identifier for fallback.
        merged_dir: Directory to save tokenizer to.

    Raises:
        ModelLoadingError: If tokenizer operations fail.
    """
    try:
        # Use fine-tuned tokenizer if available, otherwise fall back to base model
        tokenizer_path = (
            tokenizer_dir if PathManager.directory_exists(tokenizer_dir) else model_id
        )
        logger.info(f"Loading tokenizer from '{tokenizer_path}'...")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        tokenizer.save_pretrained(merged_dir)

        logger.info(f"Tokenizer saved to '{merged_dir}'")

    except Exception as e:
        raise ModelLoadingError(f"Failed to load or save tokenizer: {str(e)}") from e


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for merging PEFT adapter with base model.

    Args:
        cfg: Hydra configuration containing model and output settings.

    This function orchestrates the complete model merging pipeline:
    - Validates adapter directory existence
    - Loads base model and PEFT adapter
    - Merges adapter weights into base model
    - Saves merged model and tokenizer
    """
    try:
        logger.info("Starting model merging process...")

        # Use CPU for merging (less memory intensive)
        device = "cpu"
        logger.info(f"Using device: {device}")

        # Extract configuration
        adapter_dir = cfg.output.adapter_dir
        model_id = cfg.model.model_id
        merged_dir = cfg.output.merged_dir
        tokenizer_dir = cfg.output.tokenizer_dir

        logger.info(
            f"Configuration: Model={model_id}, Adapter={adapter_dir}, Output={merged_dir}"
        )

        # Validate adapter directory exists
        validate_adapter_directory(adapter_dir)

        # Load base model
        base_model = load_base_model(model_id)

        # Load PEFT adapter
        peft_model = load_peft_adapter(base_model, adapter_dir, device)

        # Merge and save model
        merge_and_save_model(peft_model, merged_dir)

        # Save tokenizer
        save_tokenizer(tokenizer_dir, model_id, merged_dir)

        logger.info(f"\nMerged model and tokenizer saved to: {merged_dir}")
        logger.info("You can now use this directory as a standard Hugging Face model.")
        logger.info("Model merging completed successfully!")

    except Exception as e:
        logger.error(f"Model merging failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
