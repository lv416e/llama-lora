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

from .utils.common import setup_logging
from .utils.storage import PathManager
from .utils.exceptions import ModelLoadingError, AdapterError

warnings.filterwarnings("ignore", category=UserWarning)

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
    """Load base model with error handling and Flash Attention fallback.

    Args:
        model_id: Model identifier.

    Returns:
        Loaded base model.

    Raises:
        ModelLoadingError: If model loading fails.
    """
    try:
        logger.info(f"Loading base model '{model_id}' on CPU for merging...")

        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except Exception as flash_error:
            logger.warning(
                f"FlashAttention2 not available ({str(flash_error)}), falling back to eager attention"
            )
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype="auto",
                attn_implementation="eager",
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
        device: Target device (should be 'cpu' for merging).

    Returns:
        PEFT model on specified device.

    Raises:
        AdapterError: If adapter loading fails.
    """
    try:
        logger.info(f"Loading PEFT adapter from '{adapter_dir}' on {device}...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

        if device != "cpu":
            logger.warning(f"Moving PEFT model from CPU to {device}")
            return peft_model.to(device)
        return peft_model
    except Exception as e:
        raise AdapterError(
            f"Failed to load PEFT adapter from '{adapter_dir}': {str(e)}"
        ) from e


def merge_and_save_model(
    peft_model: Any, tokenizer: AutoTokenizer, merged_dir: str
) -> Any:
    """Merge adapter weights and save model with tokenizer atomically."""
    from llama_lora.utils.storage import AtomicSaver

    try:
        logger.info("Merging adapter weights into the base model...")
        merged_model = peft_model.merge_and_unload()
        logger.info("Merge complete.")

        logger.info(
            f"Saving merged model and tokenizer atomically to '{merged_dir}'..."
        )

        with AtomicSaver(merged_dir).atomic_operation() as saver:
            saver.save_merged_artifacts(
                merged_model=merged_model, tokenizer=tokenizer, merged_dir=merged_dir
            )

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
    - Saves merge metadata
    """
    try:
        logger.info("Starting model merging process...")

        from config.schema import save_experiment_metadata
        from omegaconf import OmegaConf

        pydantic_cfg = cfg.to_pydantic_config()
        output_config = pydantic_cfg.output

        device = "cpu"
        logger.info(f"Using device: {device}")

        logger.info("Using structured output paths:")
        logger.info(f"  Adapter source: {output_config.adapter_dir}")
        logger.info(f"  Tokenizer source: {output_config.tokenizer_dir}")
        logger.info(f"  Merged output: {output_config.merged_dir}")

        model_id = cfg.model.model_id

        logger.info(
            f"Configuration: Model={model_id}, Adapter={output_config.adapter_dir}, Output={output_config.merged_dir}"
        )

        validate_adapter_directory(output_config.adapter_dir)
        base_model = load_base_model(model_id)

        peft_model = load_peft_adapter(base_model, output_config.adapter_dir, device)

        # Load tokenizer for atomic operation
        from transformers import AutoTokenizer

        tokenizer_path = (
            output_config.tokenizer_dir
            if PathManager.directory_exists(output_config.tokenizer_dir)
            else model_id
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        merge_and_save_model(peft_model, tokenizer, output_config.merged_dir)

        from datetime import datetime

        merge_metadata = {
            "operation": "model_merge",
            "timestamp": datetime.now().isoformat(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "source_adapter": output_config.adapter_dir,
            "source_tokenizer": output_config.tokenizer_dir,
            "output_merged": output_config.merged_dir,
            "base_model": model_id,
            "device_used": device,
        }

        metadata_file = save_experiment_metadata(merge_metadata, output_config)
        logger.info(f"Merge metadata saved to: {metadata_file}")

        logger.info(
            f"\nMerged model and tokenizer saved to: {output_config.merged_dir}"
        )
        logger.info("You can now use this directory as a standard Hugging Face model.")
        logger.info("Model merging completed successfully!")

    except Exception as e:
        logger.error(f"Model merging failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
