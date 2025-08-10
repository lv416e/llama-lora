"""Inference script for fine-tuned PEFT models using Hydra configuration.

This script loads a LoRA/DoRA adapter and performs text generation.
It uses Hydra to load model and path configurations while accepting
a prompt directly from the command line for interactive use, with
comprehensive error handling and structured logging.
"""

import argparse
import warnings
from typing import Any

import hydra
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .utils.common import DeviceManager, TokenizerUtils, PathManager, setup_logging
from .utils.exceptions import ModelLoadingError, AdapterError

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize logger
logger = setup_logging()


def load_tokenizer_with_fallback(
    tokenizer_path: str, model_id: str
) -> PreTrainedTokenizer:
    """Load tokenizer with fallback to model ID if path doesn't exist.

    Args:
        tokenizer_path: Path to fine-tuned tokenizer directory.
        model_id: Original model identifier for fallback.

    Returns:
        Configured tokenizer.

    Raises:
        ModelLoadingError: If tokenizer loading fails.
    """
    try:
        path = (
            tokenizer_path if PathManager.directory_exists(tokenizer_path) else model_id
        )
        logger.info(f"Loading tokenizer from '{path}'...")

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        return TokenizerUtils.setup_tokenizer(tokenizer)

    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load tokenizer from '{path}': {str(e)}"
        ) from e


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
        return AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load base model '{model_id}': {str(e)}"
        ) from e


def load_peft_model(base_model: Any, adapter_dir: str, device: str) -> Any:
    """Load PEFT adapter and set up for inference.

    Args:
        base_model: Base model to attach adapter to.
        adapter_dir: Path to adapter directory.
        device: Target device.

    Returns:
        PEFT model ready for inference.

    Raises:
        AdapterError: If adapter loading fails.
    """
    try:
        logger.info(f"Loading PEFT adapter from '{adapter_dir}'...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        return peft_model.to(device).eval()
    except Exception as e:
        raise AdapterError(
            f"Failed to load PEFT adapter from '{adapter_dir}': {str(e)}"
        ) from e


def generate_response(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate text response using the fine-tuned model.

    Args:
        model: Fine-tuned PEFT model.
        tokenizer: Configured tokenizer.
        prompt: Input text prompt.
        device: Target device.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling parameter (0.0-1.0).

    Returns:
        Generated text response.
    """
    try:
        logger.debug(f"Generating response for prompt (length: {len(prompt)} chars)")

        inputs = tokenizer(prompt.strip(), return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Generated response (length: {len(response)} chars)")
        return response

    except Exception as e:
        logger.error(f"Text generation failed: {str(e)}", exc_info=True)
        raise


def main(args: argparse.Namespace) -> None:
    """Main inference function using Hydra for configuration.

    Args:
        args: Command line arguments containing prompt and generation parameters.

    This function orchestrates the complete inference pipeline:
    - Loads Hydra configuration
    - Detects optimal device
    - Validates adapter directory
    - Loads model and tokenizer
    - Generates text response
    """
    try:
        logger.info("Starting inference process...")

        # Load Hydra config programmatically
        with hydra.initialize(version_base=None, config_path="../../config"):
            cfg = hydra.compose(config_name="config")

        # Detect device
        device = DeviceManager.detect_device()
        logger.info(f"Using device: {device}")

        # Validate adapter directory
        validate_adapter_directory(cfg.output.adapter_dir)

        # Load base model
        base_model = load_base_model(cfg.model.model_id)

        # Load PEFT model
        model = load_peft_model(base_model, cfg.output.adapter_dir, device)

        # Load tokenizer
        tokenizer = load_tokenizer_with_fallback(
            cfg.output.tokenizer_dir, cfg.model.model_id
        )

        # Log inference parameters
        logger.info(
            f"Generation parameters: max_tokens={args.max_new_tokens}, "
            f"temperature={args.temperature}, top_p={args.top_p}"
        )

        # Display prompt
        print("-" * 50)
        print(f"Prompt:\n{args.prompt}\n")
        logger.info(f"Processing prompt (length: {len(args.prompt)} chars)")

        # Generate response
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Display response
        print("Response:")
        print(response)
        print("-" * 50)

        logger.info("Inference completed successfully")

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned PEFT model using Hydra configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", type=str, help="Input text prompt for generation.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-2.0, higher = more random, default: 0.7).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (0.0-1.0, default: 0.9).",
    )

    cli_args = parser.parse_args()
    main(cli_args)
