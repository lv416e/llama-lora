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

from .utils.common import DeviceManager, TokenizerUtils, setup_logging
from .utils.storage import PathManager
from .utils.exceptions import ModelLoadingError, AdapterError

warnings.filterwarnings("ignore", category=UserWarning)

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
        try:
            PathManager.validate_directory_exists(tokenizer_path)
            path = tokenizer_path
        except FileNotFoundError:
            path = model_id
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
    try:
        PathManager.validate_directory_exists(adapter_dir, "Adapter directory")
    except FileNotFoundError as e:
        error_msg = (
            f"Adapter directory not found at '{adapter_dir}'. "
            "Please run a training script to create an adapter first."
        )
        logger.error(error_msg)
        raise AdapterError(error_msg) from e


def load_base_model(model_id: str) -> Any:
    """Load base model with latest optimization techniques.

    Args:
        model_id: Model identifier.

    Returns:
        Loaded base model.

    Raises:
        ModelLoadingError: If model loading fails.
    """
    try:
        logger.info(f"Loading base model '{model_id}' with auto optimization...")

        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except ImportError:
            logger.warning(
                "FlashAttention2 not available, falling back to eager attention"
            )
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="eager",
                trust_remote_code=True,
            )
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
    """Main inference function with enhanced configuration support."""
    import time
    from llama_lora.utils.results import InferenceLogger
    from .utils.storage import InferencePathResolver

    try:
        logger.info("Starting inference process with enhanced configuration")

        # Build Hydra overrides from command-line arguments
        overrides = []

        # Handle backward compatibility for run_id and adapter_dir
        if args.run_id:
            overrides.append(f"inference.run_id={args.run_id}")
        if args.adapter_dir:
            overrides.append(f"inference.adapter_dir={args.adapter_dir}")
            overrides.append(f"inference.tokenizer_dir={args.adapter_dir}")

        # Override generation parameters from command line
        overrides.append(f"inference.max_new_tokens={args.max_new_tokens}")
        overrides.append(f"inference.temperature={args.temperature}")
        overrides.append(f"inference.top_p={args.top_p}")

        with hydra.initialize(version_base=None, config_path="../../config"):
            cfg = hydra.compose(config_name="config", overrides=overrides)

        from config.schema import HydraConfig

        # Convert to HydraConfig first, then to Pydantic
        hydra_config = HydraConfig(
            model=cfg.model,
            dataset=cfg.dataset,
            training=cfg.training,
            peft=cfg.peft,
            output=cfg.output,
            logging=cfg.logging,
            inference=getattr(cfg, "inference", None) or HydraConfig().inference,
        )
        pydantic_cfg = hydra_config.to_pydantic_config()
        output_config = pydantic_cfg.output
        inference_config = pydantic_cfg.inference

        # Use new path resolution system
        resolved_paths = InferencePathResolver.resolve_paths(
            output_config, inference_config
        )

        # Update output_config with resolved paths
        output_config.adapter_dir = resolved_paths["adapter_dir"]
        output_config.tokenizer_dir = resolved_paths["tokenizer_dir"]

        # Set correct run_id and metadata_dir for inference logging
        if "discovered_run_id" in resolved_paths:
            discovered_run_id = resolved_paths["discovered_run_id"]
            output_config.run_id = discovered_run_id
            # Set correct metadata_dir based on discovered run_id
            experiment_base = f"{output_config.base_output_dir}/experiments/{output_config.experiment_name}"
            run_base = f"{experiment_base}/runs/{discovered_run_id}"
            output_config.metadata_dir = f"{run_base}/metadata"
            logger.info(f"Auto-discovered latest run: {discovered_run_id}")
        elif "used_fallback" in resolved_paths:
            logger.info("Used fallback run_id configuration")
        elif inference_config.run_id:
            # If specific run_id was provided via inference config
            output_config.run_id = inference_config.run_id
            experiment_base = f"{output_config.base_output_dir}/experiments/{output_config.experiment_name}"
            run_base = f"{experiment_base}/runs/{inference_config.run_id}"
            output_config.metadata_dir = f"{run_base}/metadata"

        inference_logger = InferenceLogger(output_config, "fine_tuned")

        device = DeviceManager.detect_device()
        logger.info(f"Using device: {device}")

        logger.info("Using resolved paths:")
        logger.info(f"  Adapter: {output_config.adapter_dir}")
        logger.info(f"  Tokenizer: {output_config.tokenizer_dir}")

        validate_adapter_directory(output_config.adapter_dir)
        base_model = load_base_model(cfg.model.model_id)

        model = load_peft_model(base_model, output_config.adapter_dir, device)
        tokenizer = load_tokenizer_with_fallback(
            output_config.tokenizer_dir, cfg.model.model_id
        )

        # Use generation parameters from inference config
        generation_params = {
            "max_new_tokens": inference_config.max_new_tokens,
            "temperature": inference_config.temperature,
            "top_p": inference_config.top_p,
        }

        model_config = {
            "model_id": cfg.model.model_id,
            "adapter_path": output_config.adapter_dir,
            "tokenizer_path": output_config.tokenizer_dir,
            "device": str(device),
        }

        logger.info(
            f"Generation parameters: max_tokens={inference_config.max_new_tokens}, "
            f"temperature={inference_config.temperature}, top_p={inference_config.top_p}"
        )

        print("-" * 50)
        print(f"Prompt:\n{args.prompt}\n")
        logger.info(f"Processing prompt (length: {len(args.prompt)} chars)")

        start_time = time.time()
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            **generation_params,
        )
        execution_time = time.time() - start_time

        print("Response:")
        print(response)
        print("-" * 50)

        inference_logger.log_inference(
            prompt=args.prompt,
            response=response,
            model_config=model_config,
            generation_params=generation_params,
            execution_time=execution_time,
            model_type="fine_tuned",
        )

        session_file = inference_logger.save_session()
        comparison_file = inference_logger.export_comparison_data()

        # Log session information with new session-based structure
        session_info = inference_logger.get_session_info()

        logger.info("Session completed successfully")
        logger.info(f"  Session ID: {session_info['session_id']}")
        logger.info(f"  Session directory: {session_info['session_dir']}")
        logger.info(f"  Total inferences: {session_info['inference_count']}")
        logger.info(f"  Session summary: {session_file}")
        logger.info(f"  Legacy comparison: {comparison_file}")
        logger.info(
            "Inference session saved with session accumulation (MLOps industry standard)"
        )

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Run inference with a fine-tuned PEFT model using Hydra configuration.

Enhanced Usage Patterns:
  # Auto-discover latest run (default behavior):
  python -m llama_lora.infer "your prompt"
  
  # Use specific inference config:
  python -m llama_lora.infer "your prompt" inference=manual
  
  # Use experiment configuration:
  python -m llama_lora.infer "your prompt" +experiment=my_inference
  
  # Override specific settings:
  python -m llama_lora.infer "your prompt" inference.run_id=20250811_072332
  
  # Backward compatibility (legacy args still work):
  python -m llama_lora.infer "your prompt" --run_id 20250811_072332 --max_new_tokens 256
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", type=str, help="Input text prompt for generation.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128). "
        "Overrides inference config if specified.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-2.0, higher = more random, default: 0.7). "
        "Overrides inference config if specified.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (0.0-1.0, default: 0.9). "
        "Overrides inference config if specified.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="[LEGACY] Specific run ID to use for inference. "
        "Prefer using 'inference.run_id=VALUE' in Hydra overrides.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="",
        help="[LEGACY] Direct path to adapter directory. "
        "Prefer using 'inference.adapter_dir=PATH' in Hydra overrides.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
