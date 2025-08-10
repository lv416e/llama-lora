"""Baseline inference script using Hydra for configuration.

This script performs inference using the base model without any fine-tuning.
It uses Hydra to load the model configuration, providing a baseline for
comparison with fine-tuned models.
"""

import argparse
import warnings
from typing import Any, Dict, List

import hydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .utils.common import DeviceManager, TokenizerUtils, setup_logging
from .utils.exceptions import ModelLoadingError

warnings.filterwarnings("ignore", category=UserWarning)

logger = setup_logging()


def generate(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text from a prompt using the base model."""
    device = model.device
    inputs = tokenizer(prompt.strip(), return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_chat(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 128,
    **kwargs: Any,
) -> str:
    """Generate a chat response from a list of messages."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )


def _run_inference_examples(
    model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer
) -> None:
    """Run various inference examples to demonstrate model capabilities."""
    separator = "-" * 50
    max_tokens = 64

    simple_prompt = "What is the height of Mount Fuji? Please answer briefly."
    print(f"Simple Prompt:\n{simple_prompt}\n")
    print("Response:")
    print(generate(model, tokenizer, simple_prompt, max_new_tokens=max_tokens))
    print(separator)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer based on facts and be concise.",
        },
        {"role": "user", "content": "What is the height of Mount Fuji?"},
    ]

    chat_display = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    print(f"Chat Prompt:\n{chat_display}\n")
    print("Response:")
    print(generate_chat(model, tokenizer, messages, max_new_tokens=max_tokens))
    print(separator)

    alpaca_prompt = "### Instruction:\nPlease provide the height of Mount Fuji in a brief answer.\n### Input:\n\n### Response:\n"
    print(f"Alpaca-style Prompt:\n{alpaca_prompt}\n")
    print("Response:")
    print(generate(model, tokenizer, alpaca_prompt, max_new_tokens=max_tokens))
    print(separator)


def main(args: argparse.Namespace) -> None:
    """Main function for baseline model inference using Hydra config."""
    try:
        logger.info("Starting baseline inference process...")

        with hydra.initialize(version_base=None, config_path="../../config"):
            cfg = hydra.compose(config_name="config")

        device = DeviceManager.detect_device()
        logger.info(f"Using device: {device}")

        model_id = cfg.model.model_id
        logger.info(f"Loading base model '{model_id}' with auto optimization...")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            ).eval()
        except Exception as flash_error:
            logger.warning(
                f"FlashAttention2 not available ({str(flash_error)}), falling back to eager attention"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="eager",
                trust_remote_code=True,
            ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tokenizer = TokenizerUtils.setup_tokenizer(tokenizer)

        logger.info(
            f"Generation parameters: max_tokens={args.max_new_tokens}, "
            f"temperature={args.temperature}, top_p={args.top_p}"
        )

        print("-" * 50)
        print(f"Prompt:\n{args.prompt}\n")
        logger.info(f"Processing prompt (length: {len(args.prompt)} chars)")

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print("Response:")
        print(response)
        print("-" * 50)

        logger.info("Baseline inference completed successfully")

    except Exception as e:
        logger.error(f"Baseline inference failed: {str(e)}", exc_info=True)
        raise ModelLoadingError(f"Failed to run baseline inference: {str(e)}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline inference with the base model using Hydra configuration.",
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
