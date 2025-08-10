"""Baseline inference script using Hydra for configuration.

This script performs inference using the base model without any fine-tuning.
It uses Hydra to load the model configuration, providing a baseline for
comparison with fine-tuned models.
"""

import warnings
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .utils.common import DeviceManager, TokenizerUtils, setup_logging
from .utils.exceptions import ModelLoadingError

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize logger
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

    # Simple prompt example
    simple_prompt = "日本語で簡潔に答えて。富士山の標高は？"
    print(f"Simple Prompt:\n{simple_prompt}\n")
    print("Response:")
    print(generate(model, tokenizer, simple_prompt, max_new_tokens=max_tokens))
    print(separator)

    # Chat-style prompt example
    messages = [
        {
            "role": "system",
            "content": "あなたは有能な日本語アシスタントです。事実に基づき簡潔に答えます。",
        },
        {"role": "user", "content": "富士山の標高は？"},
    ]

    chat_display = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    print(f"Chat Prompt:\n{chat_display}\n")
    print("Response:")
    print(generate_chat(model, tokenizer, messages, max_new_tokens=max_tokens))
    print(separator)

    # Alpaca-style prompt example
    alpaca_prompt = (
        "### Instruction:\n"
        "富士山の標高を日本語で簡潔に答えてください。\n"
        "### Input:\n\n"
        "### Response:\n"
    )
    print(f"Alpaca-style Prompt:\n{alpaca_prompt}\n")
    print("Response:")
    print(generate(model, tokenizer, alpaca_prompt, max_new_tokens=max_tokens))
    print(separator)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for baseline model inference using Hydra config."""
    try:
        device = DeviceManager.detect_device()
        logger.info(f"Using device: {device}")

        model_id = cfg.model.model_id
        logger.info(f"Loading base model {model_id}...")

        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tokenizer = TokenizerUtils.setup_tokenizer(tokenizer)

        logger.info("Running inference examples")
        _run_inference_examples(model, tokenizer)

    except Exception as e:
        logger.error(f"Baseline inference failed: {str(e)}", exc_info=True)
        raise ModelLoadingError(f"Failed to run baseline inference: {str(e)}") from e


if __name__ == "__main__":
    main()
