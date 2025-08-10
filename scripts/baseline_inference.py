"""Baseline inference script for unmodified base models.

This script performs inference using the base model without any fine-tuning
or adapters. It provides a baseline for comparison with fine-tuned models
and demonstrates various prompt formats.
"""

import warnings
from typing import Any, Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

import config

warnings.filterwarnings("ignore", category=UserWarning)

def generate(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text from a prompt using the base model.
    
    Args:
        model: Pre-trained causal language model.
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Input text prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling parameter.
        do_sample: Whether to use sampling or greedy decoding.
        
    Returns:
        Generated text response.
    """
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
    """Generate a chat response from a list of messages.
    
    Args:
        model: Pre-trained causal language model.
        tokenizer: Tokenizer for encoding/decoding.
        messages: List of message dictionaries with 'role' and 'content' keys.
        max_new_tokens: Maximum number of tokens to generate.
        **kwargs: Additional arguments passed to generate function.
        
    Returns:
        Generated chat response.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model=model,
        tokenizer=tokenizer, 
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        **kwargs
    )


def _detect_device() -> str:
    """Detect the best available device for inference.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_inference_examples(
    model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer
) -> None:
    """Run various inference examples to demonstrate model capabilities.
    
    Args:
        model: Loaded base model.
        tokenizer: Loaded tokenizer.
    """
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
            "content": "あなたは有能な日本語アシスタントです。事実に基づき簡潔に答えます。"
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


def main() -> None:
    """Main function for baseline model inference.
    
    This function demonstrates baseline model performance across different
    prompt formats without any fine-tuning. Results can be compared with
    fine-tuned model outputs to evaluate training effectiveness.
    """
    device = _detect_device()
    print(f"Using device: {device}")

    print(f"Loading base model {config.MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("-" * 50)
    _run_inference_examples(model, tokenizer)


if __name__ == "__main__":
    main()
