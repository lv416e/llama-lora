"""Inference script for fine-tuned PEFT models.

This script loads a LoRA/DoRA adapter trained with the training pipeline and
performs text generation with customizable parameters. Supports automatic
device detection and flexible generation settings.
"""

import argparse
import os
import warnings
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

import config

warnings.filterwarnings("ignore", category=UserWarning)

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


def _load_tokenizer(tokenizer_dir: str, model_id: str) -> PreTrainedTokenizer:
    """Load tokenizer from custom directory or fallback to model ID.
    
    Args:
        tokenizer_dir: Path to saved tokenizer directory.
        model_id: Base model ID to use as fallback.
        
    Returns:
        Loaded tokenizer instance.
    """
    tokenizer_path = tokenizer_dir if os.path.isdir(tokenizer_dir) else model_id
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


def _validate_adapter_directory(adapter_dir: str) -> None:
    """Validate that the adapter directory exists.
    
    Args:
        adapter_dir: Path to the adapter directory.
        
    Raises:
        SystemExit: If adapter directory does not exist.
    """
    if not os.path.isdir(adapter_dir):
        print(f"Error: Adapter directory not found at '{adapter_dir}'")
        print("Please run train.py to create an adapter first.")
        exit(1)


def generate_response(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text response using the fine-tuned model.
    
    Args:
        model: Fine-tuned PEFT model.
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Input prompt text.
        device: Target device for computation.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling parameter.
        
    Returns:
        Generated response text.
    """
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
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> None:
    """Main inference function for PEFT model.
    
    Args:
        prompt: Input text prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling parameter.
    """
    device = _detect_device()
    print(f"Using device: {device}")

    _validate_adapter_directory(config.ADAPTER_DIR)

    print(f"Loading base model '{config.MODEL_ID}'...")
    base_model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID)

    print(f"Loading PEFT adapter from '{config.ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
    model = model.to(device).eval()

    tokenizer = _load_tokenizer(config.TOKENIZER_DIR, config.MODEL_ID)

    print("-" * 50)
    print(f"Prompt:\n{prompt}\n")

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    print("Response:")
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned PEFT model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Input text prompt for generation."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-2.0, higher = more random, default: 0.7)."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (0.0-1.0, default: 0.9)."
    )
    
    args = parser.parse_args()
    main(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
