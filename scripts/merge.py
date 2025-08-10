"""Model merging script for PEFT adapters.

This script merges LoRA/DoRA adapter weights with the base model to create
a standalone model that can be used without PEFT. The merged model includes
both the base model parameters and the learned adapter weights.
"""

import os
import warnings

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

import config

warnings.filterwarnings("ignore", category=UserWarning)

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


def _load_and_merge_model(model_id: str, adapter_dir: str, device: str) -> AutoModelForCausalLM:
    """Load base model, attach PEFT adapter, and merge weights.
    
    Args:
        model_id: Base model identifier.
        adapter_dir: Path to the PEFT adapter.
        device: Target device for computation.
        
    Returns:
        Merged model with adapter weights integrated.
    """
    print(f"Loading base model '{model_id}'...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    print(f"Loading PEFT adapter from '{adapter_dir}'...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model = peft_model.to(device)

    print("Merging adapter weights into the base model...")
    merged_model = peft_model.merge_and_unload()
    print("Merge complete.")
    
    return merged_model


def _save_tokenizer(tokenizer_dir: str, model_id: str, output_dir: str) -> None:
    """Load and save tokenizer to the output directory.
    
    Args:
        tokenizer_dir: Path to saved tokenizer directory.
        model_id: Base model ID to use as fallback.
        output_dir: Directory to save the tokenizer.
    """
    tokenizer_path = tokenizer_dir if os.path.isdir(tokenizer_dir) else model_id
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    """Main function for merging PEFT adapter with base model.
    
    This function performs the complete merging workflow:
    - Validates adapter directory exists
    - Loads base model and PEFT adapter
    - Merges adapter weights into base model
    - Saves merged model and tokenizer
    
    The resulting merged model can be used as a standard HuggingFace model
    without requiring PEFT library.
    """
    device = "cpu"
    print(f"Using device: {device}")

    _validate_adapter_directory(config.ADAPTER_DIR)

    merged_model = _load_and_merge_model(
        model_id=config.MODEL_ID,
        adapter_dir=config.ADAPTER_DIR,
        device=device
    )

    os.makedirs(config.MERGED_DIR, exist_ok=True)

    print(f"Saving merged model to '{config.MERGED_DIR}'...")
    merged_model.save_pretrained(config.MERGED_DIR)

    _save_tokenizer(
        tokenizer_dir=config.TOKENIZER_DIR,
        model_id=config.MODEL_ID,
        output_dir=config.MERGED_DIR
    )

    print(f"\nMerged model and tokenizer saved to: {config.MERGED_DIR}")
    print("You can now use this directory as a standard HuggingFace model.")


if __name__ == "__main__":
    main()
