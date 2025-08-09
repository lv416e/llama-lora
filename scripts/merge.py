
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import warnings

# Import configuration
import config

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # --- Device Setup ---
    # Merging can be done on the CPU
    device = "cpu"
    print(f"Using device: {device}")

    # --- Model and Tokenizer Loading ---
    # Check if adapter directory exists
    if not os.path.isdir(config.ADAPTER_DIR):
        print(f"Error: Adapter directory not found at '{config.ADAPTER_DIR}'")
        print("Please run train.py to create an adapter first.")
        return

    print(f"Loading base model '{config.MODEL_ID}'...")
    base_model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID)

    print(f"Loading PEFT adapter from '{config.ADAPTER_DIR}'...")
    peft_model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
    peft_model = peft_model.to(device)

    # --- Merge and Unload ---
    print("Merging adapter weights into the base model...")
    merged_model = peft_model.merge_and_unload()
    print("Merge complete.")

    # --- Save Merged Model and Tokenizer ---
    os.makedirs(config.MERGED_DIR, exist_ok=True)

    print(f"Saving merged model to '{config.MERGED_DIR}'...")
    merged_model.save_pretrained(config.MERGED_DIR)

    # --- Tokenizer Loading and Saving ---
    tokenizer_path = (
        config.TOKENIZER_DIR if os.path.isdir(config.TOKENIZER_DIR) else config.MODEL_ID
    )
    print(
        f"Loading tokenizer from '{tokenizer_path}' to save alongside merged model..."
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tok.save_pretrained(config.MERGED_DIR)

    print(f"\nMerged model and tokenizer saved to: {config.MERGED_DIR}")
    print("You can now use this directory as a standard Hugging Face model.")


if __name__ == "__main__":
    main()
