
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import warnings
import argparse

# Import configuration
import config

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    # --- Device Setup ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
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
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
    model = model.to(device).eval()

    # --- Tokenizer Loading ---
    # Load from the tokenizer directory if it exists, otherwise from the base model
    tokenizer_path = (
        config.TOKENIZER_DIR if os.path.isdir(config.TOKENIZER_DIR) else config.MODEL_ID
    )
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Inference ---
    print("-" * 50)
    print(f"Prompt:\n{prompt}\n")

    ids = tok(prompt.strip(), return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    response = tok.decode(out[0], skip_special_tokens=True)
    print("Response:")
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned PEFT model."
    )
    parser.add_argument(
        "prompt", type=str, help="The prompt to send to the model."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Max new tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Generation top_p."
    )
    args = parser.parse_args()

    main(args.prompt, args.max_new_tokens, args.temperature, args.top_p)
