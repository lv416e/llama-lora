
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Import configuration
import config

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
):
    """
    Generates text from a prompt using the given model and tokenizer.
    """
    device = model.device
    ids = tokenizer(prompt.strip(), return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_chat(model, tokenizer, messages, max_new_tokens=128, **kwargs):
    """
    Generates a chat response from a list of messages.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, **kwargs)


def main():
    # --- Device Setup ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # --- Model and Tokenizer Loading ---
    print(f"Loading base model {config.MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID).to(device).eval()
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Inference Examples ---
    print("-" * 50)

    # 1. Simple prompt
    prompt_simple = "日本語で簡潔に答えて。富士山の標高は？"
    print(f"Prompt:\n{prompt_simple}\n")
    print("Response:")
    print(generate(model, tok, prompt_simple, max_new_tokens=64))
    print("-" * 50)

    # 2. Chat prompt
    messages = [
        {"role": "system", "content": "あなたは有能な日本語アシスタントです。事実に基づき簡潔に答えます。"},
        {"role": "user", "content": "富士山の標高は？"},
    ]

    chat_prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    print(f"Chat Prompt:\n{chat_prompt_str}\n")
    print("Response:")
    print(generate_chat(model, tok, messages, max_new_tokens=64))
    print("-" * 50)

    # 3. Alpaca-style prompt
    prompt_alpaca = f"""
### Instruction:
富士山の標高を日本語で簡潔に答えてください。
### Input:

### Response:
"""
    print(f"Alpaca-style Prompt:\n{prompt_alpaca}\n")
    print("Response:")
    print(generate(model, tok, prompt_alpaca, max_new_tokens=64))
    print("-" * 50)


if __name__ == "__main__":
    main()
