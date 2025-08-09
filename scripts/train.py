
import os
import warnings
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Import configuration
import config

# Suppress warnings
warnings.filterwarnings("ignore", message=".*fp16.*")
warnings.filterwarnings("ignore", message=".*bf16.*")
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    # --- Device Setup ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # --- Tokenizer Setup ---
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Data Preparation ---
    def format_dataset(e):
        s = f"### Instruction:\n{e['instruction']}\n### Input:\n{e['input']}\n### Response:\n{e['output']}"
        # Tokenize and pad/truncate to sequence length
        return tok(s, truncation=True, max_length=config.SEQ_LEN, padding="max_length")

    print(f"Loading dataset {config.DATASET_ID}...")
    dataset = load_dataset(config.DATASET_ID, split=config.DATASET_SPLIT)
    tok_ds = dataset.map(format_dataset, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # --- Model Loading and PEFT Setup ---
    print(f"Loading base model {config.MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID)

    # Enable gradient checkpointing for memory efficiency
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print("Setting up PEFT with DoRA/LoRA...")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.PEFT_R,
        lora_alpha=config.PEFT_LORA_ALPHA,
        lora_dropout=config.PEFT_LORA_DROPOUT,
        target_modules=config.PEFT_TARGET_MODULES,
        use_dora=config.USE_DORA,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # --- Training ---
    use_fp16 = device == "cuda"

    training_args = TrainingArguments(
        output_dir=config.BASE_OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH,
        gradient_accumulation_steps=config.ACCUM,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        fp16=use_fp16,  # Enable for CUDA
        bf16=use_fp16,  # Enable for CUDA
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- Save Artifacts ---
    print(f"Saving adapter to {config.ADAPTER_DIR}")
    model.save_pretrained(config.ADAPTER_DIR)

    print(f"Saving tokenizer to {config.TOKENIZER_DIR}")
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    tok.save_pretrained(config.TOKENIZER_DIR)

    print("\nAll done!")


if __name__ == "__main__":
    main()
