
import os
import random
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
    EarlyStoppingCallback,
)

# Import configuration
import config

# Show precision warnings to surface MPS/CUDA dtype issues
warnings.filterwarnings("ignore", category=UserWarning, message=".*tokenizers.*")


def main():
    # --- Device Setup ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # --- Seeding for reproducibility ---
    seed = getattr(config, "SEED", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np  # optional

        np.random.seed(seed)
    except Exception:
        pass

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

    # Parallelize tokenization moderately
    num_proc = max(1, (os.cpu_count() or 2) // 2)
    tok_ds = dataset.map(
        format_dataset,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # --- Train/Validation split ---
    val_ratio = getattr(config, "VAL_RATIO", 0.1)
    split = tok_ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

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
    # Precision flags: enable fp16/bf16 only on CUDA; disable on MPS/CPU
    use_fp16 = device == "cuda"
    use_bf16 = False
    if device == "cuda":
        # Prefer bf16 if available (Ampere+), else fp16
        use_bf16 = torch.cuda.is_bf16_supported()
        if use_bf16:
            use_fp16 = False

    training_args = TrainingArguments(
        output_dir=config.BASE_OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH,
        gradient_accumulation_steps=config.ACCUM,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=False,
        logging_steps=10,
        logging_dir=getattr(config, "LOG_DIR", f"{config.BASE_OUTPUT_DIR}/runs"),
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        # Evaluation & early stopping
        evaluation_strategy="steps",
        eval_steps=getattr(config, "EVAL_STEPS", 200),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Reproducibility
        seed=seed,
        data_seed=seed,
        report_to=getattr(config, "REPORT_TO", "tensorboard"),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=getattr(config, "ES_PATIENCE", 3))],
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=True)
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
