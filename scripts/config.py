# --- Constants --- #

# Base model for fine-tuning
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Dataset settings
DATASET_ID = "tatsu-lab/alpaca"
DATASET_SPLIT = "train[:1%]"  # Use a small portion for quick testing

# Fine-tuning method
USE_DORA = True

# Training parameters
SEQ_LEN = 1024
LR = 2e-5
BATCH = 1
ACCUM = 8  # Gradient accumulation steps
EPOCHS = 1

# PEFT/LoRA configuration
PEFT_R = 16
PEFT_LORA_ALPHA = 32
PEFT_LORA_DROPOUT = 0.05
PEFT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Output directories
BASE_OUTPUT_DIR = "./out-llama-lora"
ADAPTER_DIR = f"{BASE_OUTPUT_DIR}/adapter"
MERGED_DIR = f"{BASE_OUTPUT_DIR}/merged"
TOKENIZER_DIR = f"{BASE_OUTPUT_DIR}/tokenizer"

# Logging/Reporting
REPORT_TO = "tensorboard"  # or "none", "wandb"
LOG_DIR = f"{BASE_OUTPUT_DIR}/runs"

# Reproducibility & training control (used if present)
SEED = 42
VAL_RATIO = 0.1
EVAL_STEPS = 200
ES_PATIENCE = 3
