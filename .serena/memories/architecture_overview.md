# LLaMA-LoRA Architecture Overview

## 🏗️ System Architecture

### High-Level Component Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    LLaMA-LoRA Framework                     │
├─────────────────────────────────────────────────────────────┤
│  Configuration Layer (Hydra + Pydantic)                    │
│  ├── config/config.yaml (base configuration)               │
│  ├── config/schema.py (validation schemas)                 │
│  └── config/experiment/ (experiment presets)               │
├─────────────────────────────────────────────────────────────┤
│  Core Pipeline Modules                                      │
│  ├── train.py (PEFT training pipeline)                     │
│  ├── infer.py (inference with adapters)                    │
│  ├── merge.py (adapter integration)                        │
│  ├── baseline.py (base model evaluation)                   │
│  ├── validate.py (configuration validation)                │
│  └── experiment.py (multi-experiment runner)               │
├─────────────────────────────────────────────────────────────┤
│  Utility Layer                                             │
│  ├── utils/common.py (device, tokenizer, path management)  │
│  └── utils/exceptions.py (custom error handling)           │
├─────────────────────────────────────────────────────────────┤
│  External Dependencies                                      │
│  ├── PyTorch (core ML framework)                           │
│  ├── Transformers (model loading & tokenization)          │
│  ├── PEFT (LoRA/DoRA implementation)                       │
│  └── Accelerate (device optimization)                      │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components Deep Dive

### 1. Configuration Management System
**Hydra + Pydantic Dual Architecture**
```python
# Hydra: Runtime configuration management
@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Runtime configuration with overrides

# Pydantic: Type validation and safety
class TrainingConfig(BaseModel):
    lr: float = Field(gt=0.0, description="Learning rate")
    batch_size: int = Field(ge=1, description="Batch size")
```

**Configuration Flow:**
1. Base YAML config loaded by Hydra
2. CLI overrides applied (`training.lr=1e-5`)
3. Converted to Pydantic for validation
4. Used throughout pipeline with type safety

### 2. Training Pipeline (train.py)
**Modular Training Architecture**
```python
def main(cfg: DictConfig) -> None:
    validate_and_log_config(cfg)           # Pydantic validation
    device = DeviceManager.detect_device() # Auto device detection
    tokenizer = load_and_setup_tokenizer()  # HF tokenizer setup
    datasets = load_and_process_dataset()   # Alpaca formatting
    model = load_and_setup_model()          # PEFT model creation
    trainer = Trainer(...)                  # HF Trainer integration
    trainer.train()                         # Training execution
    save_artifacts()                        # Adapter persistence
```

**Key Features:**
- **Auto Device Detection**: CUDA → MPS → CPU fallback
- **Memory Optimization**: Gradient checkpointing, mixed precision
- **Flash Attention**: Automatic FlashAttention2 with eager fallback
- **PEFT Integration**: LoRA/DoRA configuration with target module selection

### 3. Inference System (infer.py)
**Adapter Composition Architecture**
```python
# Two-stage model loading
base_model = AutoModelForCausalLM.from_pretrained(model_id)
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

# Dynamic inference with parameters
response = generate_response(
    model=peft_model,
    prompt=args.prompt,
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature
)
```

**Inference Flow:**
1. Load base model with optimization flags
2. Load and attach PEFT adapter
3. Configure generation parameters
4. Execute inference with error handling

### 4. Model Merging System (merge.py)
**Adapter Integration Architecture**
```python
# Physical model integration
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = peft_model.merge_and_unload()  # Adapter → base weights
merged_model.save_pretrained(merged_dir)      # Standalone model
```

**Merge Benefits:**
- **Deployment Simplicity**: Single model directory
- **Performance**: No adapter overhead during inference
- **Compatibility**: Standard HuggingFace model format

## 🎛️ PEFT (LoRA/DoRA) Implementation

### LoRA Configuration Architecture
```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=cfg.peft.r,                    # Rank (8-32 typical)
    lora_alpha=cfg.peft.lora_alpha,  # Scaling factor
    lora_dropout=cfg.peft.lora_dropout,
    target_modules=[               # LLaMA-specific targets
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # MLP
    ],
    use_dora=cfg.model.use_dora,   # DoRA weight decomposition
)
```

### Target Module Strategy for LLaMA
```
LLaMA-3.2 Architecture Targeting:
├── Attention Layers (4 modules)
│   ├── q_proj: Query projection
│   ├── k_proj: Key projection  
│   ├── v_proj: Value projection
│   └── o_proj: Output projection
└── MLP Layers (3 modules)
    ├── gate_proj: SwiGLU gate mechanism
    ├── up_proj: Up-sampling projection
    └── down_proj: Down-sampling projection
```

## 🖥️ Device Optimization Architecture

### Multi-Device Support System
```python
class DeviceManager:
    @staticmethod
    def detect_device() -> str:
        if torch.cuda.is_available():    return "cuda"
        elif torch.backends.mps.is_available(): return "mps"
        return "cpu"
    
    @staticmethod
    def setup_device_specific_settings(device: str):
        if device == "cuda":
            use_bf16 = torch.cuda.is_bf16_supported()
            return not use_bf16, use_bf16  # fp16, bf16
        return False, False  # MPS/CPU: no mixed precision
```

### Memory Optimization Strategy
```python
# Training optimizations
model.config.use_cache = False              # Disable KV cache
model.gradient_checkpointing_enable()       # Trade compute for memory

# Data loading optimizations
DataCollatorForLanguageModeling(
    pad_to_multiple_of=8,    # Tensor core optimization
    return_tensors="pt"
)

# Training arguments
TrainingArguments(
    per_device_train_batch_size=1,      # Memory-safe default
    gradient_accumulation_steps=8,      # Maintain effective batch size
    dataloader_pin_memory=False,        # MPS compatibility
    fp16=use_fp16, bf16=use_bf16       # Device-specific precision
)
```

## 📁 Data Flow Architecture

### Training Data Pipeline
```
Raw Dataset → Alpaca Formatting → Tokenization → Batching → Training
     ↓              ↓                  ↓           ↓          ↓
tatsu-lab/alpaca  Template         HF Tokenizer  Dynamic   PEFT Model
   [1% split]     Application      [truncation]  Padding   [LoRA/DoRA]
```

### Alpaca Prompt Template
```python
def format_alpaca_prompt(example):
    return f"""### Instruction:
{example['instruction']}
### Input:
{example.get('input', '')}
### Response:
{example['output']}"""
```

### Inference Data Flow
```
User Prompt → Tokenization → Model Forward → Generation → Decoding
     ↓             ↓              ↓             ↓          ↓
CLI Argument   HF Tokenizer   PEFT Model   Sampling   Text Output
               [padding]      [base+adapter] [temp/top_p] [detokenized]
```

## 🔒 Error Handling Architecture

### Exception Hierarchy
```python
LlamaLoRAError (Base)
├── ConfigurationError     # Invalid configuration
├── ModelLoadingError     # Model/tokenizer loading failures  
├── DatasetError         # Dataset loading/processing issues
├── TrainingError        # Training execution problems
├── AdapterError         # PEFT adapter operations
└── ValidationError      # Runtime validation failures
```

### Error Context Preservation
```python
try:
    model = AutoModelForCausalLM.from_pretrained(model_id)
except Exception as e:
    raise ModelLoadingError(
        f"Failed to load model '{model_id}': {str(e)}"
    ) from e  # Preserves original traceback
```

## 🏃‍♂️ Performance Architecture

### Automatic Optimization Features
1. **Device Detection**: Optimal device selection with fallbacks
2. **Mixed Precision**: Automatic fp16/bf16 based on device capability
3. **Flash Attention**: FlashAttention2 with graceful degradation
4. **Memory Management**: Gradient checkpointing, dynamic padding
5. **Tensor Cores**: Batch size optimization for NVIDIA GPUs

### Scalability Design
- **Modular Components**: Each pipeline stage independently scalable
- **Configuration-Driven**: Easy parameter tuning without code changes
- **Memory Efficient**: Designed for constrained environments
- **Multi-Experiment**: Batch experiment execution with resource management

## 🔄 Extensibility Points

### Adding New Models
1. Update `target_modules` in PEFT configuration
2. Modify tokenizer setup if needed
3. Add model-specific optimization flags

### Adding New Datasets
1. Implement dataset formatter in `TokenizerUtils`
2. Add dataset configuration in Pydantic schema
3. Update experiment presets

### Adding New PEFT Methods
1. Extend PEFT configuration in schema.py
2. Update model loading logic in train.py
3. Add validation rules in validate.py