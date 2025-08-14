# Project Structure

```
llama-lora/
├── config/                    # Hydra configuration files
│   ├── config.yaml           # Main configuration
│   ├── schema.py            # Pydantic/Hydra schema definitions
│   ├── experiment/          # Experiment presets
│   │   ├── default.yaml
│   │   ├── default-quick.yaml
│   │   ├── default-dora.yaml
│   │   └── default-full.yaml
│   ├── inference/           # Inference configurations
│   │   ├── base.yaml
│   │   ├── latest.yaml
│   │   └── manual.yaml
│   └── config-a40-*.yaml    # Hardware-specific configs (A40 GPU)
├── src/
│   └── llama_lora/          # Main package
│       ├── __init__.py
│       ├── train.py         # Main training script
│       ├── infer.py         # Inference script
│       ├── merge.py         # Adapter merging script
│       ├── baseline.py      # Baseline evaluation
│       ├── validate.py      # Config validation
│       ├── experiment.py    # Experiment management
│       └── utils/           # Utility modules
│           ├── common.py    # Common utilities
│           ├── errors.py    # Error handling
│           ├── exceptions.py # Custom exceptions
│           ├── results.py   # Results management
│           └── storage.py   # Storage utilities
├── tests/                   # Test suite
│   ├── test_config_validation.py
│   ├── test_tokenizer_utils.py
│   ├── test_error_handling.py
│   ├── test_path_manager.py
│   └── ...
├── examples/                # Usage examples
├── docs/                    # Documentation
├── pyproject.toml          # Project dependencies and metadata
├── README.md               # Project documentation
└── outputs/                # Training outputs (generated)
    └── experiments/        # Organized by experiment
        └── {exp_name}/
            └── runs/
                └── {run_id}/
                    ├── artifacts/  # Model artifacts
                    ├── logs/       # TensorBoard logs
                    └── metadata/   # Experiment metadata
```

## Key Components

### Configuration System
- **Hydra**: Dynamic configuration management with command-line overrides
- **Pydantic**: Type-safe validation with structured schemas
- **Experiment Presets**: Pre-configured settings for different scenarios

### Training Pipeline
1. Configuration loading and validation
2. Model and tokenizer initialization
3. Dataset preparation and preprocessing
4. PEFT (LoRA/DoRA) setup
5. Training with gradient accumulation
6. Evaluation and checkpointing
7. Artifact saving

### Inference Pipeline
1. Auto-discovery of latest trained model
2. Adapter and tokenizer loading
3. Model initialization with PEFT
4. Generation with configurable parameters

### Entry Points (CLI Commands)
- `llama-lora-train`: Main training command
- `llama-lora-infer`: Inference with fine-tuned model
- `llama-lora-merge`: Merge adapter into base model
- `llama-lora-baseline`: Evaluate base model performance