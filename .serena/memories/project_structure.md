# Project Structure

## Directory Layout
```
llama-lora/
├── config/                 # Configuration files
│   ├── config.yaml        # Main Hydra config
│   ├── schema.py          # Pydantic configuration schemas
│   └── experiment/        # Experiment presets
│       ├── quick_test.yaml
│       ├── full_training.yaml
│       └── default.yaml
├── src/llama_lora/        # Main package source
│   ├── __init__.py
│   ├── train.py           # Training entry point
│   ├── infer.py           # Inference entry point
│   ├── merge.py           # Model merging entry point
│   ├── baseline.py        # Baseline evaluation
│   ├── validate.py        # Configuration validation
│   ├── experiment.py      # Experiment utilities
│   └── utils/             # Utility modules
│       ├── common.py      # Common utilities
│       └── exceptions.py  # Custom exceptions
├── tests/                 # Test suite
│   ├── test_config_validation.py
│   ├── test_tokenizer_utils.py
│   └── test_path_manager.py
├── examples/              # Example notebooks/scripts
├── docs/                  # Documentation
├── outputs/               # Generated artifacts (gitignored)
│   ├── experiments/       # Organized by experiment name
│   │   └── {exp_name}/
│   │       └── runs/
│   │           └── {run_id}/
│   │               ├── artifacts/ # Models, adapters
│   │               ├── logs/      # TensorBoard logs
│   │               └── metadata/  # Experiment metadata
├── pyproject.toml         # Python package configuration
├── uv.lock               # Dependency lock file
├── README.md             # User documentation
└── AGENTS.md             # Development guidelines (legacy)
```

## Key Modules

### Entry Points (CLI scripts)
- `llama-lora-train` → `llama_lora.train:main`
- `llama-lora-infer` → `llama_lora.infer:main`
- `llama-lora-merge` → `llama_lora.merge:main`
- `llama-lora-baseline` → `llama_lora.baseline:main`

### Configuration System
- **Hydra**: Flexible command-line configuration override
- **Pydantic**: Runtime validation and type safety
- **YAML**: Human-readable configuration files

### Output Organization
- Structured by experiment name and run ID
- Automatic timestamp-based run IDs
- Separate directories for artifacts, logs, and metadata