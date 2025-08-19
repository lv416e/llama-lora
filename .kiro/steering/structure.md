# Project Structure & Organization

## Directory Layout

```
llama-lora/
├── src/llama_lora/           # Main package source code
│   ├── __init__.py
│   ├── train.py              # Training pipeline entry point
│   ├── infer.py              # Inference pipeline
│   ├── merge.py              # Adapter merging utility
│   ├── baseline.py           # Base model evaluation
│   ├── validate.py           # Configuration validation
│   ├── experiment.py         # Experiment management
│   └── utils/                # Utility modules
│       ├── common.py         # Device management, tokenizer utils
│       ├── storage.py        # Path management, file operations
│       ├── exceptions.py     # Custom exception classes
│       ├── errors.py         # Error handling utilities
│       └── results.py        # Results processing
├── config/                   # Hydra configuration files
│   ├── config.yaml           # Base configuration
│   ├── schema.py             # Pydantic validation schemas
│   ├── experiment/           # Experiment presets
│   └── inference/            # Inference configurations
├── tests/                    # Test suite
├── examples/                 # Example notebooks and data
├── outputs/                  # Generated outputs (structured)
└── docs/                     # Documentation
```

## Configuration Architecture

### Hydra Configuration System
- **Base Config**: `config/config.yaml` - minimal default settings
- **Experiment Configs**: `config/experiment/` - full training presets
- **Inference Configs**: `config/inference/` - inference-specific settings
- **Schema Validation**: `config/schema.py` - Pydantic models for type safety

### Configuration Composition
```bash
# Override individual parameters
uv run python -m llama_lora.train training.lr=1e-5

# Use experiment presets
uv run python -m llama_lora.train +experiment=default-full

# Combine multiple configs
uv run python -m llama_lora.train +experiment=quick_test training.epochs=1
```

## Output Structure

### Structured Experiment Organization
```
outputs/
└── experiments/
    └── {experiment_name}/
        └── runs/
            └── {run_id}/
                ├── artifacts/
                │   ├── adapter/      # LoRA/DoRA weights
                │   ├── tokenizer/    # Tokenizer files
                │   └── merged/       # Merged model (optional)
                ├── logs/             # TensorBoard logs
                └── metadata/         # Experiment metadata
```

### Auto-Generated Paths
- **Run ID**: Timestamp-based unique identifier
- **Experiment Name**: Configurable grouping (default: "default")
- **Artifacts**: Separate directories for different output types
- **Metadata**: JSON files with complete experiment configuration

## Code Organization Patterns

### Module Structure
- **Entry Points**: Each main script (`train.py`, `infer.py`) is a Hydra app
- **Utilities**: Shared functionality in `utils/` with clear separation of concerns
- **Error Handling**: Custom exceptions in `utils/exceptions.py`
- **Configuration**: Dual system (Hydra dataclasses + Pydantic validation)

### Import Conventions
```python
# Relative imports within package
from .utils.common import DeviceManager, TokenizerUtils
from .utils.exceptions import ModelLoadingError

# External dependencies
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
```

### Testing Structure
- **Test Files**: Mirror source structure (`test_*.py`)
- **Fixtures**: Shared test utilities and mock data
- **Coverage**: Focus on critical paths (config validation, error handling)

## Development Workflow

### Adding New Features
1. **Configuration**: Add new parameters to `config/schema.py`
2. **Implementation**: Create/modify modules in `src/llama_lora/`
3. **Testing**: Add corresponding tests in `tests/`
4. **Documentation**: Update relevant steering documents

### Experiment Management
- Use experiment configs for reproducible setups
- Leverage structured output paths for organization
- Save metadata for experiment tracking and comparison