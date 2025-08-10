# Code Style & Conventions

## Code Quality Philosophy
- **Minimal Comments**: Only WHY comments, not WHAT comments
- **Type Safety**: Full type hints with Pydantic validation
- **Error Handling**: Comprehensive exception handling with custom exceptions
- **Modular Design**: Clear separation of concerns with utility classes

## Import Organization
```python
# Standard library imports
import warnings
from typing import Any, Dict

# Third-party imports (grouped by functionality)
import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# Local imports
from .utils.common import DeviceManager, setup_logging
from .utils.exceptions import ModelLoadingError
```

## Naming Conventions
### Variables & Functions
- **snake_case**: All variables, functions, and module names
- **Descriptive names**: `device_manager.detect_device()` not `dm.detect()`
- **Clear intent**: `load_and_setup_tokenizer()` not `load_tokenizer()`

### Classes
- **PascalCase**: `DeviceManager`, `TokenizerUtils`, `PathManager`
- **Utility classes**: Static methods for related functionality
- **Manager pattern**: For complex stateful operations

### Constants & Configuration
- **SCREAMING_SNAKE_CASE**: For true constants (rare in this codebase)
- **Pydantic fields**: lowercase with Field() descriptors
- **Hydra configs**: lowercase nested structure

## Type Hints & Validation
### Function Signatures
```python
def load_and_process_dataset(
    cfg: DictConfig, tokenizer: AutoTokenizer
) -> tuple[Any, Any]:
    """Clear docstring explaining purpose."""
```

### Pydantic Models
```python
class TrainingConfig(BaseModel):
    lr: float = Field(default=2e-5, gt=0.0, description="Learning rate")
    batch_size: int = Field(default=4, ge=1, description="Training batch size")
```

## Documentation Style
### Docstrings
- **Google Style**: Args, Returns, Raises sections
- **Purpose-focused**: Explain WHY and WHEN to use
- **Error conditions**: Always document exceptions

```python
def setup_device_specific_settings(device: str) -> tuple[bool, bool]:
    """Configure device-specific training settings.
    
    Args:
        device: Target device string.
        
    Returns:
        Tuple of (use_fp16, use_bf16) boolean flags.
    """
```

### Meaningful Comments (Non-trivial only)
```python
# Latest best practices: auto device mapping, dtype, and Flash Attention
# Try flash_attention_2 first, fall back to eager if not available

# Use CPU for merging (less memory intensive)
device = "cpu"

# Optimize for tensor cores
pad_to_multiple_of=8
```

## Error Handling Patterns
### Custom Exceptions
```python
# Specific exception types for different failure modes
class ModelLoadingError(LlamaLoRAError): pass
class DatasetError(LlamaLoRAError): pass
class TrainingError(LlamaLoRAError): pass
class ConfigurationError(LlamaLoRAError): pass
```

### Exception Chaining
```python
try:
    model = AutoModelForCausalLM.from_pretrained(model_id)
except Exception as e:
    raise ModelLoadingError(
        f"Failed to load model '{model_id}': {str(e)}"
    ) from e
```

## Configuration Management
### Hydra Integration
- **Hierarchical configs**: Base + experiment overrides
- **Type validation**: Pydantic models for runtime checking
- **CLI overrides**: `training.lr=1e-5 model.use_dora=true`

### Default Values
```python
# Sensible defaults for quick testing
batch_size: int = 1           # Memory-safe default
epochs: int = 1               # Quick iteration
dataset_split: "train[:1%]"   # Fast dataset loading
```

## Utility Class Patterns
### Static Utility Classes
```python
class DeviceManager:
    """Centralized device detection and management."""
    
    @staticmethod
    def detect_device() -> str:
        """Detect optimal device with fallback chain."""
        
class TokenizerUtils:
    """Utility functions for tokenizer management."""
```

### Manager Classes for Complex Operations
- **PathManager**: Directory operations with error handling
- **SeedManager**: Reproducibility across frameworks
- **DeviceManager**: Device detection and optimization

## Logging Standards
### Structured Logging
```python
logger = setup_logging()  # Configured once per module

# Informational progress
logger.info(f"Loading tokenizer from '{path}'...")

# Warnings for fallbacks
logger.warning("FlashAttention2 not available, falling back to eager attention")

# Errors with context
logger.error(f"Training failed: {str(e)}", exc_info=True)
```

### Library Log Suppression
```python
# Suppress verbose logs from transformers and other libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
```

## Code Formatting
- **Ruff**: Automatic formatting and linting
- **Line length**: 88 characters (ruff default)
- **String quotes**: Double quotes preferred
- **Trailing commas**: For multi-line structures