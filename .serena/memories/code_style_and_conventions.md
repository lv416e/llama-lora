# Code Style and Conventions

## Python Style Guidelines
- **Standard**: PEP 8 compliance
- **Indentation**: 4 spaces (no tabs)
- **String Formatting**: Prefer f-strings over .format() or %
- **Line Length**: Follow ruff defaults (typically 88-100 characters)

## Naming Conventions
- **Files/Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE` (especially in config files)
- **Private methods**: `_leading_underscore`

## Import Organization
Follow this order with blank lines between groups:
1. Standard library imports
2. Third-party imports
3. Local/project imports

Example:
```python
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llama_lora.utils.common import setup_logging
```

## Configuration Management
- **No hardcoding**: All configuration should be in config files
- **Hydra + Pydantic**: Use structured configuration with validation
- **Config Location**: `config/` directory with YAML files
- **Schema**: Defined in `config/schema.py` with Pydantic models

## Type Hints and Documentation
- Use type hints for function parameters and return types
- Prefer descriptive docstrings for complex functions
- Use Pydantic Field() with descriptions for config classes

## Error Handling
- Use specific exception types
- Provide informative error messages
- Implement graceful fallbacks (e.g., Flash Attention → standard attention)

## Testing Guidelines
- Test files: `test_*.py` in `tests/` directory
- Use pytest framework
- Prefer CPU-friendly, quick tests
- Focus on configuration validation and utility functions