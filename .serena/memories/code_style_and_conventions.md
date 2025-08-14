# Code Style and Conventions

## Python Style Guide

### General Conventions
- **Python Version**: 3.12+ required
- **Style Guide**: PEP 8 with modern Python practices
- **Line Length**: 88 characters (Black/Ruff default)
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Double quotes preferred for strings
- **F-strings**: Preferred for string formatting

### Type Hints
- **Required**: All function signatures must have type hints
- **Style**: Using modern Python 3.10+ syntax
  ```python
  from typing import Optional, List, Dict, Any, Literal
  
  def process_data(
      data: List[str],
      config: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
      ...
  ```

### Docstrings
- **Format**: Google-style docstrings
- **Required for**: All modules, classes, and public functions
- **Example**:
  ```python
  """Module description.
  
  This module provides functionality for...
  """
  
  def function_name(param: str) -> str:
      """Brief description of function.
      
      Args:
          param: Description of parameter.
      
      Returns:
          Description of return value.
      
      Raises:
          ValueError: When validation fails.
      """
  ```

### Import Organization
1. Standard library imports
2. Third-party imports
3. Local application imports
- Each group separated by blank line
- Imports sorted alphabetically within groups

### Naming Conventions
- **Classes**: PascalCase (e.g., `ModelConfig`, `TrainingPipeline`)
- **Functions/Methods**: snake_case (e.g., `load_model`, `prepare_dataset`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_SEQUENCE_LENGTH`)
- **Private**: Leading underscore (e.g., `_internal_method`)

### Configuration Classes
- Use Pydantic `BaseModel` for validation
- Use Hydra `@dataclass` for CLI compatibility
- Include field descriptions and validation
- Example:
  ```python
  class ModelConfig(BaseModel):
      model_id: str = Field(
          default="meta-llama/Llama-3.2-1B-Instruct",
          description="Model identifier"
      )
      seq_len: int = Field(
          default=512,
          ge=1,
          le=8192,
          description="Maximum sequence length"
      )
  ```

### Error Handling
- Use custom exceptions from `utils.exceptions`
- Provide informative error messages
- Include context and suggestions for fixes
- Example:
  ```python
  try:
      result = process_data(input_data)
  except ValueError as e:
      raise ConfigurationError(
          f"Invalid configuration: {str(e)}"
      ) from e
  ```

### Testing Conventions
- Test files: `test_*.py` in `tests/` directory
- Use pytest fixtures for setup
- Mock external dependencies
- Test both success and failure cases
- Naming: `test_<function_name>_<scenario>`

### Code Organization
- Single responsibility principle
- Separate concerns (config, training, inference)
- Use utility modules for shared functionality
- Keep functions focused and testable

### Comments
- Use inline comments sparingly
- Explain "why" not "what"
- Update comments when code changes
- Remove TODO comments before committing

### Logging
- Use structured logging with appropriate levels
- Include context in log messages
- Use TensorBoard for training metrics
- Example:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  
  logger.info(f"Starting training with config: {config}")
  logger.warning(f"Low GPU memory: {available_memory}GB")
  ```

### File Headers
- Include module docstring at top of each file
- Brief description of module purpose
- No author/date comments (use git history)