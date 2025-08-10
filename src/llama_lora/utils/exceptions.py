"""Custom exception classes for the LLaMA-LoRA training pipeline."""


class LlamaLoRAError(Exception):
    """Base exception class for LLaMA-LoRA pipeline errors."""

    pass


class ConfigurationError(LlamaLoRAError):
    """Raised when there's an error in configuration."""

    pass


class ModelLoadingError(LlamaLoRAError):
    """Raised when model loading fails."""

    pass


class DatasetError(LlamaLoRAError):
    """Raised when there's an error with dataset loading or processing."""

    pass


class TrainingError(LlamaLoRAError):
    """Raised when training fails."""

    pass


class ValidationError(LlamaLoRAError):
    """Raised when validation fails."""

    pass


class AdapterError(LlamaLoRAError):
    """Raised when there's an error with PEFT adapters."""

    pass
