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


class InferenceError(LlamaLoRAError):
    """Raised when inference process fails."""

    pass


class StorageError(LlamaLoRAError):
    """Raised for file system, I/O, or storage-related errors."""

    def __init__(self, message: str, path: str = "", operation: str = ""):
        super().__init__(message)
        self.path = path
        self.operation = operation

    def __str__(self):
        base_msg = super().__str__()
        if self.operation and self.path:
            return f"{base_msg} (operation: {self.operation}, path: {self.path})"
        elif self.path:
            return f"{base_msg} (path: {self.path})"
        return base_msg


class AtomicOperationError(StorageError):
    """Raised when atomic operation failures occur with rollback information."""

    def __init__(
        self, message: str, failed_step: str = "", completed_steps: list = None
    ):
        super().__init__(message)
        self.failed_step = failed_step
        self.completed_steps = completed_steps or []
