"""Standardized error handling utilities."""

import functools
import logging
from typing import Any, Callable, Type

from .exceptions import LlamaLoRAError, StorageError, ModelLoadingError


logger = logging.getLogger(__name__)


def handle_storage_errors(
    operation: str = "", raise_as: Type[Exception] = StorageError
):
    """Decorator for standardized storage error handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except (OSError, IOError, PermissionError) as e:
                error_msg = f"Storage operation failed: {str(e)}"
                if operation:
                    error_msg = f"{operation} failed: {str(e)}"

                logger.error(error_msg, exc_info=True)

                if raise_as == StorageError:
                    raise StorageError(error_msg, operation=operation) from e
                else:
                    raise raise_as(error_msg) from e

        return wrapper

    return decorator


def handle_model_errors(operation: str = ""):
    """Decorator for model-related error handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Model operation failed: {str(e)}"
                if operation:
                    error_msg = f"{operation} failed: {str(e)}"

                logger.error(error_msg, exc_info=True)
                raise ModelLoadingError(error_msg) from e

        return wrapper

    return decorator


class ErrorContext:
    """Context manager for operation-specific error handling."""

    def __init__(self, operation: str, raise_as: Type[Exception] = LlamaLoRAError):
        self.operation = operation
        self.raise_as = raise_as

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, Exception):
            if not issubclass(exc_type, LlamaLoRAError):
                error_msg = f"{self.operation} failed: {str(exc_val)}"
                logger.error(error_msg, exc_info=True)
                raise self.raise_as(error_msg) from exc_val
        return False
