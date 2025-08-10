"""Test suite for error handling functionality."""

import pytest
import tempfile
import os
from unittest.mock import patch

from src.llama_lora.utils.errors import (
    handle_storage_errors,
    handle_model_errors,
    ErrorContext,
)
from src.llama_lora.utils.exceptions import (
    StorageError,
    ModelLoadingError,
    LlamaLoRAError,
)
from src.llama_lora.utils.storage import PathManager


class TestStorageErrorDecorator:
    """Test storage error handling decorator."""

    def test_storage_decorator_success(self):
        """Test decorator allows successful operations."""

        @handle_storage_errors("test_operation")
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_storage_decorator_handles_os_error(self):
        """Test decorator converts OSError to StorageError."""

        @handle_storage_errors("test_operation")
        def failing_function():
            raise OSError("File not found")

        with pytest.raises(StorageError) as exc_info:
            failing_function()

        assert "test_operation failed" in str(exc_info.value)
        assert exc_info.value.operation == "test_operation"

    def test_storage_decorator_handles_permission_error(self):
        """Test decorator converts PermissionError to StorageError."""

        @handle_storage_errors("test_operation")
        def failing_function():
            raise PermissionError("Access denied")

        with pytest.raises(StorageError) as exc_info:
            failing_function()

        assert "test_operation failed" in str(exc_info.value)

    def test_storage_decorator_custom_exception(self):
        """Test decorator with custom exception type."""

        @handle_storage_errors("test_operation", raise_as=RuntimeError)
        def failing_function():
            raise IOError("IO failed")

        with pytest.raises(RuntimeError):
            failing_function()


class TestModelErrorDecorator:
    """Test model error handling decorator."""

    def test_model_decorator_success(self):
        """Test decorator allows successful operations."""

        @handle_model_errors("model_loading")
        def successful_function():
            return "model_loaded"

        result = successful_function()
        assert result == "model_loaded"

    def test_model_decorator_handles_exception(self):
        """Test decorator converts exceptions to ModelLoadingError."""

        @handle_model_errors("model_loading")
        def failing_function():
            raise ValueError("Invalid model")

        with pytest.raises(ModelLoadingError) as exc_info:
            failing_function()

        assert "model_loading failed" in str(exc_info.value)


class TestErrorContext:
    """Test ErrorContext context manager."""

    def test_error_context_success(self):
        """Test context manager allows successful operations."""
        with ErrorContext("test_operation"):
            result = "success"

        assert result == "success"

    def test_error_context_converts_exception(self):
        """Test context manager converts non-LlamaLoRA exceptions."""
        with pytest.raises(LlamaLoRAError) as exc_info:
            with ErrorContext("test_operation"):
                raise ValueError("Some error")

        assert "test_operation failed" in str(exc_info.value)

    def test_error_context_preserves_llamalora_exception(self):
        """Test context manager preserves LlamaLoRA exceptions."""
        with pytest.raises(StorageError):
            with ErrorContext("test_operation"):
                raise StorageError("Storage failed")

    def test_error_context_custom_exception(self):
        """Test context manager with custom exception type."""
        with pytest.raises(ModelLoadingError) as exc_info:
            with ErrorContext("test_operation", raise_as=ModelLoadingError):
                raise ValueError("Some error")

        assert "test_operation failed" in str(exc_info.value)


class TestPathManagerErrorHandling:
    """Test PathManager enhanced error handling."""

    def test_ensure_directory_success(self):
        """Test successful directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_dir")
            result = PathManager.ensure_directory(test_path)

            assert result.exists()
            assert result.is_dir()

    def test_ensure_directory_empty_path(self):
        """Test error handling for empty path."""
        with pytest.raises(ValueError) as exc_info:
            PathManager.ensure_directory("")

        assert "Directory path cannot be empty" in str(exc_info.value)

    def test_ensure_directory_file_exists(self):
        """Test error when path exists but is a file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(StorageError) as exc_info:
                PathManager.ensure_directory(temp_file.name)

            assert "Path exists but is not a directory" in str(exc_info.value)
            assert exc_info.value.operation == "directory_validation"

    @patch("pathlib.Path.mkdir")
    def test_ensure_directory_permission_error(self, mock_mkdir):
        """Test handling of permission errors."""
        mock_mkdir.side_effect = PermissionError("Access denied")

        with pytest.raises(StorageError) as exc_info:
            PathManager.ensure_directory("/test/path")

        assert "Cannot create directory" in str(exc_info.value)
        assert exc_info.value.operation == "directory_creation"

    def test_ensure_directory_writable_success(self):
        """Test successful write permission check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise exception
            PathManager.ensure_directory_writable(temp_dir)

    @patch("pathlib.Path.touch")
    def test_ensure_directory_writable_permission_error(self, mock_touch):
        """Test handling write permission errors."""
        mock_touch.side_effect = PermissionError("Access denied")

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(StorageError) as exc_info:
                PathManager.ensure_directory_writable(temp_dir)

            assert "Directory is not writable" in str(exc_info.value)
            assert exc_info.value.operation == "write_permission_test"

    def test_validate_directory_exists_success(self):
        """Test successful directory validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise exception
            PathManager.validate_directory_exists(temp_dir, "test directory")

    def test_validate_directory_exists_not_found(self):
        """Test directory not found error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            PathManager.validate_directory_exists("/nonexistent/path", "test directory")

        assert "Test directory not found" in str(exc_info.value)

    def test_directory_exists_true(self):
        """Test directory_exists returns True for existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert PathManager.directory_exists(temp_dir) is True

    def test_directory_exists_false(self):
        """Test directory_exists returns False for non-existing directory."""
        assert PathManager.directory_exists("/nonexistent/path") is False
        assert PathManager.directory_exists("") is False
        assert PathManager.directory_exists(None) is False
