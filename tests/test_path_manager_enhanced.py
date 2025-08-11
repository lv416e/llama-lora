"""Enhanced tests for PathManager with new validation features."""

import tempfile
import os
import pytest
from pathlib import Path

from src.llama_lora.utils.storage import PathManager


class TestPathManagerEnhanced:
    """Test cases for enhanced PathManager functionality."""

    def test_ensure_directory_empty_string_validation(self):
        """Test that empty strings are rejected."""
        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            PathManager.ensure_directory("")

        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            PathManager.ensure_directory("   ")  # whitespace only

    def test_ensure_directory_relative_path_handling(self):
        """Test proper handling of relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Test relative path conversion
                result = PathManager.ensure_directory("test_rel_dir")

                # Should exist in current directory
                assert os.path.isdir("test_rel_dir")

                # Result should be a proper Path object
                assert isinstance(result, Path)

            finally:
                os.chdir(original_cwd)

    def test_ensure_directory_explicit_relative_path(self):
        """Test handling of explicit relative paths (starting with ./)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                result = PathManager.ensure_directory("./explicit_rel_dir")

                # Should exist
                assert os.path.isdir("explicit_rel_dir")
                assert isinstance(result, Path)

            finally:
                os.chdir(original_cwd)

    def test_validate_directory_exists_empty_string(self):
        """Test that validate_directory_exists rejects empty strings."""
        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            PathManager.validate_directory_exists("")

        with pytest.raises(ValueError, match="Adapter directory path cannot be empty"):
            PathManager.validate_directory_exists("   ", "adapter directory")

    def test_directory_exists_empty_string_handling(self):
        """Test that directory_exists handles empty strings gracefully."""
        assert PathManager.directory_exists("") is False
        assert PathManager.directory_exists("   ") is False

    def test_integration_with_output_config_paths(self):
        """Test PathManager integration with OutputConfig generated paths."""
        from config.schema import OutputConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config = OutputConfig(
                base_output_dir=temp_dir, experiment_name="integration_test"
            )

            # Test that all generated paths work with PathManager
            paths_to_test = [
                config.adapter_dir,
                config.tokenizer_dir,
                config.merged_dir,
                config.log_dir,
                config.metadata_dir,
            ]

            for path in paths_to_test:
                # Should not raise any errors
                result = PathManager.ensure_directory(path)
                assert isinstance(result, Path)
                assert os.path.isdir(path)

                # Validation should pass
                PathManager.validate_directory_exists(path)
                assert PathManager.directory_exists(path) is True

    def test_nested_directory_creation(self):
        """Test creation of deeply nested directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "level1", "level2", "level3", "level4")

            result = PathManager.ensure_directory(nested_path)

            assert os.path.isdir(nested_path)
            assert isinstance(result, Path)

            # All intermediate directories should exist
            for i in range(1, 5):
                intermediate_path = os.path.join(
                    temp_dir, *[f"level{j}" for j in range(1, i + 1)]
                )
                assert os.path.isdir(intermediate_path)

    def test_path_manager_with_special_characters(self):
        """Test PathManager with paths containing special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test various special characters that are valid in directory names
            special_dirs = [
                "dir-with-dashes",
                "dir_with_underscores",
                "dir.with.dots",
                "dir with spaces",  # Note: this should work on most systems
            ]

            for dir_name in special_dirs:
                full_path = os.path.join(temp_dir, dir_name)
                try:
                    result = PathManager.ensure_directory(full_path)
                    assert os.path.isdir(full_path)
                    assert isinstance(result, Path)
                except (OSError, ValueError):
                    # Some special characters might not be supported on all systems
                    # This is acceptable system-dependent behavior
                    pass

    def test_path_manager_error_message_quality(self):
        """Test that error messages are informative."""
        # Test empty string error message
        with pytest.raises(ValueError) as exc_info:
            PathManager.ensure_directory("")
        assert "cannot be empty" in str(exc_info.value)

        # Test directory not found error message
        with pytest.raises(FileNotFoundError) as exc_info:
            PathManager.validate_directory_exists("/nonexistent/path", "test directory")
        assert "Test directory not found" in str(exc_info.value)
        assert "/nonexistent/path" in str(exc_info.value)

    def test_concurrent_directory_creation(self):
        """Test that concurrent directory creation is safe."""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            shared_path = os.path.join(temp_dir, "shared", "directory")
            results = []
            errors = []

            def create_directory():
                try:
                    result = PathManager.ensure_directory(shared_path)
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            # Create multiple threads that try to create the same directory
            threads = [threading.Thread(target=create_directory) for _ in range(5)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # Should not have any errors
            assert len(errors) == 0
            assert len(results) == 5
            assert os.path.isdir(shared_path)
