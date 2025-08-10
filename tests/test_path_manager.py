"""Tests for PathManager functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from src.llama_lora.utils.common import PathManager


class TestPathManager:
    """Test cases for PathManager class."""
    
    def test_ensure_directory_creates_new_directory(self):
        """Test that ensure_directory creates a new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_directory")
            
            result = PathManager.ensure_directory(new_dir)
            
            assert os.path.isdir(new_dir)
            assert isinstance(result, Path)
            assert str(result) == new_dir

    def test_ensure_directory_with_existing_directory(self):
        """Test that ensure_directory works with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = PathManager.ensure_directory(temp_dir)
            
            assert os.path.isdir(temp_dir)
            assert isinstance(result, Path)
            assert str(result) == temp_dir

    def test_ensure_directory_creates_nested_directories(self):
        """Test that ensure_directory creates nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "level1", "level2", "level3")
            
            result = PathManager.ensure_directory(nested_dir)
            
            assert os.path.isdir(nested_dir)
            assert isinstance(result, Path)

    def test_validate_directory_exists_with_valid_directory(self):
        """Test validate_directory_exists with an existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise an exception
            PathManager.validate_directory_exists(temp_dir, "test directory")

    def test_validate_directory_exists_with_invalid_directory(self):
        """Test validate_directory_exists with non-existing directory."""
        non_existent_dir = "/path/that/does/not/exist"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            PathManager.validate_directory_exists(non_existent_dir, "test directory")
        
        assert "Test directory not found" in str(exc_info.value)
        assert non_existent_dir in str(exc_info.value)

    def test_validate_directory_exists_with_file_instead_of_directory(self):
        """Test validate_directory_exists when path points to a file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(FileNotFoundError):
                PathManager.validate_directory_exists(temp_file.name, "directory")

    def test_ensure_directory_with_permission_error(self):
        """Test ensure_directory behavior with permission error."""
        # Try to create directory in root (should fail on most systems)
        restricted_path = "/root/restricted_directory"
        
        with pytest.raises(PermissionError) as exc_info:
            PathManager.ensure_directory(restricted_path)
        
        assert "Cannot create directory" in str(exc_info.value)