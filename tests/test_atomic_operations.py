"""Test suite for atomic operations."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from src.llama_lora.utils.storage import AtomicSaver
from src.llama_lora.utils.exceptions import AtomicOperationError


class TestAtomicSaver:
    """Test atomic save operations."""

    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.save_pretrained = Mock()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.save_pretrained = Mock()
        return tokenizer

    def test_successful_atomic_operation(self, temp_base_dir):
        """Test successful atomic operation commits changes."""
        with AtomicSaver(str(temp_base_dir)).atomic_operation() as saver:
            temp_dir = saver._create_temp_dir("test")
            saver.operations = [(temp_dir, str(temp_base_dir / "final"))]

            (temp_dir / "test_file.txt").write_text("test content")

        assert saver.success is True
        assert (temp_base_dir / "final" / "test_file.txt").exists()

    def test_failed_atomic_operation_cleanup(self, temp_base_dir):
        """Test failed operation cleans up temporary directories."""
        with pytest.raises(AtomicOperationError):
            with AtomicSaver(str(temp_base_dir)).atomic_operation() as saver:
                temp_dir = saver._create_temp_dir("test")
                (temp_dir / "test_file.txt").write_text("test content")
                raise ValueError("Simulated failure")

        remaining_dirs = list(temp_base_dir.glob("test_*"))
        assert len(remaining_dirs) == 0

    def test_save_model_artifacts_success(
        self, temp_base_dir, mock_model, mock_tokenizer
    ):
        """Test successful model artifacts saving."""
        adapter_dir = str(temp_base_dir / "adapter")
        tokenizer_dir = str(temp_base_dir / "tokenizer")

        with AtomicSaver(str(temp_base_dir)).atomic_operation() as saver:
            saver.save_model_artifacts(
                model=mock_model,
                tokenizer=mock_tokenizer,
                adapter_dir=adapter_dir,
                tokenizer_dir=tokenizer_dir,
            )

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        assert Path(adapter_dir).exists()
        assert Path(tokenizer_dir).exists()

    def test_save_model_artifacts_failure(
        self, temp_base_dir, mock_model, mock_tokenizer
    ):
        """Test model artifacts saving failure handling."""
        mock_model.save_pretrained.side_effect = RuntimeError("Save failed")

        with pytest.raises(AtomicOperationError) as exc_info:
            with AtomicSaver(str(temp_base_dir)).atomic_operation() as saver:
                saver.save_model_artifacts(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    adapter_dir=str(temp_base_dir / "adapter"),
                    tokenizer_dir=str(temp_base_dir / "tokenizer"),
                )

        assert "Failed to save model artifacts" in str(exc_info.value)

    def test_save_merged_artifacts_success(
        self, temp_base_dir, mock_model, mock_tokenizer
    ):
        """Test successful merged artifacts saving."""
        merged_dir = str(temp_base_dir / "merged")

        with AtomicSaver(str(temp_base_dir)).atomic_operation() as saver:
            saver.save_merged_artifacts(
                merged_model=mock_model, tokenizer=mock_tokenizer, merged_dir=merged_dir
            )

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        assert Path(merged_dir).exists()
