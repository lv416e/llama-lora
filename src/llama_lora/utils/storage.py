"""Atomic file operations for robust artifact saving."""

import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
from contextlib import contextmanager

from transformers import AutoTokenizer, PreTrainedModel
from .exceptions import AtomicOperationError


class AtomicSaver:
    """Manages atomic save operations with rollback capability."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.temp_dirs: List[Path] = []
        self.operations: List[Tuple[Path, str]] = []
        self.success = False

    @contextmanager
    def atomic_operation(self):
        """Context manager for atomic operations with automatic cleanup."""
        try:
            yield self
            self.success = True
        except Exception as e:
            self._cleanup_temp_dirs()
            raise AtomicOperationError(f"Atomic operation failed: {str(e)}") from e
        finally:
            if self.success:
                self._commit_all()

    def save_model_artifacts(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        adapter_dir: str,
        tokenizer_dir: str,
    ) -> None:
        """Save model and tokenizer atomically."""
        temp_adapter = self._create_temp_dir("adapter")
        temp_tokenizer = self._create_temp_dir("tokenizer")

        try:
            model.save_pretrained(temp_adapter)
            tokenizer.save_pretrained(temp_tokenizer)

            self.operations = [
                (temp_adapter, adapter_dir),
                (temp_tokenizer, tokenizer_dir),
            ]
        except Exception as e:
            raise AtomicOperationError(
                f"Failed to save model artifacts: {str(e)}"
            ) from e

    def save_merged_artifacts(
        self, merged_model: PreTrainedModel, tokenizer: AutoTokenizer, merged_dir: str
    ) -> None:
        """Save merged model and tokenizer atomically."""
        temp_merged = self._create_temp_dir("merged")

        try:
            merged_model.save_pretrained(temp_merged)
            tokenizer.save_pretrained(temp_merged)

            self.operations = [(temp_merged, merged_dir)]
        except Exception as e:
            raise AtomicOperationError(
                f"Failed to save merged artifacts: {str(e)}"
            ) from e

    def _create_temp_dir(self, prefix: str) -> Path:
        """Create temporary directory for staging."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=self.base_dir.parent))
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def _commit_all(self) -> None:
        """Move all temporary directories to final destinations."""
        for temp_path, final_path in self.operations:
            final_path = Path(final_path)
            if final_path.exists():
                shutil.rmtree(final_path)
            shutil.move(str(temp_path), str(final_path))

        self.temp_dirs.clear()

    def _cleanup_temp_dirs(self) -> None:
        """Remove all temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()


class PathManager:
    """Enhanced path management with comprehensive error handling."""

    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure directory exists with enhanced error handling."""
        from .exceptions import StorageError

        if not path or not path.strip():
            raise ValueError("Directory path cannot be empty")

        dir_path = Path(path)

        if not dir_path.is_absolute() and not str(dir_path).startswith("./"):
            dir_path = Path(".") / dir_path

        if dir_path.exists() and not dir_path.is_dir():
            raise StorageError(
                f"Path exists but is not a directory: {dir_path}",
                path=str(dir_path),
                operation="directory_validation",
            )

        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path
        except (OSError, PermissionError) as e:
            raise StorageError(
                f"Cannot create directory: {dir_path}",
                path=str(dir_path),
                operation="directory_creation",
            ) from e

    @staticmethod
    def ensure_directory_writable(path: str) -> None:
        """Verify directory is writable."""
        from .exceptions import StorageError

        dir_path = PathManager.ensure_directory(path)

        test_file = dir_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise StorageError(
                f"Directory is not writable: {dir_path}",
                path=str(dir_path),
                operation="write_permission_test",
            ) from e

    @staticmethod
    def validate_directory_exists(path: str, purpose: str = "directory") -> None:
        """Validate that a directory exists."""
        if not path or not path.strip():
            raise ValueError(f"{purpose.capitalize()} path cannot be empty")

        if not Path(path).is_dir():
            raise FileNotFoundError(f"{purpose.capitalize()} not found at '{path}'")

    @staticmethod
    def directory_exists(path: str) -> bool:
        """Check if a directory exists."""
        if not path or not path.strip():
            return False
        return Path(path).is_dir()
