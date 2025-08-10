"""Atomic file operations for robust artifact saving."""

import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from transformers import AutoTokenizer, PreTrainedModel
from .exceptions import AtomicOperationError


class AtomicSaver:
    """Manages atomic save operations with rollback capability."""

    def __init__(
        self,
        base_dir: str,
        required_space_gb: Optional[float] = None,
        show_progress: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.temp_dirs: List[Path] = []
        self.operations: List[Tuple[Path, str]] = []
        self.success = False
        self.required_space_gb = required_space_gb
        self.show_progress = show_progress and tqdm is not None

    @contextmanager
    def atomic_operation(self):
        """Context manager for atomic operations with automatic cleanup."""
        if self.required_space_gb:
            if not DiskSpaceManager.check_available_space(
                str(self.base_dir.parent), self.required_space_gb
            ):
                raise AtomicOperationError(
                    f"Insufficient disk space: {self.required_space_gb}GB required"
                )

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
        operations_iter = (
            tqdm(self.operations, desc="Committing artifacts")
            if self.show_progress
            else self.operations
        )

        for temp_path, final_path in operations_iter:
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


class DiskSpaceManager:
    """Disk space monitoring and management utilities."""

    @staticmethod
    def get_disk_usage(path: str) -> tuple[int, int, int]:
        """Get disk usage statistics for given path.

        Args:
            path: Directory path to check.

        Returns:
            Tuple of (total, used, free) bytes.
        """
        import shutil

        total, used, free = shutil.disk_usage(path)
        return total, used, free

    @staticmethod
    def check_available_space(path: str, required_gb: float) -> bool:
        """Check if sufficient disk space is available.

        Args:
            path: Directory path to check.
            required_gb: Required space in GB.

        Returns:
            True if sufficient space available.
        """
        try:
            _, _, free = DiskSpaceManager.get_disk_usage(path)
            required_bytes = required_gb * 1024**3
            return free >= required_bytes
        except OSError:
            return False

    @staticmethod
    def get_directory_size(path: str) -> int:
        """Calculate total size of directory in bytes.

        Args:
            path: Directory path.

        Returns:
            Total size in bytes.
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, FileNotFoundError):
            pass
        return total_size

    @staticmethod
    def cleanup_old_experiments(
        base_output_dir: str,
        max_experiments: Optional[int] = None,
        max_age_days: Optional[int] = None,
    ) -> List[str]:
        """Clean up old experiment directories based on policy.

        Args:
            base_output_dir: Base output directory.
            max_experiments: Maximum number of experiments to keep per experiment name.
            max_age_days: Maximum age in days for experiments.

        Returns:
            List of cleaned up experiment paths.
        """
        from .exceptions import StorageError

        cleaned_paths = []
        experiments_dir = Path(base_output_dir) / "experiments"

        if not experiments_dir.exists():
            return cleaned_paths

        try:
            for experiment_dir in experiments_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue

                runs_dir = experiment_dir / "runs"
                if not runs_dir.exists():
                    continue

                # Get all runs with timestamps
                runs_with_time = []
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir():
                        try:
                            # Extract timestamp from directory name or modification time
                            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
                            runs_with_time.append((run_dir, mtime))
                        except (OSError, ValueError):
                            continue

                # Sort by modification time (newest first)
                runs_with_time.sort(key=lambda x: x[1], reverse=True)

                # Apply cleanup policies
                to_remove = []

                # Age-based cleanup
                if max_age_days is not None:
                    cutoff_date = datetime.now() - timedelta(days=max_age_days)
                    to_remove.extend(
                        [
                            run_dir
                            for run_dir, mtime in runs_with_time
                            if mtime < cutoff_date
                        ]
                    )

                # Count-based cleanup (keep newest max_experiments)
                if (
                    max_experiments is not None
                    and len(runs_with_time) > max_experiments
                ):
                    to_remove.extend(
                        [run_dir for run_dir, _ in runs_with_time[max_experiments:]]
                    )

                # Remove duplicates and perform cleanup
                for run_dir in set(to_remove):
                    try:
                        shutil.rmtree(run_dir)
                        cleaned_paths.append(str(run_dir))
                    except (OSError, PermissionError):
                        continue

        except (OSError, PermissionError) as e:
            raise StorageError(
                f"Failed to clean up experiments: {str(e)}",
                path=str(experiments_dir),
                operation="cleanup",
            ) from e

        return cleaned_paths


class EfficientFileOperations:
    """Advanced file operations for large models."""

    @staticmethod
    def create_hardlink_if_possible(source: Path, target: Path) -> bool:
        """Create hardlink if possible, fallback to copy.

        Args:
            source: Source file path.
            target: Target file path.

        Returns:
            True if hardlink created, False if copied.
        """
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(source)
            return True
        except (OSError, NotImplementedError):
            shutil.copy2(source, target)
            return False

    @staticmethod
    def copy_with_progress(
        source: Path, target: Path, chunk_size: int = 1024 * 1024
    ) -> None:
        """Copy file with progress bar for large files.

        Args:
            source: Source file path.
            target: Target file path.
            chunk_size: Chunk size for copying.
        """
        if not tqdm:
            shutil.copy2(source, target)
            return

        file_size = source.stat().st_size
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(source, "rb") as src, open(target, "wb") as dst:
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Copying {source.name}",
            ) as pbar:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    pbar.update(len(chunk))
