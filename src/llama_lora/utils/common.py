"""Common utility functions for the LLaMA-LoRA training pipeline."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np
from transformers import PreTrainedTokenizer


class DeviceManager:
    """Centralized device detection and management."""

    @staticmethod
    def detect_device() -> str:
        """Detect the best available device for computation.

        Returns:
            Device string ('cuda', 'mps', or 'cpu').
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def setup_device_specific_settings(device: str) -> tuple[bool, bool]:
        """Configure device-specific training settings.

        Args:
            device: Target device string.

        Returns:
            Tuple of (use_fp16, use_bf16) boolean flags.
        """
        if device != "cuda":
            return False, False

        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
        return use_fp16, use_bf16


class SeedManager:
    """Centralized random seed management for reproducibility."""

    @staticmethod
    def setup_seed(seed: int) -> None:
        """Set random seeds for all relevant libraries.

        Args:
            seed: Random seed value.
        """
        random.seed(seed)
        torch.manual_seed(seed)

        try:
            np.random.seed(seed)
        except ImportError:
            pass

        # Set deterministic algorithms for CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class TokenizerUtils:
    """Utility functions for tokenizer management."""

    @staticmethod
    def setup_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """Configure tokenizer with proper padding token.

        Args:
            tokenizer: Pre-trained tokenizer instance.

        Returns:
            Configured tokenizer.
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def format_alpaca_prompt(
        example: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_length: int
    ) -> Dict[str, Any]:
        """Format dataset example into Alpaca instruction format with robustness.

        Args:
            example: Dataset example containing 'instruction', 'input', 'output' keys.
            tokenizer: Pre-trained tokenizer for encoding.
            max_length: Maximum sequence length.

        Returns:
            Dict containing tokenized inputs.

        Note:
            Handles missing or empty fields gracefully with safe defaults.
        """
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip()
        output = example.get("output", "").strip()

        if not instruction:
            instruction = "Please respond to the following."
        if not output:
            output = "I understand."

        if input_text:
            text = (
                f"### Instruction:\n{instruction}\n"
                f"### Input:\n{input_text}\n"
                f"### Response:\n{output}"
            )
        else:
            text = (
                f"### Instruction:\n{instruction}\n"
                f"### Input:\n\n"
                f"### Response:\n{output}"
            )

        return tokenizer(
            text, truncation=True, max_length=max_length, padding="max_length"
        )


class PathManager:
    """Utility functions for path and directory management."""

    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure directory exists and return Path object.

        Args:
            path: Directory path string.

        Returns:
            Path object.

        Raises:
            PermissionError: If directory cannot be created.
        """
        dir_path = Path(path)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path
        except PermissionError as e:
            raise PermissionError(f"Cannot create directory: {dir_path}") from e

    @staticmethod
    def validate_directory_exists(path: str, purpose: str = "directory") -> None:
        """Validate that a directory exists.

        Args:
            path: Directory path to check.
            purpose: Description of what the directory is for (for error messages).

        Raises:
            FileNotFoundError: If directory does not exist.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{purpose.capitalize()} not found at '{path}'")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress verbose logs from transformers and other libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def get_optimal_num_processes() -> int:
    """Get optimal number of processes for data processing.

    Returns:
        Number of processes to use.
    """
    return max(1, (os.cpu_count() or 2) // 2)
