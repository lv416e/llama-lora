"""Configuration validation utility for LLaMA-LoRA fine-tuning setup.

This script validates configuration settings using Pydantic models and provides
detailed error messages for invalid configurations. It can be used standalone
or imported by other scripts for configuration checking.
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path to allow importing schema
sys.path.append(str(Path(__file__).resolve().parents[2]))

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from config.schema import (
    ModelConfig,
    TrainingConfig,
    PEFTConfig,
    DatasetConfig,
    OutputConfig,
    LoggingConfig,
)


class ConfigValidator:
    """Configuration validator with comprehensive checks."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_model_config(self, model_cfg: DictConfig) -> bool:
        """Validate model configuration.

        Args:
            model_cfg: Model configuration section.

        Returns:
            bool: True if validation passes.
        """
        try:
            ModelConfig(**model_cfg)
        except ValidationError as e:
            self.errors.append(f"Model config validation failed: {e}")
            return False

        if model_cfg.seq_len > 4096:
            self.warnings.append(
                f"Large sequence length ({model_cfg.seq_len}) may cause memory issues"
            )

        return True

    def validate_training_config(self, training_cfg: DictConfig) -> bool:
        """Validate training configuration.

        Args:
            training_cfg: Training configuration section.

        Returns:
            bool: True if validation passes.
        """
        try:
            TrainingConfig(**training_cfg)
        except ValidationError as e:
            self.errors.append(f"Training config validation failed: {e}")
            return False

        if training_cfg.lr > 1e-3:
            self.warnings.append(
                f"High learning rate ({training_cfg.lr}) may cause instability"
            )

        if training_cfg.batch_size * training_cfg.gradient_accumulation_steps > 64:
            effective_batch = (
                training_cfg.batch_size * training_cfg.gradient_accumulation_steps
            )
            self.warnings.append(
                f"Large effective batch size ({effective_batch}) may reduce training dynamics"
            )

        return True

    def validate_peft_config(self, peft_cfg: DictConfig) -> bool:
        """Validate PEFT configuration.

        Args:
            peft_cfg: PEFT configuration section.

        Returns:
            bool: True if validation passes.
        """
        try:
            PEFTConfig(**peft_cfg)
        except ValidationError as e:
            self.errors.append(f"PEFT config validation failed: {e}")
            return False

        if peft_cfg.lora_alpha > peft_cfg.r * 4:
            self.warnings.append(
                f"LoRA alpha ({peft_cfg.lora_alpha}) is very high compared to rank ({peft_cfg.r}). "
                f"Consider reducing alpha or increasing rank."
            )

        return True

    def validate_dataset_config(self, dataset_cfg: DictConfig) -> bool:
        """Validate dataset configuration.

        Args:
            dataset_cfg: Dataset configuration section.

        Returns:
            bool: True if validation passes.
        """
        try:
            DatasetConfig(**dataset_cfg)
        except ValidationError as e:
            self.errors.append(f"Dataset config validation failed: {e}")
            return False

        return True

    def validate_output_config(self, output_cfg: DictConfig) -> bool:
        """Validate output configuration.

        Args:
            output_cfg: Output configuration section.

        Returns:
            bool: True if validation passes.
        """
        try:
            # Create OutputConfig instance to trigger automatic path generation
            pydantic_config = OutputConfig(**output_cfg)
        except ValidationError as e:
            self.errors.append(f"Output config validation failed: {e}")
            return False

        # Validate base directory
        base_dir = Path(output_cfg.base_output_dir)
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.errors.append(f"Cannot create output directory: {base_dir}")
            return False

        # Validate experiment name
        if not output_cfg.get("experiment_name"):
            self.warnings.append("No experiment name specified, using 'default'")

        # Check for conflicting manual path specifications
        manual_paths = [
            "adapter_dir",
            "tokenizer_dir",
            "merged_dir",
            "log_dir",
            "metadata_dir",
        ]
        has_manual_paths = any(output_cfg.get(path) for path in manual_paths)

        if has_manual_paths:
            self.warnings.append(
                "Manual path specifications detected. These will be ignored in favor of "
                "structured auto-generated paths. Remove manual paths from config to avoid confusion."
            )

        # Log the structured paths that will be used
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Structured output paths will be generated under: {pydantic_config.base_output_dir}/experiments/{pydantic_config.experiment_name}/runs/{{run_id}}/"
        )

        return True

    def validate_logging_config(self, logging_cfg: DictConfig) -> bool:
        """Validate logging configuration.

        Args:
            logging_cfg: Logging configuration section.

        Returns:
            bool: True if validation passes.
        """
        try:
            LoggingConfig(**logging_cfg)
        except ValidationError as e:
            self.errors.append(f"Logging config validation failed: {e}")
            return False

        return True

    def validate_full_config(self, cfg: DictConfig) -> bool:
        """Validate complete configuration.

        Args:
            cfg: Complete configuration object.

        Returns:
            bool: True if all validations pass.
        """
        success = True

        success &= self.validate_model_config(cfg.model)
        success &= self.validate_training_config(cfg.training)
        success &= self.validate_peft_config(cfg.peft)
        success &= self.validate_dataset_config(cfg.dataset)
        success &= self.validate_output_config(cfg.output)
        success &= self.validate_logging_config(cfg.logging)

        if cfg.training.epochs == 1 and cfg.training.eval_steps > 100:
            self.warnings.append(
                "Single epoch training with high eval_steps may not perform evaluation"
            )

        return success

    def print_results(self) -> None:
        """Print validation results."""
        print("Configuration Validation Results:")
        print("=" * 50)

        if not self.errors and not self.warnings:
            print("✅ All validations passed!")
            return

        if self.errors:
            print("❌ Validation Errors:")
            for error in self.errors:
                print(f"  • {error}")
            print()

        if self.warnings:
            print("⚠️  Validation Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()

        if self.errors:
            print("Please fix the errors before proceeding.")
        else:
            print("Configuration is valid but has warnings.")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for configuration validation.

    Args:
        cfg: Hydra configuration object.
    """
    print("Validating configuration...")
    print(
        f"Configuration file: {hydra.core.global_hydra.GlobalHydra.instance().hydra_cfg.runtime.config_sources}"
    )
    print()

    validator = ConfigValidator()
    success = validator.validate_full_config(cfg)
    validator.print_results()

    if not success:
        sys.exit(1)

    print("\nConfiguration Summary:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
