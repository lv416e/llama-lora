"""Structured configuration classes using Pydantic for LLaMA-LoRA fine-tuning.

This module defines type-safe configuration classes using Pydantic BaseModel
for validation and Hydra for flexible configuration management. All settings
are organized into logical groups for better maintainability and validation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field, model_validator


# Dataclass-based configuration for Hydra compatibility
@dataclass
class HydraModelConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_dora: bool = False
    seq_len: int = 512


@dataclass
class HydraDatasetConfig:
    dataset_id: str = "izumi-lab/llm-japanese-dataset"
    dataset_split: str = "train"
    val_ratio: float = 0.1


@dataclass
class HydraTrainingConfig:
    lr: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    seed: int = 42
    eval_steps: int = 200
    early_stopping_patience: int = 3


@dataclass
class HydraPEFTConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.target_modules is None:
            # Common LoRA targets for LLaMA-family models
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ]


@dataclass
class HydraOutputConfig:
    base_output_dir: str = "./outputs"
    experiment_name: str = "default"
    run_id: str = ""
    adapter_dir: str = ""
    tokenizer_dir: str = ""
    merged_dir: str = ""
    log_dir: str = ""
    metadata_dir: str = ""

    def __post_init__(self) -> None:
        # Only generate run_id if we're in training mode (not inference)
        # Inference should use existing run_ids or auto-discovery
        if not self.run_id:
            import os

            # Check if we're in inference mode by looking at the calling script
            import sys

            script_name = os.path.basename(sys.argv[0]) if sys.argv else ""

            # Don't auto-generate run_id for inference scripts
            if "infer" not in script_name.lower():
                from llama_lora.utils.common import (
                    generate_unique_run_id,
                    validate_run_id_uniqueness,
                )

                self.run_id = generate_unique_run_id()

                # Validate uniqueness and retry if needed
                max_attempts = 5
                for _ in range(max_attempts):
                    if validate_run_id_uniqueness(
                        self.run_id, self.base_output_dir, self.experiment_name
                    ):
                        break
                    self.run_id = generate_unique_run_id()
                else:
                    raise RuntimeError(
                        f"Failed to generate unique run_id after {max_attempts} attempts"
                    )

        # Only set paths if run_id is available
        if self.run_id:
            experiment_base = (
                f"{self.base_output_dir}/experiments/{self.experiment_name}"
            )
            run_base = f"{experiment_base}/runs/{self.run_id}"

            if not self.adapter_dir:
                self.adapter_dir = f"{run_base}/artifacts/adapter"
            if not self.tokenizer_dir:
                self.tokenizer_dir = f"{run_base}/artifacts/tokenizer"
            if not self.merged_dir:
                self.merged_dir = f"{run_base}/artifacts/merged"
            if not self.log_dir:
                self.log_dir = f"{run_base}/logs"
            if not self.metadata_dir:
                self.metadata_dir = f"{run_base}/metadata"


def save_experiment_metadata(
    cfg_dict: Dict[str, Any], output_config: "OutputConfig"
) -> str:
    """Save experiment metadata with comprehensive error handling."""
    import json
    from datetime import datetime
    from pathlib import Path

    try:
        metadata_dir = Path(output_config.metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "experiment_name": output_config.experiment_name,
            "run_id": output_config.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": cfg_dict,
            "output_paths": {
                "adapter_dir": output_config.adapter_dir,
                "tokenizer_dir": output_config.tokenizer_dir,
                "merged_dir": output_config.merged_dir,
                "log_dir": output_config.log_dir,
                "metadata_dir": output_config.metadata_dir,
            },
        }

        metadata_file = metadata_dir / "experiment_metadata.json"

        temp_file = metadata_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(metadata, f, indent=2)

        temp_file.rename(metadata_file)
        return str(metadata_file)

    except (OSError, IOError, PermissionError) as e:
        raise RuntimeError(f"Failed to save experiment metadata: {str(e)}") from e


def get_tensorboard_log_dir(output_config: "OutputConfig") -> str:
    """Get TensorBoard log directory path."""
    return output_config.log_dir


def start_tensorboard(output_config: "OutputConfig", port: int = 6006) -> str:
    """Start TensorBoard server for the experiment.

    Args:
        output_config: Output configuration
        port: Port to run TensorBoard on

    Returns:
        Command to start TensorBoard
    """
    log_dir = get_tensorboard_log_dir(output_config)
    command = f"tensorboard --logdir {log_dir} --port {port}"
    print(f"To start TensorBoard, run: {command}")
    print(f"Then open http://localhost:{port} in your browser")
    return command


def get_experiment_comparison_command(
    base_output_dir: str, experiment_names: List[str]
) -> str:
    """Generate TensorBoard command for comparing multiple experiments.

    Args:
        base_output_dir: Base output directory
        experiment_names: List of experiment names to compare

    Returns:
        Command to start TensorBoard with multiple experiments
    """
    log_dirs = []
    for exp_name in experiment_names:
        # Use structured path format: experiments/{exp_name}/runs/*/logs
        exp_log_dir = f"{base_output_dir}/experiments/{exp_name}/runs"
        log_dirs.append(f"{exp_name}:{exp_log_dir}")

    logdir_arg = ",".join(log_dirs)
    command = f"tensorboard --logdir_spec {logdir_arg}"
    print(f"To compare experiments, run: {command}")
    print("Note: This will aggregate logs from all runs within each experiment")
    return command


def get_single_experiment_tensorboard_command(
    base_output_dir: str, experiment_name: str
) -> str:
    """Generate TensorBoard command for viewing all runs of a single experiment.

    Args:
        base_output_dir: Base output directory
        experiment_name: Name of the experiment to view

    Returns:
        Command to start TensorBoard for the experiment
    """
    log_dir = f"{base_output_dir}/experiments/{experiment_name}/runs"
    command = f"tensorboard --logdir {log_dir}"
    print(f"To view all runs of experiment '{experiment_name}', run: {command}")
    print("Then open http://localhost:6006 in your browser")
    return command


@dataclass
class HydraLoggingConfig:
    report_to: str = "tensorboard"
    project_name: Optional[str] = None


@dataclass
class HydraInferenceConfig:
    auto_find_latest_run: bool = True
    fallback_run_id: Optional[str] = None
    run_id: Optional[str] = None
    adapter_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class HydraConfig:  # noqa
    """Main configuration class for Hydra compatibility (dataclass based)."""

    model: HydraModelConfig = field(default_factory=HydraModelConfig)
    dataset: HydraDatasetConfig = field(default_factory=HydraDatasetConfig)
    training: HydraTrainingConfig = field(default_factory=HydraTrainingConfig)
    peft: HydraPEFTConfig = field(default_factory=HydraPEFTConfig)
    output: HydraOutputConfig = field(default_factory=HydraOutputConfig)
    logging: HydraLoggingConfig = field(default_factory=HydraLoggingConfig)
    inference: HydraInferenceConfig = field(default_factory=HydraInferenceConfig)

    def __post_init__(self) -> None:
        if self.peft.target_modules is None:
            self.peft.target_modules = ["q_proj", "v_proj"]

    def to_pydantic_config(self) -> "Config":
        """Convert to Pydantic Config for validation."""
        return Config(
            model=ModelConfig(
                model_id=self.model.model_id,
                use_dora=self.model.use_dora,
                seq_len=self.model.seq_len,
            ),
            dataset=DatasetConfig(
                dataset_id=self.dataset.dataset_id,
                dataset_split=self.dataset.dataset_split,
                val_ratio=self.dataset.val_ratio,
            ),
            training=TrainingConfig(
                lr=self.training.lr,
                batch_size=self.training.batch_size,
                gradient_accumulation_steps=self.training.gradient_accumulation_steps,
                epochs=self.training.epochs,
                seed=self.training.seed,
                eval_steps=self.training.eval_steps,
                early_stopping_patience=self.training.early_stopping_patience,
            ),
            peft=PEFTConfig(
                r=self.peft.r,
                lora_alpha=self.peft.lora_alpha,
                lora_dropout=self.peft.lora_dropout,
                target_modules=self.peft.target_modules,
            ),
            output=OutputConfig(
                base_output_dir=self.output.base_output_dir,
                experiment_name=self.output.experiment_name,
                run_id=getattr(self.output, "run_id", ""),
                adapter_dir=getattr(self.output, "adapter_dir", ""),
                tokenizer_dir=getattr(self.output, "tokenizer_dir", ""),
                merged_dir=getattr(self.output, "merged_dir", ""),
                log_dir=getattr(self.output, "log_dir", ""),
                metadata_dir=getattr(self.output, "metadata_dir", ""),
            ),
            logging=LoggingConfig(
                report_to=self.logging.report_to,
                project_name=self.logging.project_name,
            ),
            inference=InferenceConfig(
                auto_find_latest_run=getattr(
                    self.inference, "auto_find_latest_run", True
                ),
                fallback_run_id=getattr(self.inference, "fallback_run_id", None),
                run_id=getattr(self.inference, "run_id", None),
                adapter_dir=getattr(self.inference, "adapter_dir", None),
                tokenizer_dir=getattr(self.inference, "tokenizer_dir", None),
                max_new_tokens=getattr(self.inference, "max_new_tokens", 128),
                temperature=getattr(self.inference, "temperature", 0.7),
                top_p=getattr(self.inference, "top_p", 0.9),
            ),
        )


# Original Pydantic classes for validation
class ModelConfig(BaseModel):
    """Model configuration with validation."""

    model_id: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", description="Model identifier"
    )
    use_dora: bool = Field(default=False, description="Use DoRA instead of LoRA")
    seq_len: int = Field(
        default=512, ge=1, le=8192, description="Maximum sequence length"
    )


class DatasetConfig(BaseModel):
    """Dataset configuration with validation."""

    dataset_id: str = Field(
        default="izumi-lab/llm-japanese-dataset", description="Dataset identifier"
    )
    dataset_split: str = Field(default="train", description="Dataset split to use")
    val_ratio: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Validation split ratio"
    )


class TrainingConfig(BaseModel):
    """Training configuration with validation."""

    lr: float = Field(default=2e-4, gt=0.0, lt=1.0, description="Learning rate")
    batch_size: int = Field(default=4, ge=1, le=128, description="Training batch size")
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, description="Gradient accumulation steps"
    )
    epochs: int = Field(
        default=3, ge=1, le=100, description="Number of training epochs"
    )
    seed: int = Field(default=42, ge=0, description="Random seed")
    eval_steps: int = Field(default=200, ge=1, description="Evaluation frequency")
    early_stopping_patience: int = Field(
        default=3, ge=1, description="Early stopping patience"
    )


class PEFTConfig(BaseModel):
    """PEFT (LoRA/DoRA) configuration with validation."""

    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        description="Target modules for LoRA",
    )


class OutputConfig(BaseModel):
    """Output configuration with validation."""

    base_output_dir: str = Field(
        default="./outputs", description="Base output directory"
    )
    experiment_name: str = Field(
        default="default", description="Experiment name for organization"
    )
    run_id: str = Field(default="", description="Run ID (auto-generated if empty)")
    adapter_dir: str = Field(default="", description="Adapter output directory")
    tokenizer_dir: str = Field(default="", description="Tokenizer output directory")
    merged_dir: str = Field(default="", description="Merged model output directory")
    log_dir: str = Field(default="", description="Logs output directory")
    metadata_dir: str = Field(default="", description="Metadata output directory")

    def model_post_init(self, __context) -> None:
        # Only generate run_id if we're in training mode (not inference)
        if not self.run_id:
            import os
            import sys

            script_name = os.path.basename(sys.argv[0]) if sys.argv else ""

            # Don't auto-generate run_id for inference scripts
            if "infer" not in script_name.lower():
                from llama_lora.utils.common import (
                    generate_unique_run_id,
                    validate_run_id_uniqueness,
                )

                self.run_id = generate_unique_run_id()

                # Validate uniqueness and retry if needed
                max_attempts = 5
                for _ in range(max_attempts):
                    if validate_run_id_uniqueness(
                        self.run_id, self.base_output_dir, self.experiment_name
                    ):
                        break
                    self.run_id = generate_unique_run_id()
                else:
                    raise RuntimeError(
                        f"Failed to generate unique run_id after {max_attempts} attempts"
                    )

        # Only set paths if run_id is available
        if self.run_id:
            experiment_base = (
                f"{self.base_output_dir}/experiments/{self.experiment_name}"
            )
            run_base = f"{experiment_base}/runs/{self.run_id}"

            if not self.adapter_dir:
                self.adapter_dir = f"{run_base}/artifacts/adapter"
            if not self.tokenizer_dir:
                self.tokenizer_dir = f"{run_base}/artifacts/tokenizer"
            if not self.merged_dir:
                self.merged_dir = f"{run_base}/artifacts/merged"
            if not self.log_dir:
                self.log_dir = f"{run_base}/logs"
            if not self.metadata_dir:
                self.metadata_dir = f"{run_base}/metadata"


class CleanupPolicy(BaseModel):
    """Cleanup policy configuration for experiment management."""

    enabled: bool = Field(default=False, description="Enable automatic cleanup")
    max_experiments_per_name: Optional[int] = Field(
        default=10, description="Maximum experiments to keep per experiment name"
    )
    max_age_days: Optional[int] = Field(
        default=30, description="Maximum age in days for experiments"
    )
    required_space_gb: Optional[float] = Field(
        default=5.0, description="Required disk space in GB for saving operations"
    )


class LoggingConfig(BaseModel):
    """Logging configuration with validation."""

    report_to: Literal["none", "wandb", "tensorboard"] = Field(
        default="none", description="Logging backend"
    )
    project_name: Optional[str] = Field(
        default=None, description="Experiment or project name for logging"
    )


class InferenceConfig(BaseModel):
    """Inference configuration with validation."""

    auto_find_latest_run: bool = Field(
        default=True, description="Automatically find the latest training run"
    )
    fallback_run_id: Optional[str] = Field(
        default=None, description="Fallback run ID if auto-discovery fails"
    )
    run_id: Optional[str] = Field(
        default=None, description="Specific run ID to use for inference"
    )
    adapter_dir: Optional[str] = Field(
        default=None, description="Direct path to adapter directory"
    )
    tokenizer_dir: Optional[str] = Field(
        default=None, description="Direct path to tokenizer directory"
    )

    # Generation parameters
    max_new_tokens: int = Field(
        default=128, ge=1, le=2048, description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )


class Config(BaseModel):
    """Main configuration class using Pydantic for unified validation."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    peft: PEFTConfig = Field(default_factory=PEFTConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    cleanup: CleanupPolicy = Field(default_factory=CleanupPolicy)

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def validate_cross_config(self):
        """Validate relationships between different configuration sections."""
        if self.training.epochs == 1 and self.training.eval_steps > 500:
            raise ValueError(
                f"eval_steps ({self.training.eval_steps}) is too high for single epoch training. "
                f"Consider reducing to < 500 or increasing epochs."
            )

        if self.peft.lora_alpha > self.peft.r * 4:
            raise ValueError(
                f"LoRA alpha ({self.peft.lora_alpha}) is very high compared to rank ({self.peft.r}). "
                f"Consider alpha <= {self.peft.r * 4} or increasing rank."
            )

        effective_batch = (
            self.training.batch_size * self.training.gradient_accumulation_steps
        )
        if effective_batch > 128:
            raise ValueError(
                f"Effective batch size ({effective_batch}) is very large. "
                f"This may cause memory issues or poor training dynamics."
            )

        return self


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_config", node=HydraConfig)

cs.store(group="model", name="small", node=HydraModelConfig())
cs.store(
    group="model",
    name="llama_3b",
    node=HydraModelConfig(model_id="meta-llama/Llama-3.2-3B-Instruct"),
)

cs.store(group="dataset", name="alpaca", node=HydraDatasetConfig())
cs.store(
    group="dataset", name="alpaca_full", node=HydraDatasetConfig(dataset_split="train")
)

cs.store(group="training", name="quick", node=HydraTrainingConfig(epochs=1, lr=2e-5))
cs.store(group="training", name="standard", node=HydraTrainingConfig(epochs=3, lr=1e-5))

cs.store(group="peft", name="lora_16", node=HydraPEFTConfig(r=16, lora_alpha=32))
cs.store(group="peft", name="lora_32", node=HydraPEFTConfig(r=32, lora_alpha=64))

# Inference configurations
cs.store(group="inference", name="base", node=HydraInferenceConfig())
cs.store(
    group="inference",
    name="latest",
    node=HydraInferenceConfig(auto_find_latest_run=True),
)
cs.store(
    group="inference",
    name="manual",
    node=HydraInferenceConfig(auto_find_latest_run=False),
)
