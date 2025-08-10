"""Structured configuration classes using Pydantic for LLaMA-LoRA fine-tuning.

This module defines type-safe configuration classes using Pydantic BaseModel
for validation and Hydra for flexible configuration management. All settings
are organized into logical groups for better maintainability and validation.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass, field


# Hydra互換性のためのdataclassベース設定
@dataclass
class HydraModelConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_dora: bool = False
    seq_len: int = 512


@dataclass
class HydraDatasetConfig:
    dataset_id: str = "tatsu-lab/alpaca"
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
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            # LLaMA 系モデル向けの一般的なLoRA対象
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
    adapter_dir: str = "./outputs/adapter"
    tokenizer_dir: str = "./outputs/tokenizer"
    merged_dir: str = "./outputs/merged"
    log_dir: str = "./outputs/runs"


@dataclass
class HydraLoggingConfig:
    report_to: str = "none"
    project_name: Optional[str] = None


@dataclass
class HydraConfig:
    """Hydra互換のmain configuration class (dataclass based)."""

    model: HydraModelConfig = field(default_factory=HydraModelConfig)
    dataset: HydraDatasetConfig = field(default_factory=HydraDatasetConfig)
    training: HydraTrainingConfig = field(default_factory=HydraTrainingConfig)
    peft: HydraPEFTConfig = field(default_factory=HydraPEFTConfig)
    output: HydraOutputConfig = field(default_factory=HydraOutputConfig)
    logging: HydraLoggingConfig = field(default_factory=HydraLoggingConfig)

    def __post_init__(self):
        # Initialize target_modules for PEFT if None
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
                adapter_dir=self.output.adapter_dir,
                tokenizer_dir=self.output.tokenizer_dir,
                merged_dir=self.output.merged_dir,
                log_dir=self.output.log_dir,
            ),
            logging=LoggingConfig(
                report_to=self.logging.report_to,
                project_name=self.logging.project_name,
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
        default="tatsu-lab/alpaca", description="Dataset identifier"
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
    adapter_dir: str = Field(
        default="./outputs/adapter", description="Adapter output directory"
    )
    tokenizer_dir: str = Field(
        default="./outputs/tokenizer", description="Tokenizer output directory"
    )
    merged_dir: str = Field(
        default="./outputs/merged", description="Merged model output directory"
    )
    log_dir: str = Field(default="./outputs/runs", description="Logs output directory")


class LoggingConfig(BaseModel):
    """Logging configuration with validation."""

    report_to: Literal["none", "wandb", "tensorboard"] = Field(
        default="none", description="Logging backend"
    )
    project_name: Optional[str] = Field(
        default=None, description="Experiment or project name for logging"
    )


class Config(BaseModel):
    """Main configuration class using Pydantic for unified validation."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    peft: PEFTConfig = Field(default_factory=PEFTConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def validate_cross_config(self):
        """Validate relationships between different configuration sections."""
        # Check if eval_steps makes sense for the number of epochs
        if self.training.epochs == 1 and self.training.eval_steps > 500:
            raise ValueError(
                f"eval_steps ({self.training.eval_steps}) is too high for single epoch training. "
                f"Consider reducing to < 500 or increasing epochs."
            )

        # Check LoRA parameter relationships
        if self.peft.lora_alpha > self.peft.r * 4:
            raise ValueError(
                f"LoRA alpha ({self.peft.lora_alpha}) is very high compared to rank ({self.peft.r}). "
                f"Consider alpha <= {self.peft.r * 4} or increasing rank."
            )

        # Validate effective batch size
        effective_batch = (
            self.training.batch_size * self.training.gradient_accumulation_steps
        )
        if effective_batch > 128:
            raise ValueError(
                f"Effective batch size ({effective_batch}) is very large. "
                f"This may cause memory issues or poor training dynamics."
            )

        return self


# Hydra ConfigStoreに登録
cs = ConfigStore.instance()
cs.store(name="base_config", node=HydraConfig)

# Register individual config groups for different configurations
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
