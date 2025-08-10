# Hydra + Pydantic Configuration Management

This document explains how to use the new Hydra + Pydantic-based configuration management system.

## Basic Usage

### 1. Run training with default settings

```bash
python -m llama_lora.train
```

### 2. Override settings with command line arguments

```bash
python -m llama_lora.train training.lr=1e-5

python -m llama_lora.train training.lr=3e-5 training.epochs=3 model.seq_len=2048

python -m llama_lora.train training.batch_size=2 training.gradient_accumulation_steps=4
```

### 3. Use experiment configurations

```bash
python -m llama_lora.train +experiment=quick_test

python -m llama_lora.train +experiment=full_training

python -m llama_lora.train +experiment=dora_experiment
```

### 4. Combine configurations

```bash
python -m llama_lora.train +experiment=quick_test training.lr=5e-5

python -m llama_lora.train +experiment=quick_test model.model_id="meta-llama/Llama-3.2-3B-Instruct"
```

## Configuration Validation

```bash
python -m llama_lora.validate
python -m llama_lora.validate +experiment=full_training
```

## Configuration File Structure

```
config/
├── config.yaml              
├── schema.py     
└── experiment/              
    ├── quick_test.yaml      
    ├── full_training.yaml   
    └── dora_experiment.yaml 
```

## Configuration Parameters

### Model Configuration
- `model.model_id`: Base model identifier
- `model.seq_len`: Maximum sequence length (128-8192)
- `model.use_dora`: Enable DoRA

### Training Configuration  
- `training.lr`: Learning rate (0-1.0)
- `training.batch_size`: Batch size (1-128)
- `training.epochs`: Number of epochs (1-100)
- `training.gradient_accumulation_steps`: Gradient accumulation steps
- `training.eval_steps`: Evaluation interval
- `training.seed`: Random seed

### Dataset Configuration
- `dataset.dataset_id`: Dataset identifier
- `dataset.dataset_split`: Dataset split specification
- `dataset.val_ratio`: Validation set ratio (0.0-0.5)

### PEFT Configuration
- `peft.r`: LoRA rank (1-1024)
- `peft.lora_alpha`: LoRA alpha value
- `peft.lora_dropout`: LoRA dropout rate (0.0-1.0)
- `peft.target_modules`: Target modules list

### Output Configuration
- `output.base_output_dir`: Base output directory

### Logging Configuration
- `logging.report_to`: Logging backend (tensorboard/wandb/none)
- `logging.project_name`: Project name

## Creating New Experiment Configurations

1. Create `config/experiment/my_experiment.yaml`
2. Write configuration:

```yaml
# @package _global_
defaults:
  - override /training: standard
  - override /peft: lora_32

training:
  epochs: 5
  lr: 2e-5
  
model:
  seq_len: 2048
  
logging:
  project_name: "my-custom-experiment"
```

3. Run:
```bash
python -m llama_lora.train +experiment=my_experiment
```

## Type Safety and Validation

- All configuration values are type-checked and validated by Pydantic
- Invalid values are detected before execution
- Configuration combination issues trigger warnings

