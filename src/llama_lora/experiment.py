"""Experiment runner script for multiple configurations.

This script demonstrates how to run multiple experiments with different
configurations programmatically using Hydra's compose API.
"""

import subprocess
import sys
from pathlib import Path
from typing import List


def run_experiment(config_overrides: List[str], experiment_name: str) -> bool:
    """Run a single experiment with given configuration overrides.

    Args:
        config_overrides: List of configuration override strings.
        experiment_name: Name for this experiment run.

    Returns:
        bool: True if experiment completed successfully.
    """
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Configuration: {' '.join(config_overrides)}")
    print(f"{'=' * 60}")

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "llama_lora.train",
        f"hydra.job.name={experiment_name}",
    ] + config_overrides

    try:
        subprocess.run(cmd, check=True, cwd=Path.cwd())
        print(f"✅ Experiment '{experiment_name}' completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment '{experiment_name}' failed with exit code {e.returncode}")
        return False


def main():
    """Run a series of experiments with different configurations."""

    experiments = [
        {
            "name": "quick_baseline",
            "config": ["+experiment=quick_test", "training.lr=2e-5"],
        },
        {
            "name": "quick_high_lr",
            "config": ["+experiment=quick_test", "training.lr=5e-5"],
        },
        {
            "name": "lora_vs_dora_16",
            "config": ["+experiment=quick_test", "model.use_dora=false", "peft.r=16"],
        },
        {
            "name": "lora_vs_dora_32",
            "config": [
                "+experiment=quick_test",
                "model.use_dora=false",
                "peft.r=32",
                "peft.lora_alpha=64",
            ],
        },
    ]

    print("Starting experiment suite...")
    print(f"Will run {len(experiments)} experiments")

    successful = 0
    failed = 0

    for exp in experiments:
        success = run_experiment(exp["config"], exp["name"])
        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print("Experiment Suite Results:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(experiments)}")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
