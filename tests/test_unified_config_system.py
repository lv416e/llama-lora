"""Test script to verify the unified configuration system works end-to-end."""

import tempfile
import os
import json

from config.schema import HydraOutputConfig, OutputConfig, save_experiment_metadata
from src.llama_lora.utils.storage import PathManager


def test_unified_config_workflow():
    """Test the complete unified configuration workflow."""
    print("🧪 Testing Unified Configuration System")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")

        # Step 1: Create HydraOutputConfig (simulates Hydra configuration)
        print("\n1️⃣ Creating HydraOutputConfig...")
        hydra_config = HydraOutputConfig(
            base_output_dir=temp_dir, experiment_name="unified_test"
        )
        print(
            f"   ✅ Hydra config created with experiment: {hydra_config.experiment_name}"
        )

        # Step 2: Convert to Pydantic OutputConfig (simulates to_pydantic_config)
        print("\n2️⃣ Converting to Pydantic OutputConfig...")
        pydantic_config = OutputConfig(
            base_output_dir=hydra_config.base_output_dir,
            experiment_name=hydra_config.experiment_name,
            run_id="",  # Auto-generated
            adapter_dir="",  # Auto-generated
            tokenizer_dir="",  # Auto-generated
            merged_dir="",  # Auto-generated
            log_dir="",  # Auto-generated
            metadata_dir="",  # Auto-generated
        )
        print(f"   ✅ Pydantic config created with run_id: {pydantic_config.run_id}")

        # Step 3: Verify structured paths
        print("\n3️⃣ Verifying structured paths...")
        expected_base = (
            f"{temp_dir}/experiments/unified_test/runs/{pydantic_config.run_id}"
        )

        paths_to_check = {
            "adapter": pydantic_config.adapter_dir,
            "tokenizer": pydantic_config.tokenizer_dir,
            "merged": pydantic_config.merged_dir,
            "logs": pydantic_config.log_dir,
            "metadata": pydantic_config.metadata_dir,
        }

        for name, path in paths_to_check.items():
            assert expected_base in path, f"{name} path doesn't contain expected base"
            print(f"   ✅ {name:10}: {path}")

        # Step 4: Test PathManager integration
        print("\n4️⃣ Testing PathManager integration...")
        for name, path in paths_to_check.items():
            try:
                PathManager.ensure_directory(path)
                assert os.path.isdir(path), f"Directory {path} was not created"
                print(f"   ✅ {name:10}: Directory created successfully")
            except Exception as e:
                print(f"   ❌ {name:10}: Failed to create directory - {e}")
                raise

        # Step 5: Test metadata saving
        print("\n5️⃣ Testing metadata saving...")
        test_metadata = {
            "test_config": {"lr": 2e-5, "batch_size": 1},
            "test_timestamp": "2024-test",
        }

        try:
            metadata_file = save_experiment_metadata(test_metadata, pydantic_config)
            assert os.path.isfile(metadata_file), "Metadata file was not created"

            # Verify metadata content
            with open(metadata_file, "r") as f:
                saved_metadata = json.load(f)

            assert saved_metadata["experiment_name"] == "unified_test"
            assert saved_metadata["run_id"] == pydantic_config.run_id
            assert "config" in saved_metadata
            print(f"   ✅ Metadata saved to: {metadata_file}")

        except Exception as e:
            print(f"   ❌ Metadata saving failed: {e}")
            raise

        # Step 6: Verify directory structure
        print("\n6️⃣ Verifying complete directory structure...")
        expected_structure = {
            f"{temp_dir}/experiments": "experiments directory",
            f"{temp_dir}/experiments/unified_test": "experiment directory",
            f"{temp_dir}/experiments/unified_test/runs": "runs directory",
            f"{temp_dir}/experiments/unified_test/runs/{pydantic_config.run_id}": "run directory",
            f"{temp_dir}/experiments/unified_test/runs/{pydantic_config.run_id}/artifacts": "artifacts directory",
            f"{temp_dir}/experiments/unified_test/runs/{pydantic_config.run_id}/logs": "logs directory",
            f"{temp_dir}/experiments/unified_test/runs/{pydantic_config.run_id}/metadata": "metadata directory",
        }

        for path, description in expected_structure.items():
            if os.path.isdir(path):
                print(f"   ✅ {description}: {path}")
            else:
                print(f"   ❌ {description}: Missing - {path}")
                raise AssertionError(f"Expected directory missing: {path}")

        print(
            "\n🎉 All tests passed! Unified configuration system is working correctly."
        )

        # Step 7: Display final structure
        print("\n📋 Final Directory Structure:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")


def test_config_validation_warnings():
    """Test that configuration validation warns about manual paths."""
    print("\n🔍 Testing Configuration Validation Warnings")
    print("=" * 50)

    from src.llama_lora.validate import ConfigValidator
    from omegaconf import DictConfig

    # Create config with manual paths (should trigger warnings)
    config_dict = {
        "base_output_dir": "./outputs",
        "experiment_name": "test",
        "adapter_dir": "./manual/adapter",  # Manual path
        "tokenizer_dir": "./manual/tokenizer",  # Manual path
    }

    validator = ConfigValidator()
    result = validator.validate_output_config(DictConfig(config_dict))

    print(f"Validation result: {result}")
    print(f"Warnings: {validator.warnings}")
    print(f"Errors: {validator.errors}")

    # Should have warnings about manual paths
    assert any(
        "Manual path specifications detected" in warning
        for warning in validator.warnings
    )
    print("✅ Configuration validation correctly warns about manual paths")


if __name__ == "__main__":
    test_unified_config_workflow()
    test_config_validation_warnings()
    print("\n🚀 All integration tests completed successfully!")
