"""Integration tests for unified output configuration system."""

import tempfile
from pathlib import Path
from datetime import datetime

from config.schema import OutputConfig, HydraOutputConfig


class TestOutputConfigIntegration:
    """Test cases for OutputConfig integration with path management."""

    def test_output_config_automatic_path_generation(self):
        """Test that OutputConfig generates structured paths automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = OutputConfig(
                base_output_dir=temp_dir, experiment_name="test_experiment"
            )

            # Check that paths are auto-generated
            assert config.run_id != ""
            assert config.adapter_dir != ""
            assert config.tokenizer_dir != ""
            assert config.merged_dir != ""
            assert config.log_dir != ""
            assert config.metadata_dir != ""

            # Check structured format
            expected_base = (
                f"{temp_dir}/experiments/test_experiment/runs/{config.run_id}"
            )
            assert config.adapter_dir == f"{expected_base}/artifacts/adapter"
            assert config.tokenizer_dir == f"{expected_base}/artifacts/tokenizer"
            assert config.merged_dir == f"{expected_base}/artifacts/merged"
            assert config.log_dir == f"{expected_base}/logs"
            assert config.metadata_dir == f"{expected_base}/metadata"

    def test_output_config_run_id_format(self):
        """Test that run_id follows expected timestamp format."""
        config = OutputConfig(experiment_name="test")

        # Should be in format YYYYMMDD_HHMMSS
        assert len(config.run_id) == 15
        assert config.run_id[8] == "_"

        # Should parse as valid datetime
        datetime.strptime(config.run_id, "%Y%m%d_%H%M%S")

    def test_output_config_manual_paths_ignored(self):
        """Test that manual path specifications are ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = OutputConfig(
                base_output_dir=temp_dir,
                experiment_name="test",
                adapter_dir="/manual/path",  # Should be ignored
                run_id="manual_id",  # Should be ignored
            )

            # Manual paths should be overridden
            assert config.adapter_dir != "/manual/path"
            assert config.run_id != "manual_id"

            # Should use structured format
            assert f"{temp_dir}/experiments/test/runs/" in config.adapter_dir

    def test_hydra_to_pydantic_config_consistency(self):
        """Test that HydraOutputConfig converts to OutputConfig correctly."""
        hydra_config = HydraOutputConfig(
            base_output_dir="./test_outputs", experiment_name="hydra_test"
        )

        # Convert to pydantic config (mimics to_pydantic_config behavior)
        pydantic_config = OutputConfig(
            base_output_dir=hydra_config.base_output_dir,
            experiment_name=hydra_config.experiment_name,
            run_id="",  # Should be auto-generated
            adapter_dir="",  # Should be auto-generated
            tokenizer_dir="",  # Should be auto-generated
            merged_dir="",  # Should be auto-generated
            log_dir="",  # Should be auto-generated
            metadata_dir="",  # Should be auto-generated
        )

        # Both should have same base configuration
        assert pydantic_config.base_output_dir == hydra_config.base_output_dir
        assert pydantic_config.experiment_name == hydra_config.experiment_name

        # Pydantic config should have auto-generated paths
        assert pydantic_config.run_id != ""
        assert "experiments/hydra_test/runs" in pydantic_config.adapter_dir

    def test_output_config_directory_creation_readiness(self):
        """Test that generated paths are valid for directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = OutputConfig(base_output_dir=temp_dir, experiment_name="path_test")

            # All paths should be absolute or relative to a valid base
            paths_to_test = [
                config.adapter_dir,
                config.tokenizer_dir,
                config.merged_dir,
                config.log_dir,
                config.metadata_dir,
            ]

            for path in paths_to_test:
                # Should not be empty
                assert path.strip() != ""

                # Should contain the experiment structure
                assert "experiments/path_test/runs" in path

                # Should be creatable (test with pathlib)
                path_obj = Path(path)
                assert path_obj.parts  # Should have valid path components

    def test_multiple_configs_unique_run_ids(self):
        """Test that multiple configs get unique run IDs."""
        config1 = OutputConfig(experiment_name="test1")
        config2 = OutputConfig(experiment_name="test2")

        assert config1.run_id != config2.run_id

    def test_output_config_experiment_name_sanitization(self):
        """Test that experiment names work in file paths."""
        # Test with various experiment name formats
        test_names = ["simple", "with-dashes", "with_underscores", "CamelCase"]

        for name in test_names:
            config = OutputConfig(experiment_name=name)

            # Should not cause path issues
            assert name in config.adapter_dir
            assert "/" in config.adapter_dir  # Should have proper path separators

            # Path should be valid
            path_obj = Path(config.adapter_dir)
            assert path_obj.parts
