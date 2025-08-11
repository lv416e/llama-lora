"""Test suite for inference logging functionality."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock

from src.llama_lora.utils.results import InferenceLogger, InferenceResult


class TestInferenceResult:
    """Test InferenceResult data structure."""

    def test_inference_result_creation(self):
        """Test InferenceResult object creation."""
        result = InferenceResult(
            prompt="Test prompt",
            response="Test response",
            model_config={"model_id": "test"},
            generation_params={"max_tokens": 50},
            execution_time=1.5,
            model_type="baseline",
        )

        assert result.prompt == "Test prompt"
        assert result.response == "Test response"
        assert result.model_type == "baseline"
        assert result.execution_time == 1.5
        assert result.timestamp is not None

    def test_inference_result_to_dict(self):
        """Test InferenceResult serialization."""
        result = InferenceResult(
            prompt="Test prompt",
            response="Test response",
            model_config={"model_id": "test"},
            generation_params={"max_tokens": 50},
            execution_time=1.5,
            model_type="baseline",
        )

        data = result.to_dict()

        assert data["prompt"] == "Test prompt"
        assert data["response"] == "Test response"
        assert data["model_type"] == "baseline"
        assert data["execution_time_seconds"] == 1.5
        assert "timestamp" in data
        assert "model_config" in data
        assert "generation_params" in data


class TestInferenceLogger:
    """Test InferenceLogger functionality."""

    @pytest.fixture
    def mock_output_config(self):
        """Create mock output config."""
        config = Mock()
        config.experiment_name = "test_experiment"
        config.run_id = "test_run"
        config.adapter_dir = "/test/adapter"

        # Create temporary metadata directory
        temp_dir = Path(tempfile.mkdtemp())
        config.metadata_dir = str(temp_dir)

        return config

    @pytest.fixture
    def inference_logger(self, mock_output_config):
        """Create InferenceLogger instance."""
        return InferenceLogger(mock_output_config, "test_session")

    def test_inference_logger_creation(self, mock_output_config):
        """Test InferenceLogger creation."""
        logger = InferenceLogger(mock_output_config, "test_session")

        assert logger.session_name == "test_session"
        assert logger.output_config == mock_output_config
        assert logger.results == []
        assert logger.inference_dir.exists()

    def test_log_inference(self, inference_logger):
        """Test logging single inference."""
        result = inference_logger.log_inference(
            prompt="Test prompt",
            response="Test response",
            model_config={"model_id": "test"},
            generation_params={"max_tokens": 50},
            execution_time=1.5,
            model_type="baseline",
        )

        assert isinstance(result, InferenceResult)
        assert len(inference_logger.results) == 1
        assert inference_logger.results[0] == result

    def test_save_session(self, inference_logger):
        """Test saving inference session."""
        # Add some test results
        inference_logger.log_inference(
            prompt="Test prompt 1",
            response="Test response 1",
            model_config={"model_id": "test"},
            generation_params={"max_tokens": 50},
            execution_time=1.5,
            model_type="baseline",
        )

        inference_logger.log_inference(
            prompt="Test prompt 2",
            response="Test response 2",
            model_config={"model_id": "test"},
            generation_params={"max_tokens": 100},
            execution_time=2.0,
            model_type="fine_tuned",
        )

        session_file = inference_logger.save_session()

        assert Path(session_file).exists()

        # Verify file content
        with open(session_file) as f:
            data = json.load(f)

        assert data["session_name"] == "test_session"
        assert data["total_inferences"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["prompt"] == "Test prompt 1"
        assert data["results"][1]["prompt"] == "Test prompt 2"

    def test_export_comparison_data(self, inference_logger):
        """Test exporting comparison data."""
        # Add test results
        inference_logger.log_inference(
            prompt="Test prompt",
            response="Test response",
            model_config={"model_id": "test"},
            generation_params={"max_tokens": 50},
            execution_time=1.5,
            model_type="baseline",
        )

        comparison_file = inference_logger.export_comparison_data()

        assert Path(comparison_file).exists()

        # Verify file content
        with open(comparison_file) as f:
            data = json.load(f)

        assert "experiment_comparison" in data
        assert "prompt_response_pairs" in data
        assert data["experiment_comparison"]["total_inferences"] == 1
        assert data["experiment_comparison"]["average_execution_time"] == 1.5
        assert len(data["prompt_response_pairs"]) == 1

    def test_empty_session_handling(self, inference_logger):
        """Test handling empty session."""
        session_file = inference_logger.save_session()

        with open(session_file) as f:
            data = json.load(f)

        assert data["total_inferences"] == 0
        assert data["session_start"] is None
        assert data["results"] == []

        comparison_file = inference_logger.export_comparison_data()

        with open(comparison_file) as f:
            data = json.load(f)

        assert data["experiment_comparison"]["average_execution_time"] == 0
        assert data["prompt_response_pairs"] == []
