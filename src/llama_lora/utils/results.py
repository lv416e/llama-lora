"""Inference results logging and persistence utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.schema import OutputConfig


class InferenceResult:
    """Structured inference result with metadata."""

    def __init__(
        self,
        prompt: str,
        response: str,
        model_config: Dict[str, Any],
        generation_params: Dict[str, Any],
        execution_time: float,
        model_type: str = "unknown",
    ):
        self.prompt = prompt
        self.response = response
        self.model_config = model_config
        self.generation_params = generation_params
        self.execution_time = execution_time
        self.model_type = model_type
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "model_type": self.model_type,
            "prompt": self.prompt,
            "response": self.response,
            "execution_time_seconds": self.execution_time,
            "model_config": self.model_config,
            "generation_params": self.generation_params,
        }


class InferenceLogger:
    """Manages inference result logging and batch operations."""

    def __init__(self, output_config: OutputConfig, session_name: Optional[str] = None):
        self.output_config = output_config
        self.session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[InferenceResult] = []

        self.inference_dir = Path(output_config.metadata_dir) / "inference"
        self.inference_dir.mkdir(parents=True, exist_ok=True)

    def log_inference(
        self,
        prompt: str,
        response: str,
        model_config: Dict[str, Any],
        generation_params: Dict[str, Any],
        execution_time: float,
        model_type: str = "fine_tuned",
    ) -> InferenceResult:
        """Log single inference result."""
        result = InferenceResult(
            prompt=prompt,
            response=response,
            model_config=model_config,
            generation_params=generation_params,
            execution_time=execution_time,
            model_type=model_type,
        )

        self.results.append(result)
        return result

    def save_session(self) -> str:
        """Save all inference results from current session."""
        session_file = self.inference_dir / f"session_{self.session_name}.json"

        session_data = {
            "session_name": self.session_name,
            "total_inferences": len(self.results),
            "session_start": self.results[0].timestamp if self.results else None,
            "session_end": datetime.now().isoformat(),
            "output_config": {
                "experiment_name": self.output_config.experiment_name,
                "run_id": self.output_config.run_id,
                "adapter_dir": self.output_config.adapter_dir,
            },
            "results": [result.to_dict() for result in self.results],
        }

        try:
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            return str(session_file)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to save inference session: {str(e)}") from e

    def export_comparison_data(self) -> str:
        """Export data optimized for model comparison."""
        comparison_file = self.inference_dir / f"comparison_{self.session_name}.json"

        comparison_data = {
            "experiment_comparison": {
                "experiment_name": self.output_config.experiment_name,
                "run_id": self.output_config.run_id,
                "model_path": self.output_config.adapter_dir,
                "total_inferences": len(self.results),
                "average_execution_time": (
                    sum(r.execution_time for r in self.results) / len(self.results)
                    if self.results
                    else 0
                ),
            },
            "prompt_response_pairs": [
                {
                    "prompt": result.prompt,
                    "response": result.response,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp,
                }
                for result in self.results
            ],
        }

        try:
            with open(comparison_file, "w") as f:
                json.dump(comparison_data, f, indent=2)
            return str(comparison_file)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to export comparison data: {str(e)}") from e
