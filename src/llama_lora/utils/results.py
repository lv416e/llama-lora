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
    """Session-based inference logging system following MLOps industry standards.
    
    Implements session accumulation pattern as used by MLflow and Langfuse for
    production observability, enabling session context tracking, user journey
    analysis, and continuous monitoring.
    """

    def __init__(self, output_config: OutputConfig, session_name: Optional[str] = None):
        self.output_config = output_config
        self.results: List[InferenceResult] = []
        
        # Generate unique session ID: timestamp + UUID for production traceability
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4()).replace('-', '')[:8]
        self.session_id = f"{timestamp}_{short_uuid}"
        self.session_name = session_name or "inference_session"
        
        # Create session-based directory structure
        self.inference_dir = Path(output_config.metadata_dir) / "inference"
        self.sessions_dir = self.inference_dir / "sessions"
        self.session_dir = self.sessions_dir / self.session_id
        
        # Ensure directory structure exists
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Session metadata
        self.session_start_time = datetime.now().isoformat()

    def log_inference(
        self,
        prompt: str,
        response: str,
        model_config: Dict[str, Any],
        generation_params: Dict[str, Any],
        execution_time: float,
        model_type: str = "fine_tuned",
    ) -> InferenceResult:
        """Log single inference result to current session."""
        result = InferenceResult(
            prompt=prompt,
            response=response,
            model_config=model_config,
            generation_params=generation_params,
            execution_time=execution_time,
            model_type=model_type,
        )

        self.results.append(result)
        
        # Save individual trace immediately for observability
        trace_file = self.session_dir / f"trace_{len(self.results):03d}.json"
        trace_data = {
            "session_id": self.session_id,
            "trace_number": len(self.results),
            "session_context": {
                "experiment_name": self.output_config.experiment_name,
                "run_id": self.output_config.run_id,
                "model_path": self.output_config.adapter_dir,
            },
            "inference": result.to_dict(),
        }
        
        try:
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2)
        except (OSError, IOError) as e:
            # Non-critical error, continue execution
            print(f"Warning: Failed to save trace {len(self.results)}: {str(e)}")
        
        return result

    def save_session(self) -> str:
        """Save session summary following industry session management patterns."""
        session_file = self.session_dir / "session_summary.json"

        session_data = {
            "session_metadata": {
                "session_id": self.session_id,
                "session_name": self.session_name,
                "session_start": self.session_start_time,
                "session_end": datetime.now().isoformat(),
                "total_inferences": len(self.results),
            },
            "experiment_context": {
                "experiment_name": self.output_config.experiment_name,
                "run_id": self.output_config.run_id,
                "adapter_dir": self.output_config.adapter_dir,
            },
            "session_analytics": {
                "total_execution_time": sum(r.execution_time for r in self.results),
                "average_execution_time": (
                    sum(r.execution_time for r in self.results) / len(self.results)
                    if self.results else 0
                ),
                "inference_count": len(self.results),
            },
            "trace_files": [
                f"trace_{i+1:03d}.json" for i in range(len(self.results))
            ]
        }

        try:
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            return str(session_file)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to save session summary: {str(e)}") from e

    def export_comparison_data(self) -> str:
        """Export legacy comparison format for backward compatibility."""
        # Create legacy comparison file in main inference directory
        comparison_file = self.inference_dir / f"comparison_{self.session_id}.json"

        comparison_data = {
            "experiment_comparison": {
                "experiment_name": self.output_config.experiment_name,
                "run_id": self.output_config.run_id,
                "model_path": self.output_config.adapter_dir,
                "total_inferences": len(self.results),
                "average_execution_time": (
                    sum(r.execution_time for r in self.results) / len(self.results)
                    if self.results else 0
                ),
                "session_id": self.session_id,  # Link to session
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

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information for logging and debugging."""
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "session_dir": str(self.session_dir),
            "inference_count": len(self.results),
            "session_start": self.session_start_time,
        }
