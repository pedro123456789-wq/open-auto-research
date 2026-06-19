from __future__ import annotations

from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, pipeline_path: str, output_dir: str) -> tuple[float, dict]:
        """Load and run the pipeline at pipeline_path.

        Returns:
            (accuracy, metadata) where accuracy is a float in [0, 1] and
            metadata is a JSON-serializable dict with any run-specific details
            (e.g. correct, total, wrong_samples, early_stopped).

        Raise on hard failure (syntax error, missing Baseline subclass, crash).
        """
