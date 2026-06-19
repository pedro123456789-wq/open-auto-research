from __future__ import annotations

from abc import ABC, abstractmethod

from utils.llm_utils import OpenRouterLLM


class ImprovementAgent(ABC):
    @abstractmethod
    async def propose_improvement(
        self,
        llm: OpenRouterLLM,
        parent_code: str,
        parent_accuracy: float,
        metadata: dict,
    ) -> tuple[str, str, str]:
        """Propose an improved pipeline given the parent source and eval metadata.

        Args:
            llm:             Proposer LLM client.
            parent_code:     Full source of the parent pipeline.py.
            parent_accuracy: Parent's accuracy score in [0, 1].
            metadata:        Eval metadata from the parent's last evaluation
                             (e.g. correct, total, wrong_samples, early_stopped).

        Returns:
            (new_code, reasoning, novelty) — all strings.

        Raise on proposer failure or unparseable response.
        """
