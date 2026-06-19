"""In-memory archive of evolved pipeline nodes.

Each Node represents one candidate pipeline. The Archive implements DGM-style
parent selection:

    score_weight   = sigmoid(lambda * (accuracy - alpha0))   # exploitation
    novelty_weight = 1 / (1 + num_children)                  # exploration
    p_i = score_weight_i * novelty_weight_i / sum_j(...)

Reference: Darwin Gödel Machine, Zhang et al. 2025 (Appendix C.2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    """One candidate pipeline version."""

    node_id: str
    parent_id: Optional[str]        # None for the root node
    round: int                      # 0 = root

    # LLM-provided descriptions of what changed
    reasoning: str = ""
    novelty: str = ""

    # Compilation
    compiles: bool = False
    compile_error: Optional[str] = None

    # Benchmark result
    accuracy: float = 0.0

    # Run-specific eval details (correct/total/wrong_samples/etc.)
    metadata: dict = field(default_factory=dict)

    # Housekeeping
    timestamp: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "round": self.round,
            "reasoning": self.reasoning,
            "novelty": self.novelty,
            "compiles": self.compiles,
            "compile_error": self.compile_error,
            "accuracy": self.accuracy,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        node = cls(
            node_id=d["node_id"],
            parent_id=d.get("parent_id"),
            round=d.get("round", 0),
        )
        node.reasoning = d.get("reasoning", "")
        node.novelty = d.get("novelty", "")
        node.compiles = d.get("compiles", False)
        node.compile_error = d.get("compile_error")
        node.accuracy = d.get("accuracy", 0.0)
        # Support old archives that stored fields at top level
        node.metadata = d.get("metadata") or {
            "correct": d.get("correct", 0),
            "total": d.get("total", 0),
            "wrong_samples": d.get("wrong_samples", []),
            "early_stopped": False,
        }
        node.timestamp = d.get("timestamp", "")
        node.model = d.get("model", "")
        return node


class Archive:
    """Flat list of all evaluated nodes with DGM-style parent selection."""

    def __init__(
        self,
        dgm_lambda: float = 10.0,
        dgm_alpha0: float = 0.5,
        min_parent_accuracy: float = 0.2,
    ) -> None:
        self._nodes: list[Node] = []
        self._lambda = dgm_lambda
        self._alpha0 = dgm_alpha0
        self._min_parent_accuracy = min_parent_accuracy

    def add(self, node: Node) -> None:
        self._nodes.append(node)

    def get(self, node_id: str) -> Optional[Node]:
        for n in self._nodes:
            if n.node_id == node_id:
                return n
        return None

    def all_nodes(self) -> list[Node]:
        return list(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def children_count(self, node_id: str) -> int:
        return sum(1 for n in self._nodes if n.parent_id == node_id)

    def _sigmoid(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-self._lambda * (score - self._alpha0)))

    def select_parent(self) -> Node:
        """Sample one parent using DGM's combined score × novelty weight."""
        if len(self._nodes) == 1:
            return self._nodes[0]

        eligible = [n for n in self._nodes if n.accuracy >= self._min_parent_accuracy]
        if not eligible:
            best = self.best()
            return best if best is not None else self._nodes[0]
        if len(eligible) == 1:
            return eligible[0]

        weights = [
            self._sigmoid(n.accuracy) * (1.0 / (1.0 + self.children_count(n.node_id)))
            for n in eligible
        ]
        total = sum(weights)
        if total == 0:
            return random.choice(eligible)
        return random.choices(eligible, weights=[w / total for w in weights], k=1)[0]

    def best(self) -> Optional[Node]:
        return max(self._nodes, key=lambda n: n.accuracy) if self._nodes else None

    def summary_rows(self) -> list[dict]:
        return sorted(
            [
                {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "round": n.round,
                    "accuracy": n.accuracy,
                    "children": self.children_count(n.node_id),
                }
                for n in self._nodes
            ],
            key=lambda r: -r["accuracy"],
        )
