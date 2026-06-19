"""
Abstract base class for all LOCOMO memory implementations.

Every concrete implementation must implement two async methods:
  ingest_conversation — store a raw LOCOMO conversation in whatever memory
                        backend the baseline uses.
  process_question    — retrieve from memory and return an answer string.

The class manages its own LLM instance so callers only need to pass a model
name, not a pre-constructed client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import logging

from utils.llm_utils import OpenRouterLLM

logger = logging.getLogger(__name__)


class Baseline(ABC):
    """Abstract base for LOCOMO memory baselines.

    Subclasses must:
      1. Define a class-level `name` attribute (used in logging and results).
      2. Implement `ingest_conversation` and `process_question` as async methods.

    Both methods are deliberately async so implementations can freely use
    async I/O (embedding APIs, LLM calls, file writes, etc.) without blocking.
    """

    # Subclasses should override this with a short descriptive name.
    name: str = "baseline"

    def __init__(self, model: str | None = None, top_k: int = 10) -> None:
        self.llm = OpenRouterLLM(model=model)
        self.top_k = top_k
        logger.debug("Initialised %s (model=%s, top_k=%d)", self, self.llm.model, top_k)

    # ── interface ──────────────────────────────────────────────────────────────
    @abstractmethod
    async def ingest_conversation(
        self,
        conv_idx: int,
        entry: dict,
        output_dir: str,
    ) -> tuple[bool, str, int]:
        """Ingest one LOCOMO conversation into the baseline's memory store.

        Args:
            conv_idx:   Index of this conversation in the dataset.
            entry:      Raw LOCOMO entry dict (keys: conversation, qa, …).
            output_dir: Root directory for any on-disk artefacts.

        Returns:
            (success, user_id, items_stored)
              success      — False if any data was lost during ingestion.
              user_id      — Opaque string that identifies this conversation
                             in the store (passed back for debugging).
              items_stored — Number of chunks / lines written.
        """

    @abstractmethod
    async def process_question(
        self,
        question: str,
        conv_idx: int,
    ) -> str:
        """Retrieve from memory and return an answer string.

        `ingest_conversation` must have been called for `conv_idx` first.

        Args:
            question:  The plain-text question to answer.
            conv_idx:  Identifies which ingested conversation to query.

        Returns:
            A short answer string.  Evaluation is the caller's responsibility.
        """

    # ── helpers available to subclasses ───────────────────────────────────────
    def _not_ingested(self, conv_idx: int) -> RuntimeError:
        """Return a clear error for use before ingest has been called."""
        return RuntimeError(
            f"{self!r} has no memory store for conv_idx={conv_idx}. "
            "Call ingest_conversation first."
        )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.llm.model!r}, top_k={self.top_k})"

    def __str__(self) -> str:
        return repr(self)

    # Prevent accidental shadowing of the two abstract methods in subclasses
    # that forget to make them async.
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        import asyncio
        for method_name in ("ingest_conversation", "process_question"):
            method = cls.__dict__.get(method_name)
            if method is not None and not asyncio.iscoroutinefunction(method):
                raise TypeError(
                    f"{cls.__name__}.{method_name} must be defined with `async def`."
                )
