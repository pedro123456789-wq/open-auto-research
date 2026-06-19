"""Improvement agent for the agentic_mem run.

Given the parent pipeline's source code and the eval metadata from the last
run, asks the LLM to produce a full replacement baseline file.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import RUN_NAME
from improvement_agents.improvement_agent_interface import ImprovementAgent
from utils.llm_utils import OpenRouterLLM

# backend/improvement_agents/agentic_mem/ -> ../../scientific_papers/<RUN_NAME>/processed/
PROCESSED_PAPERS_DIR = (
    Path(__file__).resolve().parents[2] / "scientific_papers" / RUN_NAME / "processed"
)


# ── system prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You improve a Python baseline that ingests LOCOMO conversations and answers questions about them.

## Mission

Produce a complete replacement baseline file that improves accuracy on the benchmark. 
Your design should be novel, elegant, and generalisable — inspired by how human memory works, not a minor patch on the parent.

Aim for a global maximum: explore bold architectural changes, not incremental tweaks that only fix one failure mode.

## Inputs

You will receive:
  • Parent baseline source code
  • Evaluation metadata (accuracy, wrong-answer samples)
  • Research paper summaries with key insights

Use wrong-answer samples to diagnose failures. Use the papers as inspiration \
only — adapt ideas to this codebase; do not reproduce a paper's system \
verbatim.

## Design goals

  • Outperform the parent baseline on the observed failure modes
  • Build on the provided papers while remaining original
  • Prefer practical, generalisable memory architectures
  • REQUIRED: combine a parametric memory method AND a text-based external memory \
method (see constraint 7 below)

## Hard constraints

1. Baseline interface — define exactly ONE concrete class inheriting from \
Baseline:
  class YourBaseline(Baseline):
      name = "your_baseline"

      async def ingest_conversation(
          self, conv_idx: int, entry: dict, output_dir: str,
      ) -> tuple[bool, str, int]:
          # Persist memory for conv_idx; return (success, user_id, items_stored)

      async def process_question(self, question: str, conv_idx: int) -> str:
          # Return a short answer string only — no scoring or evaluation

The base class provides self.llm (OpenRouterLLM) and self.top_k. If you \
override __init__, call super().__init__(model=model, top_k=top_k).

----
2. Agentic loop — process_question MUST keep a multi-step agent loop: the LLM \
repeatedly chooses actions (e.g. reading memory, calling tools) and only \
returns a final answer after one or more steps. Do not replace this with a \
single-shot generate call.

----
3. Harness contract — the evaluator runs:

  await baseline.ingest_conversation(conv_idx, entry, output_dir)
  answer = await baseline.process_question(question, conv_idx)

  ingest_conversation must write durable state under output_dir and set up \
instance handles (e.g. self._store) that process_question reads for the same \
conv_idx. Use self._not_ingested(conv_idx) if process_question runs before \
ingest. Ingest and query paths must match — a mismatch yields empty reads.

----
4. Imports — use flat imports (locomo_base is on sys.path), e.g. \
from baseline import Baseline. Do not import from eval; scoring is external.

----
5. Package imports (MANDATORY) — the top-of-file import block in the parent \
baseline is fixed. You MUST copy it verbatim into your replacement file:

  • Do NOT add, remove, reorder, or rename any import lines in that block.
  • Do NOT import any package that is not already present there — even if \
unused in your design. If you need functionality, use only what is already \
imported (stdlib, numpy, scipy, torch, chromadb, sklearn, etc.).
  • Keep every existing import line exactly as written, including # noqa: F401 \
comments and section headers, even when a package is unused in your code.

Only imports below that block (e.g. from baseline import Baseline) may stay \
as-is; do not introduce new third-party or stdlib imports anywhere in the file.

----
6. Generalisability — your approach should be generalisable to other tasks and \
datasets, not just the LOCOMO benchmark. DO NOT hardcode anything specific to \
the LOCOMO benchmark or questions mentioned.

----
7. Dual memory architecture (MANDATORY) — your design MUST incorporate BOTH of \
the following memory types in a meaningful, non-trivial way:

  a) PARAMETRIC memory — knowledge encoded in learned model weights or learned \
representations. Use only libraries already imported in the parent file (e.g. \
numpy, scipy, torch, sklearn). Do not add new embedding or ML imports.

  b) TEXT-BASED external memory — knowledge stored and retrieved as raw text \
through harness-level tool calls. Use only libraries already imported in the \
parent file (e.g. rank_bm25, chromadb, sklearn TfidfVectorizer, networkx).

  Both methods must be active at ingest time (populating their respective \
stores) and at query time (the agent loop must expose tool actions for each). \
A design that uses only one type will be rejected.

## What you may change

Prompts, tool schemas, action formats, constants, memory layout, ingestion \
logic, helpers, and any other implementation detail — except the top-of-file \
package import block (see constraint 5).

## Output

Return JSON with exactly three keys:
  "reasoning" — brief diagnosis of failures and what you changed
  "novelty"   — how your approach is novel and which paper ideas it builds on
  "code"      — the full replacement Python source (no markdown fences)
"""


# ── paper context ─────────────────────────────────────────────────────────────

def _format_paper_text(data: dict) -> str:
    lines = [
        f"Title: {data['title']}",
        f"Summary: {data['summary']}",
        "Key insights:",
    ]
    for insight in data.get("key_insights", []):
        lines.append(f"- {insight}")
    return "\n".join(lines)


def load_paper_context(processed_dir: Path | None = None) -> tuple[str, list[str]]:
    directory = processed_dir or PROCESSED_PAPERS_DIR
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        return "", []
    parts, titles = [], []
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        titles.append(data.get("title", path.stem))
        parts.append(_format_paper_text(data))
    return "\n\n---\n\n".join(parts), titles


# ── prompt builder ────────────────────────────────────────────────────────────

def _format_wrong_samples(samples: list[dict]) -> str:
    if not samples:
        return "  (no wrong-answer samples available)"
    lines = []
    for i, s in enumerate(samples, 1):
        lines.append(
            f"  [{i}] Category: {s.get('category', '?')}\n"
            f"      Question:  {s.get('question', '')}\n"
            f"      Model gave: {s.get('generated_answer', '')}\n"
            f"      Correct:    {s.get('ground_truth', '')}"
        )
    return "\n\n".join(lines)


def build_prompt(
    parent_code: str,
    parent_accuracy: float,
    metadata: dict,
    paper_context: str | None = None,
    paper_titles: list[str] | None = None,
) -> str:
    wrong_samples = metadata.get("wrong_samples", [])
    samples_str = _format_wrong_samples(wrong_samples)

    if paper_context is None or paper_titles is None:
        paper_context, paper_titles = load_paper_context()

    if paper_titles:
        papers_list = "\n".join(f"  • {title}" for title in paper_titles)
        papers_section = (
            f"Papers to build on ({len(paper_titles)}):\n\n"
            f"{papers_list}\n\n"
            f"Research context:\n\n"
            f"{paper_context}\n"
        )
    else:
        papers_section = "Research context: (no processed papers found)\n"

    correct = metadata.get("correct", "?")
    total = metadata.get("total", "?")

    return (
        f"Parent accuracy: {parent_accuracy:.1%} ({correct}/{total})\n\n"
        f"Wrong-answer samples:\n\n"
        f"{samples_str}\n\n"
        f"{papers_section}\n"
        f"Parent baseline source:\n\n"
        f"```python\n{parent_code}\n```\n\n"
        f"Return the improved replacement as JSON with keys "
        f"'reasoning', 'novelty', and 'code'."
    )


# ── improvement agent ─────────────────────────────────────────────────────────

_PROPOSER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "novelty": {"type": "string"},
        "code": {"type": "string"},
    },
    "required": ["reasoning", "novelty", "code"],
    "additionalProperties": False,
}


class AgenticMemImprovementAgent(ImprovementAgent):
    async def propose_improvement(
        self,
        llm: OpenRouterLLM,
        parent_code: str,
        parent_accuracy: float,
        metadata: dict,
    ) -> tuple[str, str, str]:
        user_prompt = build_prompt(parent_code, parent_accuracy, metadata)

        result = await llm.generate_structured(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            schema=_PROPOSER_SCHEMA,
        )

        if not result or not all(k in result for k in ("code", "reasoning", "novelty")):
            raise ValueError("Proposer returned empty code, reasoning, or novelty.")

        return result["code"], result["reasoning"], result["novelty"]
