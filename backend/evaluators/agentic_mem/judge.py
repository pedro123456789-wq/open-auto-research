"""LLM-as-judge for LoCoMo QA."""

from __future__ import annotations

import logging

from evaluators.agentic_mem.prompts import JUDGE_SYSTEM, JUDGE_PROMPT, JUDGE_SCHEMA

logger = logging.getLogger(__name__)


def _gold_answer(category: int, gold: str) -> str:
    # ponytail: cat-3 gold may list aliases after ';' — judge only needs the first
    if category == 3 and ";" in gold:
        return gold.split(";")[0].strip()
    return gold


async def judge_answer(
    judge_llm,
    category: int,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> dict:
    """Score one QA pair with an LLM judge.

    Returns judgment (CORRECT/WRONG), score (1.0/0.0), and reason.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=_gold_answer(category, gold_answer),
        response=generated_answer,
    )

    try:
        raw = await judge_llm.generate_structured(
            system=JUDGE_SYSTEM,
            user=prompt,
            schema=JUDGE_SCHEMA,
        )
    except Exception as exc:
        logger.warning("Judge LLM call failed: %s — defaulting to WRONG", exc)
        raw = {}

    correct = isinstance(raw, dict) and raw.get("label", "").upper() == "CORRECT"
    return {
        "judgment": "CORRECT" if correct else "WRONG",
        "score": 1.0 if correct else 0.0,
        "reason": raw.get("reasoning", "") if isinstance(raw, dict) else "",
    }
