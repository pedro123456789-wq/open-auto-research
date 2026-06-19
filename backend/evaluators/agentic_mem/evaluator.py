"""Evaluator for the agentic_mem run.

Loads a candidate pipeline.py, runs it against the LoCoMo benchmark, and
returns (accuracy, metadata) matching the Evaluator interface.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import random
import sys
from datetime import datetime

import config as cfg
from evaluators.evaluator_interface import Evaluator
from evaluators.agentic_mem.judge import judge_answer
from utils.llm_utils import OpenRouterLLM
from utils.locomo_utils import load_dataset
from typing import Optional


_DATASET: Optional[list] = None
logger = logging.getLogger(__name__)

# Resolved once at import time so every evaluate() call shares one dataset.
def _get_dataset() -> list:
    global _DATASET
    if _DATASET is None:
        _DATASET = load_dataset(cfg.DATASET)
    return _DATASET

N_CONVERSATIONS: int = 5          # conversations randomly sampled per eval
K_QUESTIONS: int = 10             # questions sampled per conversation (stratified)
NUM_PARALLEL: int = 4             # questions evaluated in parallel per batch
_TOP_K: int = 10
_SEED: int = 42

# Early-stop a child eval if fewer than EARLY_STOP_MIN_CORRECT answers are
# correct after EARLY_STOP_AFTER_N questions
EARLY_STOP_AFTER_N: int = 10
EARLY_STOP_MIN_CORRECT: int = 2

WRONG_SAMPLE_SIZE: int = 15


def _stratified_sample(questions: list, k: int, rng: random.Random) -> list:
    """Sample k questions from a list of (qi, qa) tuples, spread evenly across categories."""
    from collections import defaultdict
    buckets: dict[int, list] = defaultdict(list)
    for item in questions:
        buckets[item[1].get("category")].append(item)

    cats = sorted(buckets)
    if not cats:
        return []

    base, extra = divmod(k, len(cats))
    selected: list = []
    used: set[int] = set()

    for i, cat in enumerate(cats):
        quota = base + (1 if i < extra else 0)
        picks = rng.sample(buckets[cat], min(quota, len(buckets[cat])))
        selected.extend(picks)
        used.update(qi for qi, _ in picks)

    # fill shortfall (category too small) from remaining questions
    shortfall = k - len(selected)
    if shortfall > 0:
        leftover = [q for q in questions if q[0] not in used]
        selected.extend(rng.sample(leftover, min(shortfall, len(leftover))))

    return selected


def _load_baseline_cls(pipeline_path: str):
    """Dynamically load pipeline_path and return the concrete Baseline subclass.
    Gets the baseline class corresponding to the run name from this evaluator class.

    Puts baselines/<RUN_NAME>/ on sys.path so the candidate can do
    'from baseline import Baseline' the same way it was authored.
    Raises RuntimeError on syntax error, missing subclass, or contract breach.
    """

    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    baseline_dir = os.path.join(backend_dir, "baselines", cfg.RUN_NAME)

    for path in (backend_dir, baseline_dir, os.path.join(backend_dir, "utils")):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("_candidate_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # needed so @dataclass can resolve cls.__module__
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise RuntimeError(f"Pipeline failed to load: {exc}") from exc

    # Import Baseline from the run's baselines folder (same identity as the candidate uses).
    from baseline import Baseline  # type: ignore[import]

    candidates = [
        cls for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, Baseline) and cls is not Baseline and cls.__module__ == module.__name__
    ]
    if not candidates:
        raise RuntimeError("No concrete Baseline subclass found in pipeline.")
    return candidates[0]


async def _eval_question(baseline, judge, conv_idx: int, qi: int, qa: dict) -> dict:
    question, category = qa["question"], qa["category"]
    gold = str(qa.get("answer") or qa.get("adversarial_answer", ""))
    try:
        answer = await baseline.process_question(question, conv_idx)
    except Exception as exc:
        logger.warning("conv_idx=%d q%d process_question failed: %s", conv_idx, qi, exc)
        answer = ""
    try:
        judgment = await judge_answer(judge, category, question, gold, answer)
    except Exception as exc:
        logger.warning("conv_idx=%d q%d judge failed: %s", conv_idx, qi, exc)
        judgment = {"judgment": "WRONG", "score": 0.0, "reason": str(exc)}
    return {"conv_idx": conv_idx, "question": question, "category": category,
            "answer": answer, "gold": gold, **judgment}


class AgenticMemEvaluator(Evaluator):
    async def evaluate(self, pipeline_path: str, output_dir: str) -> tuple[float, dict]:
        """Run the candidate against the LoCoMo benchmark.

        Returns (accuracy, metadata) where metadata contains correct, total,
        early_stopped, wrong_samples (for the improvement agent).
        Raises on hard failure so the core loop can skip this candidate.
        """
        baseline_cls = _load_baseline_cls(pipeline_path)

        data = _get_dataset()
        os.makedirs(output_dir, exist_ok=True)

        judge = OpenRouterLLM(model=cfg.JUDGE_MODEL)
        baseline = baseline_cls(model=cfg.ANSWERER_MODEL, top_k=_TOP_K)

        rng = random.Random(_SEED)
        valid_convs = rng.sample(range(len(data)), min(N_CONVERSATIONS, len(data)))
        correct_count = 0
        total_count = 0
        all_results: list[dict] = []
        early_stopped = False

        total_questions = N_CONVERSATIONS * K_QUESTIONS
        logger.info("Evaluating %s | %d convs × %d q = %d total", baseline_cls.__name__, len(valid_convs), K_QUESTIONS, total_questions)

        for conv_num, conv_idx in enumerate(valid_convs, 1):
            entry = data[conv_idx]
            pool = [(qi, qa) for qi, qa in enumerate(entry.get("qa", []))]
            questions = _stratified_sample(pool, K_QUESTIONS, random.Random(_SEED + conv_idx))

            logger.info("Conv %d/%d | ingesting ...", conv_num, len(valid_convs))

            try:
                await baseline.ingest_conversation(conv_idx, entry, output_dir)
            except Exception as exc:
                raise RuntimeError(f"Ingest failed conv_idx={conv_idx}: {exc}") from exc

            for batch_start in range(0, len(questions), NUM_PARALLEL):
                batch = questions[batch_start:batch_start + NUM_PARALLEL]
                batch_results = await asyncio.gather(
                    *[_eval_question(baseline, judge, conv_idx, qi, qa) for qi, qa in batch]
                )
                for r in batch_results:
                    is_correct = r["score"] >= 0.5
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    all_results.append(r)

                logger.info(
                    "Conv %d/%d | %d/%d answered | acc %.1f%% (%d/%d)",
                    conv_num, len(valid_convs),
                    min(batch_start + NUM_PARALLEL, len(questions)), len(questions),
                    correct_count / total_count * 100 if total_count else 0.0,
                    correct_count, total_count,
                )

                if (
                    EARLY_STOP_AFTER_N
                    and total_count >= EARLY_STOP_AFTER_N
                    and correct_count < EARLY_STOP_MIN_CORRECT
                ):
                    logger.warning("Early stop: %d/%d correct after %d questions", correct_count, total_count, total_count)
                    early_stopped = True
                    break

            if early_stopped:
                break

        accuracy = correct_count / total_count if total_count else 0.0

        # Sample wrong answers for the improvement agent.
        wrong = [r for r in all_results if r.get("score", 0) < 0.5]
        n = min(WRONG_SAMPLE_SIZE, len(wrong))
        wrong_samples = [
            {
                "question": r["question"],
                "category": r.get("category"),
                "generated_answer": r.get("answer", ""),
                "ground_truth": r.get("gold", ""),
            }
            for r in (random.sample(wrong, n) if n else [])
        ]

        out_path = os.path.join(output_dir, "eval_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {
                        "accuracy": accuracy,
                        "correct": correct_count,
                        "total": total_count,
                        "early_stopped": early_stopped,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "results": all_results,
                },
                f,
                indent=2,
            )

        logger.info(
            "Eval done | acc %.1f%% (%d/%d) | early_stopped=%s | %s",
            accuracy * 100, correct_count, total_count, early_stopped, out_path,
        )

        metadata = {
            "correct": correct_count,
            "total": total_count,
            "accuracy": accuracy,
            "early_stopped": early_stopped,
            "wrong_samples": wrong_samples,
        }
        return accuracy, metadata
