"""DGM-inspired self-improving memory agent — main entrypoint.

This file is run-agnostic. It resolves the evaluator and improvement agent
dynamically from cfg.RUN_NAME, so switching runs only requires changing config.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

import config as cfg
from utils.archive import Archive, Node
import utils.storage as storage
from utils.log_utils import setup_run_logging
from utils.llm_utils import OpenRouterLLM
from evaluators.evaluator_interface import Evaluator
from improvement_agents.improvement_agent_interface import ImprovementAgent

load_dotenv()
logger = logging.getLogger(__name__)


# ── Dynamic loader helpers ────────────────────────────────────────────────────

def _load_evaluator() -> Evaluator:
    """Import evaluators/<RUN_NAME>/evaluator.py and return an Evaluator instance."""
    module = importlib.import_module(f"evaluators.{cfg.RUN_NAME}.evaluator")
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, Evaluator) and cls is not Evaluator:
            return cls()
    raise RuntimeError(f"No Evaluator subclass found in evaluators/{cfg.RUN_NAME}/evaluator.py")


def _load_improvement_agent() -> ImprovementAgent:
    """Import improvement_agents/<RUN_NAME>/improvement_agent.py and return an ImprovementAgent instance."""
    module = importlib.import_module(f"improvement_agents.{cfg.RUN_NAME}.improvement_agent")
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, ImprovementAgent) and cls is not ImprovementAgent:
            return cls()
    raise RuntimeError(
        f"No ImprovementAgent subclass found in improvement_agents/{cfg.RUN_NAME}/improvement_agent.py"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _child_node_id(round_num: int, child_num: int) -> str:
    return f"gen{round_num}_c{child_num}"


# ── Phase 1: seed root ────────────────────────────────────────────────────────

async def _ensure_root(run_dir: str, evaluator: Evaluator) -> tuple[Archive, Node]:
    """Ensure the archive has a fully evaluated root node."""
    archive = storage.load_archive(run_dir, cfg.DGM_LAMBDA, cfg.DGM_ALPHA0, cfg.DGM_MIN_PARENT_ACCURACY)

    if storage.root_exists(run_dir):
        root = storage.load_root_node(run_dir)
        if archive.get("root") is None:
            archive.add(root)
        logger.info("Root already exists — skipping seed eval")
        logger.info("Root | accuracy=%.1f%% | %s", root.accuracy * 100, root.metadata)
        return archive, root

    root_pipeline_path = storage.seed_root_node(run_dir, cfg.SEED_PIPELINE)
    logger.info("Seeding root node from %s", cfg.SEED_PIPELINE)

    node_output_dir = os.path.join(run_dir, "nodes", "root", "eval_output")
    try:
        accuracy, metadata = await evaluator.evaluate(root_pipeline_path, node_output_dir)
    except Exception as exc:
        raise RuntimeError(f"Root evaluation failed: {exc}") from exc

    root = Node(
        node_id="root",
        parent_id=None,
        round=0,
        reasoning="Baseline seed pipeline.",
        compiles=True,
        accuracy=accuracy,
        metadata=metadata,
        timestamp=datetime.now().isoformat(),
        model="baseline",
    )

    with open(root_pipeline_path, encoding="utf-8") as f:
        seed_code = f.read()
    storage.write_node(run_dir, root, seed_code, parent_code=None)

    if archive.get("root") is None:
        archive.add(root)
    storage.save_archive(run_dir, archive)

    logger.info(
        "Root | accuracy=%.1f%% | correct=%s/%s",
        accuracy * 100,
        metadata.get("correct", "?"),
        metadata.get("total", "?"),
    )
    return archive, root


# ── Phase 2: single child attempt ────────────────────────────────────────────
async def _attempt_child(
    proposer_llm: OpenRouterLLM,
    improvement_agent: ImprovementAgent,
    evaluator: Evaluator,
    parent: Node,
    parent_code: str,
    run_dir: str,
    node_id: str,
    round_num: int,
) -> Node | None:
    """Propose, validate, and evaluate one child baseline. Returns None on any failure."""
    logger.info("  Proposing %s (parent=%s) ...", node_id, parent.node_id)

    try:
        new_code, reasoning, novelty = await improvement_agent.propose_improvement(
            llm=proposer_llm,
            parent_code=parent_code,
            parent_accuracy=parent.accuracy,
            metadata=parent.metadata,
        )
    except Exception as exc:
        logger.warning("  Proposer failed: %s", exc)
        return None

    child_pipeline_path = os.path.join(run_dir, "nodes", node_id, "pipeline.py")
    os.makedirs(os.path.dirname(child_pipeline_path), exist_ok=True)
    with open(child_pipeline_path, "w", encoding="utf-8") as f:
        f.write(new_code)

    logger.info("  Code generated | reasoning: %s", reasoning)
    logger.info("  Novelty: %s", novelty)

    node_output_dir = os.path.join(run_dir, "nodes", node_id, "eval_output")

    timed_out = False
    accuracy = 0.0
    metadata: dict = {}

    try:
        eval_coro = evaluator.evaluate(child_pipeline_path, node_output_dir)
        if cfg.RUN_TIMEOUT is not None:
            accuracy, metadata = await asyncio.wait_for(eval_coro, timeout=cfg.RUN_TIMEOUT)
        else:
            accuracy, metadata = await eval_coro
    except asyncio.TimeoutError:
        timeout_msg = f"Timed out after {cfg.RUN_TIMEOUT}s"
        logger.warning("  %s: %s", node_id, timeout_msg)
        timed_out = True
        metadata = {"timed_out": True, "error": timeout_msg}
        # Write a stub so the node directory is well-formed
        os.makedirs(node_output_dir, exist_ok=True)
        with open(os.path.join(node_output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": {**metadata, "accuracy": 0.0, "timestamp": datetime.now().isoformat()}, "results": []},
                f, indent=2,
            )
    except Exception as exc:
        # Hard failure (syntax error, missing Baseline, crash) — skip this candidate
        logger.info("  Evaluation failed for %s: %s", node_id, exc)
        return None

    if metadata.get("early_stopped"):
        logger.warning(
            "  %s stopped early: %s/%s correct (%.1f%%)",
            node_id,
            metadata.get("correct", "?"),
            metadata.get("total", "?"),
            accuracy * 100,
        )
    elif timed_out:
        logger.warning("  %s recorded with accuracy=0 due to timeout", node_id)

    node = Node(
        node_id=node_id,
        parent_id=parent.node_id,
        round=round_num,
        reasoning=f"{reasoning}\n\n[TIMEOUT] {metadata.get('error', '')}" if timed_out else reasoning,
        novelty=novelty,
        compiles=True,
        accuracy=accuracy,
        metadata=metadata,
        timestamp=datetime.now().isoformat(),
        model=proposer_llm.model,
    )

    logger.info("  Node created | accuracy=%.1f%% | metadata=%s", node.accuracy * 100, node.metadata)
    storage.write_node(run_dir, node, new_code, parent_code=parent_code)
    return node


# ── Phase 2: evolution loop ───────────────────────────────────────────────────
async def _evolution_loop(
    archive: Archive,
    run_dir: str,
    improvement_agent: ImprovementAgent,
    evaluator: Evaluator,
    child_counter: int = 0,
) -> None:
    proposer_llm = OpenRouterLLM(model=cfg.PROPOSER_MODEL)

    for round_num in range(1, cfg.K_ITERATIONS + 1):
        parent = archive.select_parent()
        parent_code = storage.read_pipeline_code(run_dir, parent.node_id)

        logger.info(
            "Round %d/%d | Parent: %s (acc=%.1f%%)",
            round_num, cfg.K_ITERATIONS, parent.node_id, parent.accuracy * 100,
        )

        children_compiled = 0
        attempts = 0

        while children_compiled < cfg.CHILDREN_PER_PARENT and attempts < cfg.MAX_ATTEMPTS_PER_PARENT:
            attempts += 1
            child_counter += 1
            node_id = _child_node_id(round_num, child_counter)

            child = await _attempt_child(
                proposer_llm=proposer_llm,
                improvement_agent=improvement_agent,
                evaluator=evaluator,
                parent=parent,
                parent_code=parent_code,
                run_dir=run_dir,
                node_id=node_id,
                round_num=round_num,
            )

            if child is not None:
                archive.add(child)
                storage.save_archive(run_dir, archive)
                children_compiled += 1
                logger.info(
                    "  + Added %s | acc=%.1f%% | archive size=%d",
                    child.node_id, child.accuracy * 100, len(archive),
                )

        if children_compiled == 0:
            logger.warning("  No compiling children in round %d after %d attempts.", round_num, attempts)
        else:
            logger.info("  Round %d: %d/%d children compiled.", round_num, children_compiled, attempts)

        logger.info("Leaderboard:")
        for row in archive.summary_rows()[:10]:
            logger.info(
                "  %s | acc=%.1f%% | children=%d | parent=%s",
                row["node_id"], row["accuracy"] * 100, row["children"], row["parent_id"] or "-",
            )


# ── Entrypoint ────────────────────────────────────────────────────────────────
async def _run(run_dir: str) -> Archive:
    evaluator = _load_evaluator()
    improvement_agent = _load_improvement_agent()

    archive, _root = await _ensure_root(run_dir, evaluator)

    child_counter = storage.max_child_counter(archive)
    if child_counter:
        logger.info("Resuming: %d existing nodes (child counter=%d)", len(archive), child_counter)

    await _evolution_loop(archive, run_dir, improvement_agent, evaluator, child_counter=child_counter)
    return archive


def main() -> None:
    os.makedirs(cfg.RUNS_DIR, exist_ok=True)
    run_dir = storage.resolve_run_dir(cfg.RUNS_DIR, cfg.RESUME_RUN_DIR)
    setup_run_logging(run_dir)

    logger.info("Self-Improving Memory Agent | run=%s", cfg.RUN_NAME)
    logger.info("Run dir: %s", run_dir)
    logger.info("Seed:    %s", cfg.SEED_PIPELINE)
    if cfg.RESUME_RUN_DIR:
        logger.info("Resuming: %s", cfg.RESUME_RUN_DIR)
    logger.info(
        "Config: children_per_parent=%d | run_timeout=%s",
        cfg.CHILDREN_PER_PARENT,
        f"{cfg.RUN_TIMEOUT}s" if cfg.RUN_TIMEOUT is not None else "disabled",
    )

    archive = asyncio.run(_run(run_dir))
    best = archive.best()
    if best is None:
        logger.warning("Archive is empty — nothing to report.")
        return

    best_path = storage.pipeline_path(run_dir, best.node_id)
    logger.info("DONE")
    logger.info("Best node:  %s", best.node_id)
    logger.info("Accuracy:   %.1f%%", best.accuracy * 100)
    logger.info("Correct:    %s/%s", best.metadata.get("correct", "?"), best.metadata.get("total", "?"))
    logger.info("Parent:     %s", best.parent_id or "-")
    logger.info("Pipeline:   %s", best_path)
    snippet = best.reasoning[:300] + ("..." if len(best.reasoning) > 300 else "")
    logger.info("Reasoning:  %s", snippet)

    print(f"\nBest pipeline: {best_path}")
    print(f"Accuracy: {best.accuracy:.1%}")


if __name__ == "__main__":
    main()
