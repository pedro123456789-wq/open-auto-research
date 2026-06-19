"""Configuration for the self-improving memory agent.

All values here are intended to be edited by the user. The orchestrator
reads this module at startup; change a value and re-run.
"""

from __future__ import annotations
import os

# ── Run identity ──────────────────────────────────────────────────────────────
RUN_NAME = "agentic_mem"

# ── Paths (resolved relative to this file) ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# Where each evolution run's artefacts are stored
RUNS_DIR = os.path.join(_HERE, "runs")

# Resume an existing run folder (None = start fresh)
RESUME_RUN_DIR: str | None = None

# Seed pipeline that defines the root node
SEED_PIPELINE = os.path.join(_HERE, "baselines", RUN_NAME, "base_agentic_pipeline_openrouter.py")

# Dataset JSON (LoCoMo subset used for evaluation)
DATASET = os.path.join(_HERE, "evaluators", RUN_NAME, "locomo10.json")

# ── Evaluation ────────────────────────────────────────────────────────────────
# Per-evaluation timeout in seconds (None = no limit)
RUN_TIMEOUT: int = 8000

# ── DGM parent selection ──────────────────────────────────────────────────────

DGM_LAMBDA: float = 10.0
DGM_ALPHA0: float = 0.5
DGM_MIN_PARENT_ACCURACY: float = 0.2

# ── Evolution loop ────────────────────────────────────────────────────────────

K_ITERATIONS: int = 10
CHILDREN_PER_PARENT: int = 3
MAX_ATTEMPTS_PER_PARENT: int = 10

# ── LLM models ────────────────────────────────────────────────────────────────

PROPOSER_MODEL: str = "deepseek/deepseek-v4-pro"
ANSWERER_MODEL: str = "deepseek/deepseek-v4-flash"
JUDGE_MODEL: str = "deepseek/deepseek-v4-flash"
