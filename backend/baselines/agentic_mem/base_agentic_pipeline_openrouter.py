"""
LOCOMO agentic baseline — plain-text store + tool-use loop.
Can be edited by the meta-agent.
"""

from __future__ import annotations

# ── core ──────────────────────────────────────────────────────────────────────
import bisect                                           # noqa: F401
import copy                                             # noqa: F401
import functools                                        # noqa: F401
import itertools                                        # noqa: F401
import json
import logging
import math                                              # noqa: F401
import os
import re
import sqlite3                                          # noqa: F401
from collections import Counter, defaultdict            # noqa: F401
from dataclasses import dataclass, field                # noqa: F401
from datetime import datetime                           # noqa: F401
from typing import Any                                  # noqa: F401

# ── numerics / search  (available for meta-agent rewrites) ───────────────────
import difflib                                          # noqa: F401
import hashlib                                          # noqa: F401
import heapq                                            # noqa: F401
import unicodedata                                      # noqa: F401

import numpy as np                                      # noqa: F401
import scipy                                            # noqa: F401
import scipy.sparse                                     # noqa: F401
import torch                                            # noqa: F401  # δ-mem, Titans-style modules

# ── retrieval / graphs  (HippoRAG, Mem0g, A-Mem, GAM) ───────────────────────
import networkx as nx                                   # noqa: F401  # KG + PageRank
import nltk                                             # noqa: F401  # tokenization / stemming
from rank_bm25 import BM25Okapi                         # noqa: F401  # hybrid keyword search
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401  # Generative Agents

# ── data / IO / utils ─────────────────────────────────────────────────────────
import aiohttp                                          # noqa: F401
import chromadb                                         # noqa: F401  # vector store
import httpx                                            # noqa: F401
import orjson                                           # noqa: F401
import pydantic                                         # noqa: F401  # structured memory notes
import regex                                            # noqa: F401
import requests                                         # noqa: F401
import tenacity                                         # noqa: F401
import tiktoken                                         # noqa: F401  # MemGPT context budgeting
import yaml                                             # noqa: F401
from dateutil import parser as date_parser              # noqa: F401  # temporal QA
from tqdm import tqdm                                   # noqa: F401

from baseline import Baseline
from utils.llm_utils import OpenRouterLLM
from utils.locomo_utils import get_sorted_sessions

logger = logging.getLogger(__name__)


# ── constants ─────────────────────────────────────────────────────────────────
STORE_DIR_NAME = "textstore"
MAX_LINES_PER_READ = 120       # hard cap on lines returned per tool call
MAX_TOOL_CALLS = 8             # maximum read_memory calls per question
ANSWER_TIMEOUT_MSG = (
    "Maximum tool calls reached. Give your best answer now based on what "
    "you have read.  Respond with {\"action\": \"answer\", \"text\": \"...\"}."
)


# ── text-store handle ─────────────────────────────────────────────────────────
@dataclass
class TextStore:
    """Lightweight handle to the on-disk plain-text memory store."""
    store_dir: str

    def path_for(self, conv_idx: int) -> str:
        """Return the file path for a specific conversation."""
        return os.path.join(self.store_dir, f"conv_{conv_idx}.txt")

    def load(self, conv_idx: int) -> list[str]:
        """Load all lines for a conversation (list is 0-indexed; store is 1-indexed)."""
        p = self.path_for(conv_idx)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Text store not found: {p}")
        with open(p, encoding="utf-8") as fh:
            return fh.readlines()


def get_text_store(store_dir: str) -> TextStore:
    """Create the store directory if needed and return a TextStore handle."""
    os.makedirs(store_dir, exist_ok=True)
    return TextStore(store_dir=store_dir)


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a memory-recall agent.  You have access to a plain-text
memory store that records past conversations between {speaker_a} and {speaker_b}.

━━ MEMORY STORE ━━
Total lines : {total_lines}
{session_index}

━━ TOOL ━━
Call read_memory to retrieve a slice of the memory store.
  Arguments : start (int, 1-indexed), end (int, inclusive)
  Max lines per call : {max_lines}

━━ ANSWER RULES ━━
• Use only information from the memory store — do not guess from prior knowledge.
• Be concise.  Answer with a short phrase; use exact words from the store where possible.
• For temporal questions include the date.
• For multi-hop questions list all relevant items, comma-separated.
• If the information is genuinely not in the store, say "not mentioned".

━━ RESPONSE FORMAT (strict JSON, no markdown) ━━
  To read more memory : {{"action": "read_memory", "start": N, "end": M}}
  To give final answer: {{"action": "answer",      "text":  "..."}}"""

FIRST_USER_PROMPT = """Question: {question}

Start by reading a small portion of the memory store to orient yourself,
then retrieve what you need and answer."""

_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["read_memory", "answer"]},
        "start":  {"type": "integer"},
        "end":    {"type": "integer"},
        "text":   {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


# ── module-level helpers ───────────────────────────────────────────────────────
def _format_turn(turn: dict, speaker_a: str) -> str:
    """Render one conversation turn as a plain-text line."""
    speaker = turn.get("speaker", "?")
    text = turn.get("text", "").strip()
    blip = turn.get("blip_caption", "").strip()
    query = turn.get("query", "").strip()

    if query and blip:
        photo = f" [shares image — query: {query}; shows: {blip}]"
    elif blip:
        photo = f" [shares image: {blip}]"
    elif query:
        photo = f" [shares image — query: {query}]"
    else:
        photo = ""

    return f"{speaker}: {text}{photo}" if text or photo else ""


def _build_store_text(
    conv_idx: int, entry: dict
) -> tuple[str, list[tuple[str, str, int, int]]]:
    """Serialise a full conversation to a numbered plain-text string.

    Returns:
        text          — file content (header block + numbered body lines)
        session_index — list of (session_key, date_str, first_line, last_line)
                        with 1-based line numbers in the final text.
    """
    conversation = entry["conversation"]
    speaker_a = conversation.get("speaker_a", "A")
    speaker_b = conversation.get("speaker_b", "B")

    body_lines: list[str] = []
    session_index: list[tuple[str, str, int, int]] = []

    for session_key, date_str, turns in get_sorted_sessions(conversation):
        body_lines.append(f"=== {session_key.upper()} | DATE: {date_str} ===")
        session_start = len(body_lines)
        for turn in turns:
            line = _format_turn(turn, speaker_a)
            if line:
                body_lines.append(line)
        session_index.append((session_key, date_str, session_start, len(body_lines)))
        body_lines.append("")   # blank separator between sessions

    session_map = [
        f"  {sk} | {dt} | lines {s}–{e}"
        for sk, dt, s, e in session_index
    ]
    header_block = [
        f"=== MEMORY STORE | CONV {conv_idx} | {speaker_a} & {speaker_b} ===",
        f"Total content lines : {len(body_lines)}",
        "Sessions:",
        *session_map,
        "─" * 60,
    ]
    offset = len(header_block)
    numbered_body = [f"L{(i + 1 + offset):04d}: {ln}" for i, ln in enumerate(body_lines)]
    adjusted_index = [
        (sk, dt, s + offset, e + offset)
        for sk, dt, s, e in session_index
    ]

    return "\n".join(header_block + numbered_body), adjusted_index


# ── baseline class ─────────────────────────────────────────────────────────────

class AgenticBaseline(Baseline):
    """Plain-text memory store + LLM tool-use loop baseline."""

    name = "agentic"

    def __init__(self, model: str | None = None, top_k: int = 10) -> None:
        super().__init__(model=model, top_k=top_k)
        self._store: TextStore | None = None

    def _ensure_store(self) -> TextStore:
        """Raise clearly if ingest has not been called yet."""
        if self._store is None:
            raise self._not_ingested(-1)
        return self._store

    # ── Baseline interface ────────────────────────────────────────────────────

    async def ingest_conversation(
        self,
        conv_idx: int,
        entry: dict,
        output_dir: str,
    ) -> tuple[bool, str, int]:
        """Write the conversation to a numbered plain-text file.

        No embedding API is required — the file is used directly by the agent.
        """
        store_dir = os.path.join(output_dir, STORE_DIR_NAME)
        self._store = get_text_store(store_dir)

        try:
            text, _ = _build_store_text(conv_idx, entry)
        except Exception as exc:
            logger.error("Failed to build text store for conv %d: %s", conv_idx, exc)
            return False, f"locomo_conv_{conv_idx}", 0

        dest = self._store.path_for(conv_idx)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)

        n_lines = text.count("\n") + 1
        logger.debug("Text store written: %s (%d lines)", dest, n_lines)
        return True, f"locomo_conv_{conv_idx}", n_lines

    async def process_question(self, question: str, conv_idx: int) -> str:
        """Run the read_memory tool-use loop and return the final answer string."""
        store = self._ensure_store()
        all_lines = store.load(conv_idx)
        total_lines = len(all_lines)

        def read_memory(start: int, end: int) -> str:
            """Return lines [start, end] clamped to store bounds (1-indexed, inclusive)."""
            s = max(1, start)
            e = min(total_lines, min(end, start + MAX_LINES_PER_READ - 1))
            return "".join(all_lines[s - 1 : e])

        # Extract speaker names and session index from the file header
        header_line = all_lines[0] if all_lines else ""
        m = re.search(r"\|\s*(.+?)\s*&\s*(.+?)\s*===", header_line)
        speaker_a = m.group(1) if m else "Speaker A"
        speaker_b = m.group(2) if m else "Speaker B"

        raw_sessions = [
            ln.strip() for ln in all_lines[:20]
            if "→ lines" in ln or ln.strip().startswith("session_")
        ]
        session_index_str = (
            "Sessions:\n" + "\n".join(f"  {l}" for l in raw_sessions)
            if raw_sessions else "(see store header at lines 1–10)"
        )

        system = SYSTEM_PROMPT.format(
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            total_lines=total_lines,
            session_index=session_index_str,
            max_lines=MAX_LINES_PER_READ,
        )
        history: list[dict[str, str]] = [
            {"role": "user", "content": FIRST_USER_PROMPT.format(question=question)},
        ]
        generated_answer = "(no answer produced)"

        for call_n in range(MAX_TOOL_CALLS + 1):
            if call_n == MAX_TOOL_CALLS:
                history.append({"role": "user", "content": ANSWER_TIMEOUT_MSG})

            try:
                raw = await self.llm.generate_structured(
                    system=system,
                    user=history,
                    schema=_ACTION_SCHEMA,
                )
            except Exception as exc:
                logger.warning("Agent LLM call failed on call %d: %s", call_n, exc)
                break

            if not isinstance(raw, dict):
                logger.warning("Agent returned non-dict on call %d: %r", call_n, raw)
                break

            action = raw.get("action", "")

            if action == "answer":
                generated_answer = str(raw.get("text", "")).strip()
                break

            if action == "read_memory":
                start = int(raw.get("start", 1))
                end = int(raw.get("end", min(start + 49, total_lines)))
                snippet = read_memory(start, end)
                n_lines = len(snippet.splitlines())
                history.append({"role": "assistant", "content": json.dumps(raw)})
                history.append({
                    "role": "user",
                    "content": (
                        f"[read_memory({start}, {end}) → {n_lines} lines]\n"
                        f"{snippet}"
                    ),
                })
            else:
                logger.warning("Unknown agent action %r on call %d", action, call_n)
                break

        return generated_answer