"""
Extract text from PDFs in relevant_papers/ and summarise them via OpenRouterLLM.

Outputs one JSON file per paper in relevant_papers/processed/ (skipped if already present).

Run from locomo_base:
    python improvement_agents/ingest_papers.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_LOCOMO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LOCOMO_BASE not in sys.path:
    sys.path.insert(0, _LOCOMO_BASE)

from pypdf import PdfReader
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from utils.llm_utils import OpenRouterLLM

load_dotenv()
logger = logging.getLogger(__name__)



# ===============================================================================
# PATHS
# ===============================================================================
AGENT_DIR = Path(__file__).resolve().parent
PAPERS_DIR = AGENT_DIR / "relevant_papers"
PROCESSED_DIR = PAPERS_DIR / "processed"
CACHE_DIR = os.getenv("CACHE_DIR")

# HuggingFace repo for DeepSeek-V4 tokenizer (matches OpenRouter deepseek/* models)
DEEPSEEK_TOKENIZER_REPO = "deepseek-ai/DeepSeek-V4-Flash"
_deepseek_tokenizer: Tokenizer | None = None

# ===============================================================================
# SYSTEM PROMPT — edit this to steer summarisation for your goal
# ===============================================================================
SYSTEM_PROMPT = """\
You summarise research papers on conversational and agentic memory systems.

Given the full text of a paper, produce:
  • title — the paper's title as printed on the document
  • summary — a concise overview of the paper's problem, approach, and results
  • key_insights — a list of concrete, actionable ideas relevant to building memory systems that ingest long conversations and answer questions about them

The goal is to build generic memory systems that are practical, generalise and inspired by the human brain.

Focus on implementation-relevant details (architecture, ingestion, retrieval, \
update policies, agent loops). Omit boilerplate, related-work fluff, and \
citation minutiae.
"""

# ===============================================================================
# LLM SCHEMA
# ===============================================================================
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["title", "summary", "key_insights"],
    "additionalProperties": False,
}

# ===============================================================================
# PDF EXTRACTION
# ===============================================================================
def extract_pdf_text(pdf_path: Path) -> str:
    """Return all extractable text from a PDF, page by page."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
        else:
            logger.debug("No text on page %d of %s", i + 1, pdf_path.name)
    return "\n\n".join(pages)


def _output_path(pdf_path: Path) -> Path:
    return PROCESSED_DIR / f"{pdf_path.stem}.json"


def _is_processed(pdf_path: Path) -> bool:
    return _output_path(pdf_path).is_file()


# ===============================================================================
# SUMMARISATION
# ===============================================================================
async def summarise_paper(llm: OpenRouterLLM, pdf_path: Path, text: str) -> dict:
    user_prompt = (
        f"Paper filename: {pdf_path.name}\n\n"
        f"--- BEGIN PAPER TEXT ---\n{text}\n--- END PAPER TEXT ---"
    )
    result = await llm.generate_structured(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        schema=_OUTPUT_SCHEMA,
    )
    if not result or not all(k in result for k in ("title", "summary", "key_insights")):
        raise ValueError(f"LLM returned incomplete structured output for {pdf_path.name}")
    return result


def _write_output(pdf_path: Path, result: dict) -> Path:
    out_path = _output_path(pdf_path)
    payload = {
        "title": result["title"],
        "summary": result["summary"],
        "key_insights": result["key_insights"],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


# ===============================================================================
# TOKEN COUNTING (ensures that we don't exceed the token limit of the LLM)
# ===============================================================================
def _get_deepseek_tokenizer() -> Tokenizer:
    """Lazy-load the DeepSeek-V4 tokenizer from HuggingFace."""

    global _deepseek_tokenizer
    if _deepseek_tokenizer is None:
        tokenizer_path = hf_hub_download(
            DEEPSEEK_TOKENIZER_REPO, 
            "tokenizer.json", 
            cache_dir=CACHE_DIR
        )

        _deepseek_tokenizer = Tokenizer.from_file(tokenizer_path)
    return _deepseek_tokenizer


def _format_paper_text(data: dict) -> str:
    """Format a processed paper JSON object into a single string."""
    lines = [
        f"Title: {data['title']}",
        f"Summary: {data['summary']}",
        "Key insights:",
    ]
    for insight in data.get("key_insights", []):
        lines.append(f"- {insight}")
    return "\n".join(lines)


def count_tokens(processed_dir: Path | None = None) -> int:
    """
    Read all JSON files in processed/, format each into text, join them,
    and return the total DeepSeek token count for the combined string.
    """
    directory = processed_dir or PROCESSED_DIR
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        return 0

    parts: list[str] = []
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        parts.append(_format_paper_text(data))

    combined = "\n\n".join(parts)
    tokenizer = _get_deepseek_tokenizer()
    return len(tokenizer.encode(combined).ids)


# ===============================================================================
# MAIN LOOP
# ===============================================================================
async def ingest_all(model: str | None = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", PAPERS_DIR)
        return

    llm = OpenRouterLLM(model=model)
    logger.info("Found %d PDF(s) in %s (model=%s)", len(pdf_files), PAPERS_DIR, llm.model)

    skipped = processed = failed = 0
    total = len(pdf_files)

    for i, pdf_path in enumerate(pdf_files, 1):
        if _is_processed(pdf_path):
            logger.info("[%d/%d] Skipping %s (already processed)", i, total, pdf_path.name)
            skipped += 1
            continue

        logger.info("[%d/%d] Processing %s ...", i, total, pdf_path.name)
        try:
            logger.info("  Extracting text from %s", pdf_path.name)
            text = extract_pdf_text(pdf_path)
            if not text.strip():
                raise ValueError("No extractable text found")
            logger.info("  Extracted %d characters — calling LLM ...", len(text))

            result = await summarise_paper(llm, pdf_path, text)
            out_path = _write_output(pdf_path, result)
            logger.info("  Wrote %s (%s)", out_path.name, result["title"])
            processed += 1
        except Exception as exc:
            logger.error("  Failed to process %s: %s", pdf_path.name, exc)
            failed += 1

    logger.info(
        "Done — processed=%d skipped=%d failed=%d (total=%d)",
        processed, skipped, failed, total,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(ingest_all())

    n_files = len(list(PROCESSED_DIR.glob("*.json")))
    total_tokens = count_tokens()
    logger.info(
        "Total DeepSeek tokens across %d processed file(s): %d",
        n_files, total_tokens,
    )


if __name__ == "__main__":
    main()
