"""
Aggregated per-category accuracy stats.
"""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import List

logger = logging.getLogger(__name__)

# Official display order for categories (mirrors evaluation_stats.py)
CATEGORY_ORDER = [4, 1, 2, 3, 5]

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

def aggregate_scores(
    qas: List[dict],
    score_key: str = "score",
) -> dict:
    """Compute per-category and overall average scores from a flat list of QA dicts.

    Each dict is expected to have at minimum 'category' and the key named
    by `score_key` (a float).  Mirrors the aggregation loop inside
    analyze_aggr_acc in evaluation_stats.py.

    Returns:
        {
          "per_category": {cat_id: {"count": int, "mean": float}, ...},
          "overall":      {"count": int, "mean": float},
        }
    """
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)

    for qa in qas:
        category = qa.get("category")
        value = qa.get(score_key)
        if category is None or value is None:
            continue
        counts[category] += 1
        totals[category] += float(value)

    per_category = {}
    for cat in CATEGORY_ORDER:
        n = counts[cat]
        per_category[cat] = {
            "count": n,
            "mean": round(totals[cat] / n, 4) if n > 0 else 0.0,
        }

    total_n = sum(counts.values())
    total_sum = sum(totals.values())
    overall = {
        "count": total_n,
        "mean": round(total_sum / total_n, 4) if total_n > 0 else 0.0,
    }

    return {"per_category": per_category, "overall": overall}


def log_stats(stats: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    logger.info("%sper-category scores:", prefix)
    for cat in CATEGORY_ORDER:
        entry = stats["per_category"].get(cat, {"count": 0, "mean": 0.0})
        name = CATEGORY_NAMES.get(cat, str(cat))
        logger.info(
            "  cat %d (%-12s): %3d questions — %.4f",
            cat, name, entry["count"], entry["mean"],
        )
    ov = stats["overall"]
    logger.info("  Overall              : %3d questions — %.4f", ov["count"], ov["mean"])


def format_stats_table(stats: dict) -> str:
    """Return a human-readable stats table as a plain string."""
    
    lines = []
    for cat in CATEGORY_ORDER:
        entry = stats["per_category"].get(cat, {"count": 0, "mean": 0.0})
        name = CATEGORY_NAMES.get(cat, str(cat))
        lines.append(
            f"  cat {cat} ({name:<12}): {entry['count']:3d} questions — {entry['mean']:.4f}"
        )
    ov = stats["overall"]
    lines.append(f"  Overall              : {ov['count']:3d} questions — {ov['mean']:.4f}")
    return "\n".join(lines)
