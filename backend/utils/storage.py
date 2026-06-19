"""Read and write evolution run files on disk.

Each run lives under runs/<timestamp>/:
  archive.json       — all nodes in the archive
  nodes/<node_id>/   — pipeline.py, metadata.json, and diff.patch (children only)
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from datetime import datetime

try:
    from archive import Archive, Node  # flat import when backend/ is on sys.path
except ModuleNotFoundError:
    from utils.archive import Archive, Node  # package import fallback


def resolve_run_dir(runs_dir: str, resume: str | None) -> str:
    """Start a new run folder, or reopen an existing one when resuming."""
    if resume:
        run_dir = resume if os.path.isabs(resume) else os.path.join(runs_dir, resume)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_dir, ts)
    os.makedirs(os.path.join(run_dir, "nodes"), exist_ok=True)
    return run_dir


def _root_meta(run_dir: str) -> str:
    return os.path.join(run_dir, "nodes", "root", "metadata.json")


def root_exists(run_dir: str) -> bool:
    """True when root was already seeded and finished at least one eval question."""
    path = _root_meta(run_dir)
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
        # Support both old (flat total) and new (metadata.total) layouts
        total = d.get("metadata", {}).get("total") or d.get("total", 0)
        return total > 0


def load_root_node(run_dir: str) -> Node:
    with open(_root_meta(run_dir), encoding="utf-8") as f:
        return Node.from_dict(json.load(f))


def max_child_counter(archive: Archive) -> int:
    """Highest cN suffix in gen*_cN node ids — used to resume child numbering."""
    max_n = 0
    for node in archive.all_nodes():
        if m := re.fullmatch(r"gen\d+_c(\d+)", node.node_id):
            max_n = max(max_n, int(m.group(1)))
    return max_n


def _node_dir(run_dir: str, node_id: str) -> str:
    path = os.path.join(run_dir, "nodes", node_id)
    os.makedirs(path, exist_ok=True)
    return path


def seed_root_node(run_dir: str, seed_pipeline_path: str) -> str:
    """Copy the seed pipeline into nodes/root/. Returns the new pipeline.py path."""
    dest = os.path.join(_node_dir(run_dir, "root"), "pipeline.py")
    shutil.copy2(seed_pipeline_path, dest)
    return dest


def pipeline_path(run_dir: str, node_id: str) -> str:
    return os.path.join(run_dir, "nodes", node_id, "pipeline.py")


def read_pipeline_code(run_dir: str, node_id: str) -> str:
    with open(pipeline_path(run_dir, node_id), encoding="utf-8") as f:
        return f.read()


def write_node(
    run_dir: str,
    node: Node,
    code: str,
    parent_code: str | None,
) -> None:
    """Save pipeline.py, metadata.json, and a diff.patch against the parent."""
    ndir = _node_dir(run_dir, node.node_id)

    with open(os.path.join(ndir, "pipeline.py"), "w", encoding="utf-8") as f:
        f.write(code)

    if parent_code is not None:
        diff = difflib.unified_diff(
            parent_code.splitlines(keepends=True),
            code.splitlines(keepends=True),
            fromfile=f"parent ({node.parent_id})/pipeline.py",
            tofile=f"{node.node_id}/pipeline.py",
        )
        with open(os.path.join(ndir, "diff.patch"), "w", encoding="utf-8") as f:
            f.writelines(diff)

    with open(os.path.join(ndir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(node.to_dict(), f, indent=2)


def save_archive(run_dir: str, archive: Archive) -> None:
    path = os.path.join(run_dir, "archive.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([n.to_dict() for n in archive.all_nodes()], f, indent=2)


def load_archive(
    run_dir: str,
    dgm_lambda: float,
    dgm_alpha0: float,
    min_parent_accuracy: float = 0.2,
) -> Archive:
    archive = Archive(
        dgm_lambda=dgm_lambda,
        dgm_alpha0=dgm_alpha0,
        min_parent_accuracy=min_parent_accuracy,
    )
    path = os.path.join(run_dir, "archive.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for d in json.load(f):
                archive.add(Node.from_dict(d))
    return archive
