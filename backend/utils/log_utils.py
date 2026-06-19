"""Configure stdout + file logging for a run."""

import logging
import os
from datetime import datetime


def setup_run_logging(run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    for h in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    root.info("Log: %s", path)
    
    return path