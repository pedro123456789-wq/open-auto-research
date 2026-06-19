"""
LOCOMO Utils.
Cannot be edited by the agent
"""
from __future__ import annotations

import re
import json
from datetime import datetime, timezone


def load_dataset(dataset_dir):
    """Load LOCOMO dataset from JSON file."""
    with open(dataset_dir, "r") as f:
        data = json.load(f)
    return data

def parse_locomo_date(date_str: str) -> datetime:
    """Parse LOCOMO date: '1:56 pm on 8 May, 2023' to datetime object."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def locomo_date_to_epoch(date_str: str) -> int | None:
    """Parse LOCOMO date: '1:56 pm on 8 May, 2023' to epoch timestamp."""
    parsed = parse_locomo_date(date_str)
    if parsed:
        return int(parsed.replace(tzinfo=timezone.utc).timestamp())
    return None


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """Extract and sort sessions chronologically."""
    session_keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    paired = []

    for key in session_keys:
        date_key = f"{key}_date_time"
        date_str = conversation.get(date_key, "")
        turns = conversation[key]
        paired.append((key, date_str, turns))

    def sort_key(item: tuple) -> tuple:
        parsed = parse_locomo_date(item[1])
        if parsed:
            return (0, parsed)
        num = int(re.search(r"\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    paired.sort(key=sort_key)
    return paired


def load_evidence_lookup(dataset_path: str) -> dict[tuple, str]:
    """Build lookup: (conv_idx, dia_id) -> formatted turn text."""

    with open(dataset_path) as f:
        data = json.load(f)
    lookup = {}
    for conv_idx, conv in enumerate(data):
        conversation = conv["conversation"]
        session_dates = {}
        
        for key in conversation:
            if key.endswith("_date_time") and key.startswith("session_"):
                session_num = key.replace("session_", "").replace("_date_time", "")
                session_dates[session_num] = conversation[key]

        for key in conversation:
            if key.startswith("session_") and not key.endswith("date_time"):
                if not isinstance(conversation[key], list):
                    continue
                for turn in conversation[key]:
                    dia_id = turn.get("dia_id", "")
                    if dia_id:
                        speaker = turn.get("speaker", "")
                        text = turn.get("text", "")
                        dia_match = re.match(r"D(\d+):", dia_id)
                        date_suffix = ""
                        if dia_match:
                            snum = dia_match.group(1)
                            sdate = session_dates.get(snum, "")
                            if sdate:
                                date_suffix = f", said on {sdate}"
                        lookup[(conv_idx, dia_id)] = f'[{dia_id}{date_suffix}] {speaker}: "{text}"'
    return lookup