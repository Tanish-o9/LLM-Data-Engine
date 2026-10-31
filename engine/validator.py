"""Data validation and text normalization module.

Detects missing or malformed records and cleans whitespace.
"""

import re
from typing import Any, Dict, List, Tuple


def normalize_text(text: str) -> str:
    """Normalizes whitespace and standardizes line breaks."""
    if not isinstance(text, str):
        return ""
    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking and unusual spaces with standard space
    text = re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]", " ", text)
    # Collapse multiple inline spaces and tabs into single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Re-join and collapse excessive consecutive blank lines (max 2)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def validate_and_normalize(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validates records for required fields, normalizes text formatting, and separates invalid records.

    Returns:
        (valid_records, invalid_records)
    """
    valid_records = []
    invalid_records = []

    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            invalid_records.append({
                "index": idx,
                "record": row,
                "reason": "Record is not a dictionary / malformed object",
            })
            continue

        raw_inst = row.get("instruction")
        raw_resp = row.get("response")

        if raw_inst is None or raw_resp is None:
            invalid_records.append({
                "index": idx,
                "record": row,
                "reason": "Missing required field ('instruction' or 'response')",
            })
            continue

        norm_inst = normalize_text(str(raw_inst))
        norm_resp = normalize_text(str(raw_resp))

        if not norm_inst:
            invalid_records.append({
                "index": idx,
                "record": row,
                "reason": "Empty or whitespace-only instruction",
            })
            continue

        if not norm_resp:
            invalid_records.append({
                "index": idx,
                "record": row,
                "reason": "Empty or whitespace-only response",
            })
            continue

        # Keep valid normalized record
        cleaned_row = dict(row)
        cleaned_row["instruction"] = norm_inst
        cleaned_row["response"] = norm_resp
        valid_records.append(cleaned_row)

    return valid_records, invalid_records

