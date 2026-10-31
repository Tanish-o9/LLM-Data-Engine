"""Safety and prompt injection filtering module.

Protects training datasets from adversarial attacks, jailbreaks, and toxic content.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from config import SafetyConfig


# Known prompt injection & jailbreak patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+prompts?",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"act\s+as\s+(an\s+)?unrestricted",
    r"do\s+anything\s+now\s*(\(|DAN)",
    r"bypass\s+(all\s+)?(content|safety)\s+filters?",
    r"system\s*override",
    r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)",
    r"pretend\s+you\s+have\s+no\s+(rules|guidelines|restrictions)",
    r"simulated\s+jailbreak",
]

# Patterns for offensive / harmful / toxic directives
TOXIC_PATTERNS = [
    r"\b(how\s+to\s+build\s+a\s+(bomb|weapon|explosive))\b",
    r"\b(create\s+ransomware|malware\s+script|keylogger\s+code)\b",
    r"\b(hack\s+into\s+someone's\s+account)\b",
    r"\b(steal\s+credit\s+card\s+numbers)\b",
]

COMPILED_INJECTIONS = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]
COMPILED_TOXIC = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]


def check_safety_violations(text: str) -> List[str]:
    """Scans text for prompt injections and toxic keywords."""
    violations = []
    for pattern in COMPILED_INJECTIONS:
        if pattern.search(text):
            violations.append("Adversarial prompt injection / jailbreak detected")
            break

    for pattern in COMPILED_TOXIC:
        if pattern.search(text):
            violations.append("Harmful or unsafe directive detected")
            break

    return violations


def filter_safety(
    records: List[Dict[str, Any]],
    config: Optional[SafetyConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filters unsafe instructions and responses from the dataset.

    Returns:
        (safe_records, unsafe_records)
    """
    if config is None:
        config = SafetyConfig()

    if not config.enabled:
        return records, []

    safe_records = []
    unsafe_records = []

    for idx, row in enumerate(records):
        inst = row.get("instruction", "")
        resp = row.get("response", "")
        combined = f"{inst} {resp}"

        violations = check_safety_violations(combined)

        if violations:
            unsafe_records.append({
                "index": idx,
                "record": row,
                "reasons": violations,
            })
        else:
            safe_records.append(row)

    return safe_records, unsafe_records

