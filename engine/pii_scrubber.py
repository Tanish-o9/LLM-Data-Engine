"""PII (Personally Identifiable Information) detection and redaction module.

Uses explainable, regex-based rules to detect emails, phones, SSNs, credit cards,
IP addresses, and sensitive tokens, replacing them with standardized placeholders.
"""

from collections import defaultdict
import re
from typing import Any, Dict, List, Tuple


# Explainable regular expressions for standard PII entities
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,48}|ghp_[a-zA-Z0-9]{36}|AIza[0-9A-Za-z-_]{35})\b"),
}


def scrub_text(text: str) -> Tuple[str, Dict[str, int]]:
    """Scrubs PII from a single text string, returning the scrubbed text and counts of matches."""
    if not isinstance(text, str):
        return "", {}

    counts: Dict[str, int] = defaultdict(int)
    scrubbed = text

    # Process each entity pattern
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(scrubbed)
        if matches:
            counts[pii_type] += len(matches)
            scrubbed = pattern.sub(f"[{pii_type}]", scrubbed)

    return scrubbed, dict(counts)


def scrub_pii(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scrubs PII from instructions and responses across the dataset.

    Returns:
        (scrubbed_records, pii_audit_report)
    """
    scrubbed_records = []
    total_pii_counts: Dict[str, int] = defaultdict(int)
    records_with_pii = 0

    for idx, row in enumerate(records):
        new_row = dict(row)
        inst = row.get("instruction", "")
        resp = row.get("response", "")

        scrubbed_inst, inst_counts = scrub_text(inst)
        scrubbed_resp, resp_counts = scrub_text(resp)

        has_pii = bool(inst_counts or resp_counts)
        if has_pii:
            records_with_pii += 1
            for k, v in inst_counts.items():
                total_pii_counts[k] += v
            for k, v in resp_counts.items():
                total_pii_counts[k] += v

        new_row["instruction"] = scrubbed_inst
        new_row["response"] = scrubbed_resp
        new_row["had_pii"] = has_pii
        scrubbed_records.append(new_row)

    audit_report = {
        "records_modified": records_with_pii,
        "total_replacements_by_type": dict(total_pii_counts),
        "total_pii_entities_detected": sum(total_pii_counts.values()),
    }

    return scrubbed_records, audit_report
