"""Data decontamination module.

Prevents benchmark test set contamination by detecting and removing exact
and n-gram overlaps between training data and held-out evaluation datasets.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import DecontaminationConfig


def extract_ngrams(text: str, n: int = 8) -> Set[str]:
    """Extracts lowercase word n-grams from text."""
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def calculate_ngram_overlap(train_ngrams: Set[str], eval_ngrams: Set[str]) -> float:
    """Calculates ratio of training n-grams that appear in eval n-grams."""
    if not train_ngrams or not eval_ngrams:
        return 0.0
    common = train_ngrams.intersection(eval_ngrams)
    return len(common) / len(train_ngrams)


def decontaminate_dataset(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    config: Optional[DecontaminationConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filters training records that overlap with held-out evaluation records.

    Returns:
        (clean_train_records, contaminated_records)
    """
    if config is None:
        config = DecontaminationConfig()

    if not eval_records:
        return train_records, []

    # Pre-index evaluation data
    eval_exact_inst: Set[str] = set()
    eval_exact_resp: Set[str] = set()
    eval_all_ngrams: Set[str] = set()

    for r in eval_records:
        inst = r.get("instruction", "").strip().lower()
        resp = r.get("response", "").strip().lower()
        if inst:
            eval_exact_inst.add(inst)
            eval_all_ngrams.update(extract_ngrams(inst, n=config.ngram_size))
        if resp:
            eval_exact_resp.add(resp)
            eval_all_ngrams.update(extract_ngrams(resp, n=config.ngram_size))

    clean_records = []
    contaminated_records = []

    for idx, train_row in enumerate(train_records):
        t_inst = train_row.get("instruction", "").strip()
        t_resp = train_row.get("response", "").strip()

        # 1. Exact match check
        if t_inst.lower() in eval_exact_inst:
            contaminated_records.append({
                "index": idx,
                "record": train_row,
                "reason": "Exact instruction match with evaluation set",
            })
            continue

        if t_resp.lower() in eval_exact_resp:
            contaminated_records.append({
                "index": idx,
                "record": train_row,
                "reason": "Exact response match with evaluation set",
            })
            continue

        # 2. N-gram overlap check
        combined = f"{t_inst} {t_resp}"
        train_ngrams = extract_ngrams(combined, n=config.ngram_size)
        overlap = calculate_ngram_overlap(train_ngrams, eval_all_ngrams)

        if overlap >= config.ngram_overlap_threshold:
            contaminated_records.append({
                "index": idx,
                "record": train_row,
                "overlap_ratio": round(overlap, 3),
                "reason": f"{config.ngram_size}-gram overlap ratio ({overlap:.2f}) >= {config.ngram_overlap_threshold}",
            })
            continue

        clean_records.append(train_row)

    return clean_records, contaminated_records

