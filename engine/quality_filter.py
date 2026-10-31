"""Quality filtering module with rule-based heuristics and explainable quality scoring.
"""

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Tuple

from config import QualityConfig


def compute_word_repetition_ratio(text: str) -> float:
    """Calculates the ratio of the most frequent word to total words."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    if not words:
        return 0.0
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count / len(words)


def has_character_repetition(text: str, max_repeat: int = 5) -> bool:
    """Detects repeated single characters like 'aaaaaa' or '!!!!!!'."""
    pattern = rf"(.)\1{{{max_repeat - 1},}}"
    return bool(re.search(pattern, text))


def is_response_incomplete(response: str) -> bool:
    """Checks whether a response appears truncated mid-sentence."""
    trimmed = response.strip()
    if not trimmed:
        return True

    # Truncated endings
    if trimmed.endswith(("...", "…", "etc.", "http://", "https://", "TODO")):
        return True

    # Hanging conjunctions or prepositions at the end
    last_word = re.findall(r"\b[a-zA-Z]+\b", trimmed.lower())
    if last_word:
        hanging_words = {"and", "or", "because", "but", "so", "with", "that", "which", "if", "then"}
        if last_word[-1] in hanging_words:
            return True

    # Check terminal punctuation (allow closing quotes or brackets after punctuation)
    valid_endings = (".", "!", "?", '."', '!"', '?"', ".)", "!)", "?)", "```", ">", ":", ";")
    if not any(trimmed.endswith(end) for end in valid_endings):
        # If response has more than 15 words and ends without any terminal punctuation, likely cut off
        words = trimmed.split()
        if len(words) >= 15:
            return True

    return False


def calculate_quality_score(instruction: str, response: str) -> float:
    """Computes an explainable quality score between 0.0 and 1.0 based on:
    - Length balance
    - Vocabulary richness
    - Structural completeness
    - Low repetition
    """
    inst_words = instruction.split()
    resp_words = response.split()

    if not inst_words or not resp_words:
        return 0.0

    # 1. Length suitability (0.0 to 1.0)
    len_score = 1.0
    if len(inst_words) < 3 or len(inst_words) > 250:
        len_score -= 0.3
    if len(resp_words) < 5 or len(resp_words) > 600:
        len_score -= 0.3
    len_score = max(0.0, len_score)

    # 2. Vocabulary diversity (unique words / total words)
    unique_resp_ratio = len(set(w.lower() for w in resp_words)) / max(len(resp_words), 1)
    vocab_score = min(1.0, unique_resp_ratio * 1.2)

    # 3. Structure and completeness
    structure_score = 1.0
    if is_response_incomplete(response):
        structure_score -= 0.4
    if not instruction.strip().endswith(("?", ".", "!")):
        structure_score -= 0.1
    structure_score = max(0.0, structure_score)

    # 4. Repetition penalty
    rep_ratio = compute_word_repetition_ratio(response)
    rep_penalty = 1.0
    if rep_ratio > 0.20:
        rep_penalty -= (rep_ratio - 0.20) * 2.0
    if has_character_repetition(response, max_repeat=5):
        rep_penalty -= 0.3
    rep_penalty = max(0.0, min(1.0, rep_penalty))

    # Weighted aggregate
    final_score = (
        0.25 * len_score +
        0.25 * vocab_score +
        0.25 * structure_score +
        0.25 * rep_penalty
    )
    return round(float(final_score), 3)


def filter_quality(
    records: List[Dict[str, Any]],
    config: Optional[QualityConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filters records based on rule-based heuristics and assigns quality scores.

    Returns:
        (passed_records, filtered_records)
    """
    if config is None:
        config = QualityConfig()

    passed = []
    filtered = []

    for idx, row in enumerate(records):
        inst = row.get("instruction", "")
        resp = row.get("response", "")

        inst_len = len(inst)
        resp_len = len(resp)
        inst_w_len = len(inst.split())
        resp_w_len = len(resp.split())

        reasons = []

        # Length rules
        if inst_len < config.min_instruction_chars:
            reasons.append(f"Instruction too short ({inst_len} chars < {config.min_instruction_chars})")
        if inst_len > config.max_instruction_chars:
            reasons.append(f"Instruction too long ({inst_len} chars > {config.max_instruction_chars})")
        if resp_len < config.min_response_chars:
            reasons.append(f"Response too short ({resp_len} chars < {config.min_response_chars})")
        if resp_len > config.max_response_chars:
            reasons.append(f"Response too long ({resp_len} chars > {config.max_response_chars})")
        if inst_w_len < config.min_instruction_words:
            reasons.append(f"Instruction has too few words ({inst_w_len} < {config.min_instruction_words})")
        if resp_w_len < config.min_response_words:
            reasons.append(f"Response has too few words ({resp_w_len} < {config.min_response_words})")

        # Repetition rules
        rep_ratio = compute_word_repetition_ratio(resp)
        if rep_ratio > config.max_word_repetition_ratio:
            reasons.append(f"High word repetition in response ({rep_ratio:.2f} > {config.max_word_repetition_ratio})")

        if has_character_repetition(resp, max_repeat=config.max_char_repetition_count):
            reasons.append("Character repetition detected in response")

        # Incompleteness rule
        if is_response_incomplete(resp):
            reasons.append("Incomplete or truncated response detected")

        # Overall quality score
        score = calculate_quality_score(inst, resp)
        if score < config.min_quality_score:
            reasons.append(f"Quality score {score:.2f} below threshold {config.min_quality_score:.2f}")

        row_with_score = dict(row)
        row_with_score["quality_score"] = score

        if reasons:
            filtered.append({
                "index": idx,
                "record": row_with_score,
                "reasons": reasons,
                "score": score,
            })
        else:
            passed.append(row_with_score)

    return passed, filtered

