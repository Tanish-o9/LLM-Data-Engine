"""Cognitive Difficulty & Complexity Scorer for Curriculum Learning.

Analyzes instructions to evaluate reasoning depth, technical/code density,
and structural complexity, assigning difficulty tiers (Easy, Medium, Hard).
"""

import math
import re
from typing import Any, Dict, List, Tuple


# Heuristic keywords associated with higher-order cognitive reasoning
REASONING_KEYWORDS = {
    "why", "prove", "derive", "calculate", "explain why", "analyze", "synthesize",
    "compare and contrast", "evaluate", "justify", "step by step", "algorithm",
    "pseudocode", "optimize", "implications", "tradeoff", "trade-offs", "theorem"
}

CODE_KEYWORDS = {
    "def ", "class ", "import ", "function", "lambda", "return ", "async ", "await ",
    "void ", "int ", "public ", "SELECT ", "WHERE ", "JOIN ", "CREATE TABLE"
}


def estimate_syllables(word: str) -> int:
    """Estimates the number of syllables in an English word."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Count vowel groups
    vowel_runs = re.findall(r"[aeiouy]+", word)
    count = len(vowel_runs)
    if word.endswith("e") and not word.endswith("le"):
        count = max(1, count - 1)
    return max(1, count)


def compute_flesch_kincaid_grade(text: str) -> float:
    """Computes an approximate Flesch-Kincaid Grade Level for readability.

    Formula: 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(1, len(sentences))

    words = re.findall(r"\b[a-zA-Z0-9_]+\b", text)
    num_words = max(1, len(words))

    total_syllables = sum(estimate_syllables(w) for w in words)

    avg_sentence_len = num_words / num_sentences
    avg_syllables_per_word = total_syllables / num_words

    grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables_per_word - 15.59
    return round(max(1.0, min(20.0, grade)), 2)


def score_prompt_complexity(instruction: str, response: str = "") -> Dict[str, Any]:
    """Calculates multidimensional complexity metrics for an instruction-response pair.

    Returns:
        dict containing complexity_score (0.0 - 1.0), difficulty_tier ('Easy', 'Medium', 'Hard'),
        and explanatory breakdown features.
    """
    inst_lower = instruction.lower()
    combined_text = f"{instruction} {response}"

    # 1. Length features
    words = instruction.split()
    num_words = len(words)
    resp_words = len(response.split())

    # 2. Reasoning indicators
    reasoning_hits = sum(1 for kw in REASONING_KEYWORDS if kw in inst_lower)
    has_code = any(kw in combined_text for kw in CODE_KEYWORDS)

    # 3. Readability grade
    fk_grade = compute_flesch_kincaid_grade(instruction)

    # 4. Synthesize composite complexity score (0.0 - 1.0)
    # Length factor (up to 0.30):
    length_factor = min(0.30, (num_words / 100.0) * 0.30)

    # Reasoning factor (up to 0.35):
    reasoning_factor = min(0.35, reasoning_hits * 0.12)

    # Technical / Code factor (up to 0.20):
    code_factor = 0.20 if has_code else 0.0

    # Readability / vocabulary factor (up to 0.15):
    grade_factor = min(0.15, (fk_grade / 16.0) * 0.15)

    raw_score = length_factor + reasoning_factor + code_factor + grade_factor
    complexity_score = round(max(0.05, min(1.0, raw_score)), 4)

    # Determine Tier
    if complexity_score < 0.35:
        difficulty_tier = "Easy"
    elif complexity_score < 0.65:
        difficulty_tier = "Medium"
    else:
        difficulty_tier = "Hard"

    return {
        "complexity_score": complexity_score,
        "difficulty_tier": difficulty_tier,
        "flesch_kincaid_grade": fk_grade,
        "instruction_words": num_words,
        "response_words": resp_words,
        "reasoning_signals_count": reasoning_hits,
        "contains_code_or_math": has_code,
    }


def annotate_dataset_with_complexity(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Annotates each record with complexity scoring and returns the distribution summary."""
    annotated = []
    tier_counts = {"Easy": 0, "Medium": 0, "Hard": 0}
    scores = []

    for r in records:
        inst = r.get("instruction", "")
        resp = r.get("response", "")
        metrics = score_prompt_complexity(inst, resp)

        tier_counts[metrics["difficulty_tier"]] += 1
        scores.append(metrics["complexity_score"])

        r_copy = dict(r)
        r_copy.update({
            "complexity_score": metrics["complexity_score"],
            "difficulty_tier": metrics["difficulty_tier"],
            "flesch_kincaid_grade": metrics["flesch_kincaid_grade"],
        })
        annotated.append(r_copy)

    total = max(1, len(records))
    summary = {
        "total_records": len(records),
        "distribution_count": tier_counts,
        "distribution_percentage": {
            k: round((v / total) * 100, 1) for k, v in tier_counts.items()
        },
        "avg_complexity_score": round(sum(scores) / total, 4) if scores else 0.0,
    }

    return annotated, summary

