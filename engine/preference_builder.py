"""DPO (Direct Preference Optimization) & RLHF Preference Dataset Builder.

Constructs paired preference datasets containing 'instruction', 'chosen',
and 'rejected' responses for fine-tuning reward models or DPO training.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.quality_filter import calculate_quality_score, compute_word_repetition_ratio


def build_dpo_pairs_from_candidates(
    instruction: str,
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Takes multiple candidate responses for a single instruction and pairs

    the highest-scoring response as 'chosen' and the lowest-scoring response as 'rejected'.
    """
    if len(candidates) < 2:
        return None

    scored = []
    for c in candidates:
        text = c.get("response", "")
        score = c.get("quality_score", calculate_quality_score(instruction, text))
        scored.append((score, text, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_resp, _ = scored[0]
    worst_score, worst_resp, _ = scored[-1]

    # Require a meaningful quality margin between chosen and rejected
    if best_score <= worst_score:
        return None

    return {
        "instruction": instruction,
        "chosen": best_resp,
        "rejected": worst_resp,
        "chosen_score": round(best_score, 4),
        "rejected_score": round(worst_score, 4),
        "score_delta": round(best_score - worst_score, 4),
    }


def synthesize_negative_variant(
    instruction: str,
    gold_response: str,
    perturbation_type: str = "repetition",
) -> Dict[str, Any]:
    """Synthesizes an aligned 'rejected' response from a high-quality response

    to enable DPO dataset creation even when only single-turn gold responses are available.

    Supported perturbation types:
    - 'repetition': Appends redundant, repetitive loops degrading quality.
    - 'truncation': Cuts off response mid-sentence at a conjunction.
    - 'unhelpful': Replaces with an evasive/low-effort refusal.
    """
    if perturbation_type == "repetition":
        words = gold_response.split()
        if len(words) >= 4:
            loop_phrase = " ".join(words[-4:])
            rejected_text = f"{gold_response} {loop_phrase} {loop_phrase} {loop_phrase} and so on."
        else:
            rejected_text = f"{gold_response} {gold_response} {gold_response}"
        reason = "Degraded by excessive repetitive token loops."

    elif perturbation_type == "truncation":
        words = gold_response.split()
        half_len = max(3, len(words) // 2)
        truncated_body = " ".join(words[:half_len])
        rejected_text = f"{truncated_body} because and therefore..."
        reason = "Degraded by incomplete mid-sentence truncation."

    else:  # unhelpful / low-effort
        rejected_text = "I don't know the answer. Look it up online yourself."
        reason = "Degraded by dismissive low-effort refusal."

    chosen_score = calculate_quality_score(instruction, gold_response)
    rejected_score = calculate_quality_score(instruction, rejected_text)

    return {
        "instruction": instruction,
        "chosen": gold_response,
        "rejected": rejected_text,
        "chosen_score": round(chosen_score, 4),
        "rejected_score": round(rejected_score, 4),
        "score_delta": round(max(0.0, chosen_score - rejected_score), 4),
        "rejected_reason": reason,
        "perturbation_type": perturbation_type,
    }


def generate_dpo_dataset(
    records: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    perturbation_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """Generates a complete DPO preference dataset from input instruction records.

    If candidate alternatives exist, pairs them; otherwise synthesizes
    principled negatives based on common failure modes.
    """
    dpo_records = []
    perturbation_types = ["repetition", "truncation", "unhelpful"]

    for i, record in enumerate(records):
        instruction = record.get("instruction", "").strip()
        response = record.get("response", "").strip()

        if not instruction or not response:
            continue

        ptype = perturbation_types[i % len(perturbation_types)]
        pair = synthesize_negative_variant(
            instruction=instruction,
            gold_response=response,
            perturbation_type=ptype,
        )
        dpo_records.append(pair)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for item in dpo_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return dpo_records

