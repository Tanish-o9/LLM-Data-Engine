"""LLM-as-a-Judge automated evaluation module.

Uses Groq API (Llama 3.3) or heuristic evaluation to score model generation
quality across Relevance, Coherence, and Overall Quality on a 1.0 to 5.0 scale.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
from config import JudgeConfig
from engine.quality_filter import calculate_quality_score, compute_word_repetition_ratio


def evaluate_with_groq_judge(
    instruction: str,
    response: str,
    config: JudgeConfig,
) -> Optional[Dict[str, Any]]:
    """Calls Groq API to judge response quality on 1-5 scale."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = f"""You are an impartial, strict LLM evaluator. Score the quality of the following model completion based on the given user instruction.

[User Instruction]
{instruction}

[Model Response]
{response}

Evaluate on a scale of 1.0 to 5.0 (where 1.0 is completely irrelevant/broken and 5.0 is excellent):
1. relevance (1.0 to 5.0)
2. coherence (1.0 to 5.0)
3. overall (1.0 to 5.0)

Respond ONLY with a valid JSON object in this exact format:
{{"relevance": 4.5, "coherence": 4.0, "overall": 4.2, "feedback": "One sentence summary"}}
"""

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional AI judge. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            model=config.groq_model,
            temperature=config.temperature,
            max_tokens=256,
        )

        content = completion.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        parsed = json.loads(content)
        return {
            "relevance": float(parsed.get("relevance", 3.0)),
            "coherence": float(parsed.get("coherence", 3.0)),
            "overall": float(parsed.get("overall", 3.0)),
            "feedback": str(parsed.get("feedback", "Evaluated via Groq Llama 3.3.")),
            "evaluator": "groq_judge",
        }
    except Exception as e:
        print(f"[Warning] Groq judge call failed: {e}. Falling back to heuristic judge.")
        return None


def evaluate_with_heuristic_judge(instruction: str, response: str) -> Dict[str, Any]:
    """Fallback rule-based heuristic judge when Groq API key is not present."""
    base_q = calculate_quality_score(instruction, response)  # 0.0 to 1.0
    rep_ratio = compute_word_repetition_ratio(response)

    # Convert to 1.0 - 5.0 scale
    overall = round(1.0 + base_q * 4.0, 2)
    coherence = round(5.0 - (rep_ratio * 10.0), 2)
    coherence = max(1.0, min(5.0, coherence))

    # Relevance heuristic: word overlap between instruction keywords and response
    inst_words = set(re.findall(r"\b\w{4,}\b", instruction.lower()))
    resp_words = set(re.findall(r"\b\w{4,}\b", response.lower()))
    overlap_count = len(inst_words.intersection(resp_words))
    relevance = 2.5 + min(2.5, overlap_count * 0.8)
    relevance = round(max(1.0, min(5.0, relevance)), 2)

    return {
        "relevance": relevance,
        "coherence": coherence,
        "overall": overall,
        "feedback": "Evaluated via rule-based heuristic scoring (Groq API key not set).",
        "evaluator": "heuristic_judge",
    }


def judge_model_generations(
    experiment_results: List[Dict[str, Any]],
    config: Optional[JudgeConfig] = None,
) -> List[Dict[str, Any]]:
    """Evaluates sample generations for each experiment and attaches judge scores."""
    if config is None:
        config = JudgeConfig()

    updated_results = []

    for exp in experiment_results:
        exp_copy = dict(exp)
        sample_gens = exp.get("sample_generations", [])
        judged_samples = []

        scores_relevance = []
        scores_coherence = []
        scores_overall = []

        for sample in sample_gens:
            inst = sample.get("instruction", "")
            resp = sample.get("generated_response", "")

            # Try Groq Judge first, fallback to heuristic
            judge_res = None
            if config.enabled:
                judge_res = evaluate_with_groq_judge(inst, resp, config)

            if judge_res is None:
                judge_res = evaluate_with_heuristic_judge(inst, resp)

            judged_samples.append({
                **sample,
                "judge": judge_res,
            })

            scores_relevance.append(judge_res["relevance"])
            scores_coherence.append(judge_res["coherence"])
            scores_overall.append(judge_res["overall"])

        avg_relevance = round(sum(scores_relevance) / max(1, len(scores_relevance)), 2)
        avg_coherence = round(sum(scores_coherence) / max(1, len(scores_coherence)), 2)
        avg_overall = round(sum(scores_overall) / max(1, len(scores_overall)), 2)

        exp_copy["sample_generations"] = judged_samples
        exp_copy["judge_scores"] = {
            "avg_relevance": avg_relevance,
            "avg_coherence": avg_coherence,
            "avg_overall": avg_overall,
            "evaluator": judged_samples[0]["judge"]["evaluator"] if judged_samples else "none",
        }
        updated_results.append(exp_copy)

    return updated_results

