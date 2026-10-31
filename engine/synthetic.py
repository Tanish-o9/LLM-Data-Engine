"""Optional synthetic data generation module using the Groq API.

Generates high-quality instruction-response pairs based on seed dataset examples.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from config import SyntheticConfig
from engine.validator import validate_and_normalize
from engine.quality_filter import filter_quality

load_dotenv()


def is_groq_available() -> bool:
    """Checks if Groq API key is present in environment."""
    return bool(os.environ.get("GROQ_API_KEY"))


def generate_synthetic_data(
    seed_records: List[Dict[str, Any]],
    num_samples: int = 5,
    config: Optional[SyntheticConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generates synthetic instruction-response pairs using Groq.

    Returns:
        (valid_generated_records, generation_report)
    """
    if config is None:
        config = SyntheticConfig()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return [], {
            "status": "skipped",
            "reason": "GROQ_API_KEY not set. Synthetic generation was skipped.",
            "generated_count": 0,
        }

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except Exception as e:
        return [], {
            "status": "failed",
            "reason": f"Failed to initialize Groq client: {e}",
            "generated_count": 0,
        }

    # Select up to 3 seed examples for few-shot context
    few_shots = seed_records[:3] if seed_records else []
    seed_context = ""
    for idx, s in enumerate(few_shots, 1):
        seed_context += f"Example {idx}:\nInstruction: {s.get('instruction')}\nResponse: {s.get('response')}\n\n"

    system_prompt = (
        "You are an expert synthetic dataset generator for LLM fine-tuning. "
        "Create novel, informative, high-quality instruction-response pairs following the style and domain of the examples provided. "
        "Respond ONLY with a valid JSON array of objects, where each object has strictly two keys: 'instruction' and 'response'. "
        "Do not include markdown code block backticks, explanatory text, or conversational preambles."
    )

    user_prompt = (
        f"Generate exactly {num_samples} distinct, realistic, high-quality training pairs.\n\n"
        f"Context / Seed Examples:\n{seed_context}\n"
        "Output JSON format:\n"
        '[{"instruction": "...", "response": "..."}]'
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=config.groq_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        response_content = chat_completion.choices[0].message.content.strip()

        # Clean JSON if wrapped in markdown code fence
        if response_content.startswith("```"):
            response_content = re.sub(r"^```(?:json)?\n?", "", response_content)
            response_content = re.sub(r"\n?```$", "", response_content)

        raw_parsed = json.loads(response_content)
        if not isinstance(raw_parsed, list):
            raw_parsed = [raw_parsed]

        # Validate structure and run basic quality filtering on generated data
        validated, _ = validate_and_normalize(raw_parsed)
        quality_passed, _ = filter_quality(validated)

        # Tag as synthetic
        for item in quality_passed:
            item["source"] = "synthetic_groq"

        return quality_passed, {
            "status": "success",
            "raw_generated": len(raw_parsed),
            "valid_and_passed": len(quality_passed),
            "model_used": config.groq_model,
        }

    except Exception as e:
        return [], {
            "status": "error",
            "reason": str(e),
            "generated_count": 0,
        }
