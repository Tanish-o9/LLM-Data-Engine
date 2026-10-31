"""Evaluation and benchmark comparison module.

Calculates cross-entropy loss, perplexity, and generates sample completions
on the held-out evaluation dataset.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EXPERIMENTS_DIR, TrainingConfig
from training.dataset_builder import InstructionDataset, format_prompt
from training.trainer import get_target_device


def evaluate_model(
    checkpoint_dir: str,
    eval_records: List[Dict[str, Any]],
    config: Optional[TrainingConfig] = None,
    sample_prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluates a trained model checkpoint on the held-out dataset.

    Returns:
        dict containing eval_loss, perplexity, and sample generations.
    """
    if config is None:
        config = TrainingConfig()

    device = get_target_device(config)
    ckpt_path = Path(checkpoint_dir)

    # 1. Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path if ckpt_path.exists() else config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if (ckpt_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    else:
        model = base_model

    model.to(device)
    model.eval()

    # 2. Evaluation Loss & Perplexity
    eval_dataset = InstructionDataset(eval_records, tokenizer, max_length=config.max_seq_length)
    if len(eval_dataset) == 0:
        return {"eval_loss": None, "perplexity": None, "sample_generations": []}

    dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += float(loss.item())
                total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap exponent to avoid overflow

    # 3. Sample Generations for Qualitative Inspection
    test_prompts = sample_prompts or [
        "What is the primary role of an activation function in neural networks?",
        "Explain the concept of learning rate in optimization.",
    ]

    sample_generations = []
    with torch.no_grad():
        for prompt in test_prompts:
            formatted = format_prompt(prompt)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Extract just the response part after "### Response:\n"
            if "### Response:\n" in generated_full:
                generated_resp = generated_full.split("### Response:\n", 1)[1].strip()
            else:
                generated_resp = generated_full.strip()

            sample_generations.append({
                "instruction": prompt,
                "generated_response": generated_resp,
            })

    return {
        "eval_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
        "eval_samples_count": len(eval_dataset),
        "sample_generations": sample_generations,
    }


def generate_single_completion(
    checkpoint_dir: str,
    instruction: str,
    config: Optional[TrainingConfig] = None,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> str:
    """Generates a completion for an interactive single instruction prompt using a saved checkpoint."""
    if config is None:
        config = TrainingConfig()

    device = get_target_device(config)
    ckpt_path = Path(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path if ckpt_path.exists() else config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if (ckpt_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    else:
        model = base_model

    model.to(device)
    model.eval()

    formatted = format_prompt(instruction)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "### Response:\n" in generated_full:
        return generated_full.split("### Response:\n", 1)[1].strip()
    return generated_full.strip()


def save_experiment_results(results: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Saves benchmark results to JSON and CSV formats."""
    out_dir = Path(output_dir or EXPERIMENTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"

    # Save JSON with full details including generation samples
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Flatten for CSV comparison
    rows = []
    for r in results:
        judge_scores = r.get("judge_scores", {})
        rows.append({
            "experiment": r.get("experiment_name"),
            "dataset_type": r.get("dataset_type"),
            "num_samples": r.get("num_samples_trained"),
            "training_time_sec": r.get("training_time_seconds"),
            "peak_gpu_mem_mb": r.get("peak_gpu_memory_mb"),
            "device": r.get("device"),
            "final_train_loss": r.get("final_loss"),
            "eval_loss": r.get("eval_loss"),
            "perplexity": r.get("perplexity"),
            "judge_relevance": judge_scores.get("avg_relevance", "N/A"),
            "judge_coherence": judge_scores.get("avg_coherence", "N/A"),
            "judge_overall": judge_scores.get("avg_overall", "N/A"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return json_path, csv_path
