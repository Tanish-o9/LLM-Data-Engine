"""Three-way fine-tuning and evaluation experiment runner.

Compares:
1. Raw dataset
2. Cleaned dataset
3. Cleaned + synthetic dataset
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import DATA_DIR, EXPERIMENTS_DIR, JudgeConfig, SyntheticConfig, TrainingConfig
from pipeline import run_pipeline
from training.trainer import train_lora_model
from training.evaluator import evaluate_model, save_experiment_results
from training.judge import judge_model_generations


def run_three_way_experiments(
    raw_source: Optional[Path] = None,
    eval_source: Optional[Path] = None,
    train_config: Optional[TrainingConfig] = None,
    synthetic_config: Optional[SyntheticConfig] = None,
    judge_config: Optional[JudgeConfig] = None,
) -> List[Dict[str, Any]]:
    """Runs data cleaning, trains 3 LoRA models, evaluates on held-out set, and saves results."""
    raw_path = raw_source or (DATA_DIR / "raw_sample.jsonl")
    eval_path = eval_source or (DATA_DIR / "eval_sample.jsonl")

    if train_config is None:
        train_config = TrainingConfig()
    if judge_config is None:
        judge_config = JudgeConfig()

    print("=" * 60)
    print("STEP 1: Executing Data Preparation Pipeline")
    print("=" * 60)
    pipeline_res = run_pipeline(
        raw_source=raw_path,
        eval_source=eval_path,
        synth_cfg=synthetic_config,
    )

    raw_data = pipeline_res["datasets"]["raw"]
    cleaned_data = pipeline_res["datasets"]["cleaned"]
    cleaned_plus_synth = pipeline_res["datasets"]["cleaned_plus_synthetic"]
    eval_data = pipeline_res["datasets"]["eval"]

    print(f"Dataset Counts -> Raw: {len(raw_data)}, Cleaned: {len(cleaned_data)}, Cleaned+Synth: {len(cleaned_plus_synth)}, Eval: {len(eval_data)}")

    experiments = [
        ("exp_1_raw", "Raw Dataset", raw_data),
        ("exp_2_cleaned", "Cleaned Dataset", cleaned_data),
        ("exp_3_cleaned_synthetic", "Cleaned + Synthetic Dataset", cleaned_plus_synth),
    ]

    all_results: List[Dict[str, Any]] = []

    for exp_id, label, dataset in experiments:
        print("\n" + "=" * 60)
        print(f"TRAINING: {label} ({exp_id}) with {len(dataset)} examples")
        print("=" * 60)

        # Train model
        train_stats = train_lora_model(
            train_records=dataset,
            experiment_name=exp_id,
            config=train_config,
        )
        print(f"Training completed in {train_stats['training_time_seconds']}s (Peak GPU RAM: {train_stats['peak_gpu_memory_mb']} MB, Final Loss: {train_stats['final_loss']})")

        # Evaluate model on identical held-out test set
        print(f"EVALUATING: {exp_id} on {len(eval_data)} held-out samples...")
        eval_stats = evaluate_model(
            checkpoint_dir=train_stats["checkpoint_dir"],
            eval_records=eval_data,
            config=train_config,
        )
        print(f"Evaluation Loss: {eval_stats['eval_loss']} | Perplexity: {eval_stats['perplexity']}")

        combined_record = {
            "experiment_id": exp_id,
            "experiment_name": exp_id,
            "dataset_type": label,
            **train_stats,
            **eval_stats,
        }
        all_results.append(combined_record)

    # Run LLM-as-a-Judge Evaluation across all experiment completions
    print("\n" + "=" * 60)
    print("STEP 3: Running Automated LLM-as-a-Judge Evaluation")
    print("=" * 60)
    all_results = judge_model_generations(all_results, config=judge_config)
    for r in all_results:
        scores = r.get("judge_scores", {})
        print(f"[{r['dataset_type']}] Judge Overall Score: {scores.get('avg_overall', 'N/A')}/5.0 (Relevance: {scores.get('avg_relevance', 'N/A')}, Coherence: {scores.get('avg_coherence', 'N/A')})")

    # Save benchmark results
    json_path, csv_path = save_experiment_results(all_results, output_dir=EXPERIMENTS_DIR)
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS & BENCHMARKS COMPLETE")
    print(f"Results saved to:\n  JSON: {json_path}\n  CSV:  {csv_path}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    results = run_three_way_experiments()

