"""End-to-end data quality pipeline orchestrator.

Executes:
Loading -> Validation -> Deduplication -> Quality Filtering -> PII Redaction -> Decontamination -> Optional Synthetic Generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    DeduplicationConfig,
    DecontaminationConfig,
    QualityConfig,
    SafetyConfig,
    SyntheticConfig,
)
from engine.loader import load_dataset, get_dataset_stats
from engine.validator import validate_and_normalize
from engine.deduplicator import deduplicate_dataset
from engine.semantic_dedup import deduplicate_semantic
from engine.quality_filter import filter_quality
from engine.safety_filter import filter_safety
from engine.pii_scrubber import scrub_pii
from engine.decontaminator import decontaminate_dataset
from engine.synthetic import generate_synthetic_data
from engine.preference_builder import generate_dpo_dataset
from engine.complexity_scorer import annotate_dataset_with_complexity


def run_pipeline(
    raw_source: Any,
    eval_source: Optional[Any] = None,
    quality_cfg: Optional[QualityConfig] = None,
    dedup_cfg: Optional[DeduplicationConfig] = None,
    safety_cfg: Optional[SafetyConfig] = None,
    decontam_cfg: Optional[DecontaminationConfig] = None,
    synth_cfg: Optional[SyntheticConfig] = None,
    file_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the full LLM data preparation pipeline and returns cleaned data and audit reports."""
    # 1. Dataset Loading (supports JSON, JSONL, CSV & Multi-turn Chat formats)
    raw_data = load_dataset(raw_source, file_format=file_format)
    eval_data = load_dataset(eval_source) if eval_source else []
    raw_stats = get_dataset_stats(raw_data)

    # 2. Validation & Normalization
    valid_data, invalid_records = validate_and_normalize(raw_data)

    # 3. Exact & Near Deduplication (MinHash LSH)
    unique_data, exact_dups, near_dups = deduplicate_dataset(valid_data, config=dedup_cfg)

    # 4. Neural Semantic Deduplication (Sentence Transformers)
    sem_unique, sem_dups = deduplicate_semantic(unique_data, config=dedup_cfg)

    # 5. Quality Filtering & Rule-Based Scoring
    quality_passed, low_quality_records = filter_quality(sem_unique, config=quality_cfg)

    # 6. Safety & Adversarial Prompt Injection Filter
    safe_data, unsafe_records = filter_safety(quality_passed, config=safety_cfg)

    # 7. PII Redaction
    pii_scrubbed, pii_report = scrub_pii(safe_data)

    # 8. Decontamination against Evaluation dataset
    decontaminated, contaminated_records = decontaminate_dataset(
        pii_scrubbed, eval_data, config=decontam_cfg
    )

    # 9. Optional Synthetic Generation
    synthetic_records = []
    synthetic_report = {"status": "disabled", "generated_count": 0}
    if synth_cfg and synth_cfg.enabled:
        synthetic_records, synthetic_report = generate_synthetic_data(
            decontaminated,
            num_samples=synth_cfg.num_samples_to_generate,
            config=synth_cfg,
        )

    # Cleaned dataset (without synthetic)
    cleaned_dataset = decontaminated

    # 10. Complexity & Difficulty Scoring (Curriculum Learning)
    cleaned_annotated, complexity_summary = annotate_dataset_with_complexity(cleaned_dataset)

    # 11. DPO Preference Dataset Generation (Direct Preference Optimization)
    dpo_pairs = generate_dpo_dataset(cleaned_dataset)

    # Cleaned + Synthetic dataset
    cleaned_plus_synthetic = list(cleaned_annotated) + list(synthetic_records)

    cleaned_stats = get_dataset_stats(cleaned_annotated)
    cleaned_plus_synth_stats = get_dataset_stats(cleaned_plus_synthetic)

    # Build comprehensive audit report
    report = {
        "summary": {
            "raw_total": len(raw_data),
            "invalid_removed": len(invalid_records),
            "exact_duplicates_removed": len(exact_dups),
            "near_duplicates_removed": len(near_dups),
            "semantic_duplicates_removed": len(sem_dups),
            "low_quality_removed": len(low_quality_records),
            "unsafe_records_removed": len(unsafe_records),
            "pii_records_scrubbed": pii_report["records_modified"],
            "contaminated_removed": len(contaminated_records),
            "cleaned_total": len(cleaned_annotated),
            "synthetic_added": len(synthetic_records),
            "cleaned_plus_synth_total": len(cleaned_plus_synthetic),
            "dpo_pairs_generated": len(dpo_pairs),
        },
        "stats": {
            "raw": raw_stats,
            "cleaned": cleaned_stats,
            "cleaned_plus_synthetic": cleaned_plus_synth_stats,
            "complexity_summary": complexity_summary,
        },
        "details": {
            "invalid_records": invalid_records,
            "exact_duplicates": exact_dups,
            "near_duplicates": near_dups,
            "semantic_duplicates": sem_dups,
            "low_quality_records": low_quality_records,
            "unsafe_records": unsafe_records,
            "pii_audit": pii_report,
            "contaminated_records": contaminated_records,
            "synthetic_report": synthetic_report,
            "complexity_breakdown": complexity_summary,
        },
        "datasets": {
            "raw": raw_data,
            "cleaned": cleaned_annotated,
            "cleaned_plus_synthetic": cleaned_plus_synthetic,
            "dpo_preference": dpo_pairs,
            "eval": eval_data,
        },
    }

    return report


def save_pipeline_output(result: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, str]:
    """Saves cleaned datasets and JSON report to disk."""
    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_dir / "cleaned_dataset.jsonl"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for row in result["datasets"]["cleaned"]:
            f.write(json.dumps(row) + "\n")

    cleaned_synth_path = out_dir / "cleaned_plus_synthetic.jsonl"
    with open(cleaned_synth_path, "w", encoding="utf-8") as f:
        for row in result["datasets"]["cleaned_plus_synthetic"]:
            f.write(json.dumps(row) + "\n")

    dpo_path = out_dir / "dpo_dataset.jsonl"
    with open(dpo_path, "w", encoding="utf-8") as f:
        for row in result["datasets"].get("dpo_preference", []):
            f.write(json.dumps(row) + "\n")

    report_path = out_dir / "pipeline_report.json"
    serializable_report = {
        "summary": result["summary"],
        "stats": result["stats"],
        "details": {
            k: v for k, v in result["details"].items()
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2)

    return {
        "cleaned_path": str(cleaned_path),
        "cleaned_synth_path": str(cleaned_synth_path),
        "dpo_path": str(dpo_path),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    raw_sample = DATA_DIR / "raw_sample.jsonl"
    eval_sample = DATA_DIR / "eval_sample.jsonl"

    print("Running LLM Data Engine preprocessing pipeline on sample data...")
    res = run_pipeline(raw_sample, eval_sample)
    saved = save_pipeline_output(res)
    print("Pipeline execution complete! Summary:")
    print(json.dumps(res["summary"], indent=2))
    print(f"Files saved to: {saved['cleaned_path']}")

