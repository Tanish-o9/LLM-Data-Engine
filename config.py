"""Central configuration for LLM Data Engine.

Defines default hyperparameters, paths, filtering thresholds, and model settings.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QualityConfig:
    min_instruction_chars: int = 10
    max_instruction_chars: int = 1500
    min_response_chars: int = 15
    max_response_chars: int = 4000
    min_instruction_words: int = 3
    min_response_words: int = 4
    max_word_repetition_ratio: float = 0.25  # Max ratio of most common word
    max_char_repetition_count: int = 5       # E.g. "aaaaa" or "!!!!!"
    min_quality_score: float = 0.50          # Overall score threshold (0.0 to 1.0)


@dataclass
class DeduplicationConfig:
    exact_hash_fields: List[str] = field(default_factory=lambda: ["instruction", "response"])
    minhash_threshold: float = 0.70          # Jaccard similarity threshold for near-duplicates
    minhash_num_perm: int = 128              # Number of permutations for MinHash
    minhash_shingle_size: int = 2            # Word bigrams for near-duplicate shingles
    enable_semantic: bool = True             # Enable neural embedding semantic deduplication
    semantic_threshold: float = 0.80         # Cosine similarity threshold for semantic duplicates
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SafetyConfig:
    enabled: bool = True
    block_prompt_injections: bool = True
    block_toxic_content: bool = True


@dataclass
class JudgeConfig:
    enabled: bool = True
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2


@dataclass
class DecontaminationConfig:
    ngram_size: int = 8                      # n-gram size for overlap detection with eval set
    ngram_overlap_threshold: float = 0.70    # Overlap ratio threshold


@dataclass
class SyntheticConfig:
    enabled: bool = False
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1024
    num_samples_to_generate: int = 10


@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    output_dir: str = str(EXPERIMENTS_DIR)
    max_seq_length: int = 256
    learning_rate: float = 5e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    warmup_steps: int = 5
    weight_decay: float = 0.01
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    use_cuda_if_available: bool = True
