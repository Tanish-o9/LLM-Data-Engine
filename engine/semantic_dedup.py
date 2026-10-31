"""Semantic deduplication module using neural sentence embeddings.

Detects conceptually identical instruction pairs that share meaning even when phrasing differs.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import DeduplicationConfig

_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Loads and caches the SentenceTransformer embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[Warning] Failed to load embedding model '{model_name}': {e}")
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def deduplicate_semantic(
    records: List[Dict[str, Any]],
    config: Optional[DeduplicationConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filters semantic duplicates using embedding cosine similarity.

    Returns:
        (unique_records, semantic_duplicates)
    """
    if config is None:
        config = DeduplicationConfig()

    if not config.enable_semantic or len(records) <= 1:
        return records, []

    model = get_embedding_model(config.embedding_model)
    if model is None:
        # Fallback if sentence-transformers model cannot be loaded
        return records, []

    # Extract instruction text for semantic similarity
    texts = [r.get("instruction", "").strip() for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Compute pairwise cosine similarity matrix
    # Since embeddings are normalized, dot product equals cosine similarity
    sim_matrix = np.dot(embeddings, embeddings.T)

    unique_indices = []
    dropped_indices = set()
    semantic_duplicates = []

    for i in range(len(records)):
        if i in dropped_indices:
            continue

        unique_indices.append(i)

        # Check subsequent records for high semantic similarity
        for j in range(i + 1, len(records)):
            if j in dropped_indices:
                continue

            similarity = float(sim_matrix[i, j])
            if similarity >= config.semantic_threshold:
                dropped_indices.add(j)
                semantic_duplicates.append({
                    "index": j,
                    "record": records[j],
                    "matched_with_index": i,
                    "matched_with_instruction": records[i].get("instruction"),
                    "similarity": round(similarity, 3),
                    "reason": f"Semantic duplicate (cosine similarity {similarity:.2f} >= {config.semantic_threshold})",
                })

    unique_records = [records[i] for i in unique_indices]
    return unique_records, semantic_duplicates

