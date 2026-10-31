"""Hugging Face Hub Exporter.

Enables programmatic publishing of curated SFT datasets or DPO preference pairs
directly to the Hugging Face Hub (huggingface.co/datasets).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


def export_to_hf_hub(
    records: List[Dict[str, Any]],
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload curated dataset via LLM Data Engine",
) -> Dict[str, Any]:
    """Pushes an instruction dataset or DPO dataset to Hugging Face Datasets Hub.

    Args:
        records: List of dictionaries (SFT or DPO format).
        repo_id: Hugging Face repo ID, e.g. "Tanish-o9/curated-instruction-dataset".
        token: Hugging Face API token (falls back to HF_TOKEN env var).
        private: Whether the created repository should be private.
        commit_message: Commit message for the upload.

    Returns:
        Dictionary reporting success status, repo URL, or actionable error instructions.
    """
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not hf_token:
        return {
            "status": "error",
            "message": "Hugging Face token not found. Please set HF_TOKEN in your .env or pass token parameter.",
            "guide": "You can get an Access Token at https://huggingface.co/settings/tokens (Role: Write).",
            "preview_count": len(records),
        }

    try:
        from huggingface_hub import HfApi
        import tempfile

        api = HfApi(token=hf_token)

        # 1. Ensure repository exists on HF Hub
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )

        # 2. Write records to temporary JSONL file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file = Path(tmp_dir) / "data.jsonl"
            with open(temp_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 3. Upload file to repo
            api.upload_file(
                path_or_fileobj=str(temp_file),
                path_in_repo="train.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )

        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        return {
            "status": "success",
            "message": f"Dataset successfully published to Hugging Face Hub!",
            "repo_url": repo_url,
            "num_records": len(records),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to upload to Hugging Face Hub: {str(e)}",
            "repo_id": repo_id,
        }

