"""Dataset loader and statistics generator for LLM Data Engine.

Supports JSON, JSONL, and CSV files, standardizing to instruction-response format.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import pandas as pd


# Common alternate column names mapped to canonical names
COLUMN_MAPPINGS = {
    "prompt": "instruction",
    "query": "instruction",
    "question": "instruction",
    "input_text": "instruction",
    "input": "instruction",
    "text_in": "instruction",
    "completion": "response",
    "output": "response",
    "answer": "response",
    "target": "response",
    "output_text": "response",
    "text_out": "response",
}


def load_dataset(source: Union[str, Path, io.BytesIO, io.StringIO], file_format: str = None) -> List[Dict[str, Any]]:
    """Loads a dataset from path, raw text string, or buffer and returns standardized records."""
    raw_records: List[Dict[str, Any]] = []

    # Check if source is a raw JSON/JSONL/CSV text string directly (not a file path)
    if isinstance(source, str) and ("\n" in source or source.strip().startswith(("[", "{"))):
        source = io.StringIO(source)

    # If source is a string/Path pointing to a file
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        ext = path.suffix.lower().lstrip(".")
        format_to_use = (file_format or ext).lower()

        if format_to_use in ["jsonl", "jsonlines"]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_records.append(json.loads(line))
        elif format_to_use == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_records = data
                elif isinstance(data, dict):
                    # Handle dict with records key
                    for key in ["data", "records", "samples", "items"]:
                        if key in data and isinstance(data[key], list):
                            raw_records = data[key]
                            break
                    if not raw_records:
                        raw_records = [data]
        elif format_to_use == "csv":
            df = pd.read_csv(path)
            raw_records = df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported file format: {format_to_use}. Use json, jsonl, or csv.")
    else:
        # Source is a buffer (e.g. from Streamlit)
        format_to_use = (file_format or "jsonl").lower()
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        if format_to_use in ["jsonl", "jsonlines"]:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))
        elif format_to_use == "json":
            data = json.loads(content)
            if isinstance(data, list):
                raw_records = data
            elif isinstance(data, dict):
                for key in ["data", "records", "samples", "items"]:
                    if key in data and isinstance(data[key], list):
                        raw_records = data[key]
                        break
                if not raw_records:
                    raw_records = [data]
        elif format_to_use == "csv":
            df = pd.read_csv(io.StringIO(content))
            raw_records = df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported format: {format_to_use}")

    # Standardize column names and multi-turn conversations
    standardized = []
    for row in raw_records:
        if not isinstance(row, dict):
            continue

        # Handle Multi-turn Chat (OpenAI 'messages' format)
        if "messages" in row and isinstance(row["messages"], list):
            inst, resp = parse_chat_messages(row["messages"])
            new_row = dict(row)
            new_row["instruction"] = inst
            new_row["response"] = resp
            standardized.append(new_row)
            continue

        # Handle ShareGPT 'conversations' format
        if "conversations" in row and isinstance(row["conversations"], list):
            inst, resp = parse_sharegpt_conversations(row["conversations"])
            new_row = dict(row)
            new_row["instruction"] = inst
            new_row["response"] = resp
            standardized.append(new_row)
            continue

        new_row = {}
        for k, v in row.items():
            canonical_key = COLUMN_MAPPINGS.get(str(k).strip().lower(), str(k).strip().lower())
            new_row[canonical_key] = v
        # Ensure instruction and response keys exist (even if None)
        if "instruction" not in new_row and "text" in new_row:
            new_row["instruction"] = new_row["text"]
            new_row["response"] = ""
        standardized.append(new_row)

    return standardized


def parse_chat_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Converts OpenAI-style messages array to instruction-response pair."""
    if not messages:
        return "", ""

    # Find the last assistant message
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx == -1:
        # No assistant message, concatenate all as instruction
        inst = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return inst, ""

    # Context up to the last assistant message
    context_turns = messages[:last_assistant_idx]
    if len(context_turns) == 1 and context_turns[0].get("role") == "user":
        # Simple single-turn
        instruction = context_turns[0].get("content", "")
    else:
        # Multi-turn dialogue context
        instruction = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in context_turns)

    response = messages[last_assistant_idx].get("content", "")
    return instruction, response


def parse_sharegpt_conversations(conversations: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Converts ShareGPT-style conversations array to instruction-response pair."""
    mapped = []
    for turn in conversations:
        sender = turn.get("from", "").lower()
        content = turn.get("value", "")
        role = "assistant" if sender in ["gpt", "bot", "assistant"] else "user"
        mapped.append({"role": role, "content": content})
    return parse_chat_messages(mapped)


def get_dataset_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes descriptive statistics on instruction and response lengths and vocabulary."""
    if not records:
        return {
            "total_records": 0,
            "avg_instruction_chars": 0.0,
            "avg_instruction_words": 0.0,
            "avg_response_chars": 0.0,
            "avg_response_words": 0.0,
            "max_instruction_chars": 0,
            "max_response_chars": 0,
            "vocabulary_size": 0,
        }

    inst_chars = []
    inst_words = []
    resp_chars = []
    resp_words = []
    vocab = set()

    for r in records:
        inst = str(r.get("instruction") or "")
        resp = str(r.get("response") or "")

        inst_chars.append(len(inst))
        w_inst = inst.split()
        inst_words.append(len(w_inst))
        vocab.update([w.lower().strip(".,!?:;\"'()[]{}") for w in w_inst])

        resp_chars.append(len(resp))
        w_resp = resp.split()
        resp_words.append(len(w_resp))
        vocab.update([w.lower().strip(".,!?:;\"'()[]{}") for w in w_resp])

    n = len(records)
    return {
        "total_records": n,
        "avg_instruction_chars": round(sum(inst_chars) / n, 1),
        "avg_instruction_words": round(sum(inst_words) / n, 1),
        "avg_response_chars": round(sum(resp_chars) / n, 1),
        "avg_response_words": round(sum(resp_words) / n, 1),
        "max_instruction_chars": max(inst_chars) if inst_chars else 0,
        "max_response_chars": max(resp_chars) if resp_chars else 0,
        "vocabulary_size": len(vocab),
    }

