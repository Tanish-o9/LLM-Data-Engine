"""Dataset builder and tokenization for instruction fine-tuning.
"""

from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)

PROMPT_INPUT_ONLY = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt(instruction: str, response: str = "") -> str:
    """Formats an instruction and optional response into the standard instruction-following format."""
    if response:
        return PROMPT_TEMPLATE.format(instruction=instruction.strip(), response=response.strip())
    return PROMPT_INPUT_ONLY.format(instruction=instruction.strip())


class InstructionDataset(Dataset):
    """PyTorch Dataset that tokenizes instruction-response pairs and masks prompt tokens with -100."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for row in records:
            inst = row.get("instruction", "").strip()
            resp = row.get("response", "").strip()
            if not inst or not resp:
                continue

            full_text = format_prompt(inst, resp) + self.tokenizer.eos_token
            prompt_text = format_prompt(inst)

            encoded_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            encoded_prompt = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encoded_full["input_ids"].squeeze(0)
            attention_mask = encoded_full["attention_mask"].squeeze(0)

            # Mask prompt tokens so loss is computed strictly on response tokens
            labels = input_ids.clone()
            prompt_len = min(encoded_prompt["input_ids"].shape[1], self.max_length)
            labels[:prompt_len] = -100

            # Also mask padding tokens
            labels[attention_mask == 0] = -100

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

