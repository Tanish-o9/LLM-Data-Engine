"""LoRA fine-tuning module for Causal Language Models.

Supports GPU/CPU hardware awareness, training wall-clock time tracking,
and peak GPU VRAM monitoring.
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TrainingConfig
from training.dataset_builder import InstructionDataset


def get_target_device(config: TrainingConfig) -> torch.device:
    """Selects CUDA device if available and configured, otherwise CPU."""
    if config.use_cuda_if_available and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def train_lora_model(
    train_records: List[Dict[str, Any]],
    experiment_name: str,
    config: Optional[TrainingConfig] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """Trains a LoRA adapter on the specified dataset and saves artifacts.

    Returns:
        dict containing training metrics, checkpoint paths, and hardware stats.
    """
    if config is None:
        config = TrainingConfig()

    device = get_target_device(config)
    save_path = Path(config.output_dir) / experiment_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Reset GPU memory tracking if using CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    start_time = time.time()

    # 1. Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    base_model.to(device)

    # 2. Configure LoRA
    # Detect appropriate target modules for gpt2/distilgpt2 vs llama
    target_modules = config.lora_target_modules
    if "gpt2" in config.model_name.lower():
        target_modules = ["c_attn", "c_proj"]
    elif "llama" in config.model_name.lower() or "qwen" in config.model_name.lower():
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.train()

    # 3. Create Dataset and DataLoader
    dataset = InstructionDataset(train_records, tokenizer, max_length=config.max_seq_length)
    if len(dataset) == 0:
        raise ValueError(f"No valid training samples for experiment '{experiment_name}'.")

    batch_size = min(config.batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Optimizer and Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = config.num_train_epochs * len(dataloader)
    current_step = 0
    epoch_losses: List[float] = []

    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0
        steps_in_epoch = 0

        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            optimizer.step()

            current_step += 1
            steps_in_epoch += 1
            loss_val = float(loss.item())
            epoch_loss += loss_val

            if progress_callback:
                progress_callback(current_step, total_steps, loss_val)

        avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
        epoch_losses.append(round(avg_epoch_loss, 4))

    elapsed_time = round(time.time() - start_time, 2)

    # Measure Peak Memory
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

    # 5. Save LoRA Adapter & Tokenizer
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    return {
        "experiment_name": experiment_name,
        "model_name": config.model_name,
        "checkpoint_dir": str(save_path),
        "device": device.type,
        "training_time_seconds": elapsed_time,
        "peak_gpu_memory_mb": peak_memory_mb,
        "num_epochs": config.num_train_epochs,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "num_samples_trained": len(dataset),
    }

