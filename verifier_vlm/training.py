#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLModel
from peft import LoraConfig, get_peft_model
from model import QwenVLRegressionModel
from dataset import VerifierDataset

# Change cache paths
HF_HOME = "/scratch/cse692w25_class_root/cse692w25_class/oyadav/cache/huggingface"
parent_dir = "/scratch/cse692w25_class_root/cse692w25_class/oyadav/temp_extract/"
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_NAME)

# Offload parts to the CPU
vlm = Qwen2_5_VLModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,

    device_map={"": "cuda", "encoder": "cpu"},
)

lora_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["regressor"]
)
vlm = get_peft_model(vlm, lora_config)

# Freeze base model parameters (except LoRA adapters)
for name, param in vlm.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

with open("../data/verifier.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

train_ds = VerifierDataset(splits["train"], processor, resolution=112)
val_ds = VerifierDataset(splits["dev"], processor, resolution=112)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QwenVLRegressionModel(vlm).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} / {total_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
epochs = 5

accumulation_steps = 4
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    total_train_loss = 0.0
    num_train_batches = 0

    # Compute and log training
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = model(**batch)["loss"] / accumulation_steps
        scaler.scale(loss).backward()
        
        total_train_loss += loss.item()
        num_train_batches += 1

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / num_train_batches
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    # Compute and log the validation
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                val_out = model(**batch)
                val_loss = val_out["loss"]
                total_val_loss += val_loss.item()
                num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")


os.makedirs("../models/qwen3b-regressor", exist_ok=True)
torch.save(model.regressor.state_dict(), "../models/qwen3b-regressor/regressor.pt")
print("✅ Saved regression head to ../models/qwen3b-regressor/regressor.pt")
