import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# This is where we stored the npy files that are much too large to place on github
parent_dir = "/scratch/cse692w25_class_root/cse692w25_class/oyadav/temp_extract/"

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])



class VerifierDataset(Dataset):
    def __init__(self, examples, processor, max_length=77, resolution=112):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length
        self.resolution = resolution

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        arr = np.load(os.path.join(parent_dir, ex["image"]))
        frame0 = arr[0]  # Assuming arr[0] gives you the image tensor
        frame0 = np.transpose(frame0, (1, 2, 0))
        frame0 = frame0 * std + mean
        frame0 = np.clip(frame0, 0, 1)
        # Instead of multiplying by 255, keep normalized if processor expects that
        frame0 = (frame0 * 255).astype(np.uint8)
        image = Image.fromarray(frame0)
        # Downsample the image to the new resolution
        image = image.resize((self.resolution, self.resolution), resample=Image.BILINEAR)
        
        goal = ex['goal']
        history_str = " ".join(ex['history'])
        candidate = ex['target']
        prompt = f"[GOAL]: {goal}  [HISTORY]: {history_str}  [CANDIDATE]: {candidate}"
        
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["score"], dtype=torch.float)
        }
