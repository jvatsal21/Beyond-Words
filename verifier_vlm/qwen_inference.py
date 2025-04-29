import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLModel, Qwen2_5_VLProcessor
from peft import LoraConfig, get_peft_model
import os
from model import QwenVLRegressionModel

# ─── Setup ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_resolution = 112  
num_channels = 6 

# ─── Load base model and processor ─────────────────────────────────────────────
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_NAME)

vlm = Qwen2_5_VLModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters if used
lora_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["regressor"]
)
vlm = get_peft_model(vlm, lora_config)


# ─── Build model, load regressor weights ───────────────────────────────────────
model = QwenVLRegressionModel(vlm).to(device)
regressor_weights = torch.load("qwen3b-regressor/regressor.pt", map_location=device)
model.regressor.load_state_dict(regressor_weights)
model.eval()

# ─── Preprocessing Constants ───────────────────────────────────────────────────
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# ─── Prediction Function ───────────────────────────────────────────────────────
def predict_score(image_path: str, text: str) -> float:
    # Load and normalize the first frame
    arr = np.load(image_path)  # (frames, C, H, W)
    frame = arr[0]  # (C, H, W)
    frame = np.transpose(frame, (1, 2, 0))  # to (H, W, C)
    frame = frame * std + mean
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    image = Image.fromarray(frame).resize((image_resolution, image_resolution))

    inputs = processor(
        text=[text],
        images=[image],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    return out["preds"].cpu().item()

# ─── Example Usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_path = "/scratch/cse692w25_class_root/cse692w25_class/oyadav/temp_extract/"

    ex = {
      "image": "EgoCOT_clear/EGO_2009.npy",
      "goal": "[GOAL] touches the floor.",
      "history": [
        "[STEP] stand up and touch the floor"
      ],
      "target": "[EOP] levitate",
      "score": 0.001
    }

    ex = dict(ex)
    
    goal = ex['goal']
    history_str = " ".join(ex['history'])
    candidate = ex['target']
    prompt = f"[GOAL]: {goal}  [HISTORY]: {history_str}  [CANDIDATE]: {candidate}"

    score = predict_score(os.path.join(example_path, ex["image"]), prompt)
    print(f"✅ Predicted continuous score: {score:.4f}")
