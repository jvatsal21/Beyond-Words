from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from PIL import Image
from transformers import TrainingArguments, Trainer

torch.cuda.empty_cache()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
# model = model.to("cuda")
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    prompt = processor.apply_chat_template(
        [
            {"role": "system", "content": example["system"]},
            {"role": "user",   "content": [
                {"type": "text", "text": example["user"]},
                {"image": "file://" + example["image"]},
            ]},
            {"role": "assistant", "content": example["response"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    # Pass the PIL directly to avoid the extra 1-dim on pixel_values/Grid
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Squeeze away that leading-1 on all tensors
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
            inputs[k] = v.squeeze(0)

    # Make labels for causal LM
    inputs["labels"] = inputs["input_ids"]
    return inputs

dataset = load_dataset("json", data_files="final_train.json")
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=processor,
)

trainer.train()

trainer.save_model("qwen-finetuned-lora")
