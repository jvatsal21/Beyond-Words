import base64
import re
import numpy as np
from io import BytesIO
from PIL import Image
import os
from openai import OpenAI
import json

# This is where we stored the npy files that are much too large to place on github
parent_dir = "/scratch/cse692w25_class_root/cse692w25_class/oyadav/temp_extract/"

# OpenAI client set up
client = OpenAI(api_key="API-KEY")

# Training normalization
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def score_with_gpt4v(image_npy, goal, history, candidate):
    # Load + normalize first frame exactly as in training
    arr   = np.load(os.path.join(parent_dir, image_npy))                     
    frame = np.transpose(arr[0], (1, 2, 0))        
    frame = np.clip(frame * std + mean, 0, 1)   
    frame = (frame * 255).astype("uint8")         
    img   = Image.fromarray(frame).resize((224,224), Image.BILINEAR)

    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Build the prompt
    prompt = (
        f"[GOAL]: {goal}\n"
        f"[HISTORY]: {' '.join(history)}\n"
        f"[CANDIDATE]: {candidate}\n\n"
        "On a scale from 0 (impossible) to 1 (certain), how likely is the candidate action? Respond with the numeric continous value and then an explanation"
    )

    # Call GPT-4 Turbo Vision
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": [
                {"type":"text",      "text": prompt},
                {"type":"image_url", "image_url":{
                    "url": f"data:image/png;base64,{img_b64}"
                }}
            ]
        }],
        temperature=0.7
    )

    # Extract the first numeric value - if it does not exist throw an error 
    text = resp.choices[0].message.content
    print(text)
    m = re.search(r"\b(?:0|1|0?\.\d+)\b", text)
    if not m:
        raise ValueError(f"No score found in response: {text!r}")
    score = float(m.group())
    return score

# Example Usage
if __name__ == "__main__":

    with open("data/verifier.json", "r", encoding="utf-8") as f:
        parsed = json.load(f)
    test_examples = parsed["test"]

    index = 0

    print(test_examples[index])

    score = score_with_gpt4v(
        test_examples[index]['image'],
        test_examples[index]['goal'],
        test_examples[index]['history'],
        test_examples[index]['target']
    )
    print("GPT-4V score:", score)
