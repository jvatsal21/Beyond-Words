# Beyond Words: Integrating Image-Based Context for Situation-Grounded Procedural Planning

This project explores using step-wise, verifier-guided beam search for **vision-language procedural planning** in embodied AI settings.  
We extend prior text-only planning methods by conditioning plans not just on textual goals, but also on **egocentric visual context**.

Building off from [PlaSma](https://arxiv.org/abs/2305.19472) (symbolic verifier-guided planning), we develop a **stepwise beam search** system that iteratively generates action steps while consulting a vision-language verifier at each step.

Our work aims to answer three research questions:
- Does step-wise verifier-guided beam search improve planning compared to full-plan generation?
- How does the number of beams and candidates impact the quality of planning?
- Does finetuning on egocentric chain-of-thought ([EgoCOT](https://github.com/EmbodiedGPT/EgoCOT_Dataset)) data improve planning quality over few-shot prompting?

[Read our Paper (PDF)](Beyond_Words.pdf)

## 🧠 Key Contributions

- **Stepwise Planning with Verifier-Guided Beam Search**  
  Incrementally generate plans step-by-step, scoring candidate steps with a vision-language model-based verifier.

- **EgoCOT Finetuning**  
  Finetuned Qwen2.5-VL-3B on EgoCOT for next-step prediction in egocentric environments.

- **LLM-Based Evaluation**  
  Used GPT-4-turbo as a judge for pairwise plan comparisons based on conciseness, minimality, and actionability.

## 📂 Project Structure

```plaintext
/
├── Beyond_Words.ipynb          # (Main notebook: beam search, evaluation experiments, inference)
├── Beyond_Words.pdf            # (Our research paper)
├── README.md                   # (You're reading it!)
├── requirements.txt            # (Python dependencies)
├── checkpoints/                
│   ├── qwen-finetuned-lora/     # (LoRA adapters for the finetuned planning model)
│   └── regressor.pt             # (Weights for the Qwen-based verifier model)
├── data/                       
│   ├── images_png/              # (Egocentric images for testing)
│   ├── test_set.json            # (List of goals and associated image paths for evaluation)
│   └── verifier_data.json       # (Training schema for Qwen-based verifier model)
├── planning_vlm/
│   ├── train_logs/              # (Training logs for planner finetuning)
│   └── train_planner.py         # (Finetuning script for planner model)
├── verifier_vlm/
│   ├── dataset.py               # (Creates the verifier dataset structure)
│   ├── model.py                 # (Regression model for computing continuous [0, 1] score)
│   ├── training.py              # (Training script for verifier LoRA finetuning)
│   ├── qwen_inference.py        # (Inference script using Qwen as verifier)
│   └── gpt_inference.py         # (Inference script using GPT API as verifier)
```

## 🚀 How to Run
1. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

2. **Run Inference and Evaluation**

    The main experiments and results are inside `Beyond_Words.ipynb`.
   
    Open the notebook and run all cells sequentially to:
    
    - Perform verifier-guided beam search planning
    - Compare full-plan generation vs step-wise generation
    - Evaluate plans using GPT-4-turbo as a judge
    - Analyze results for all three research questions

4. **Finetune the Planner (Optional)**

    If you want to retrain the planning model (`Qwen2.5-VL-3B-Instruct`):

    ```bash
    python planning_vlm/train_planner.py
    ```

## 📈 Notes

- **Test Set**: We evaluate performance of our framework with 50 examples sampled from EgoCOT (located in `data/test_set.json`).

- **Verifier**: We switched from a trained verifier model (`verifier_vlm/`) to a GPT-4-turbo API-based verifier for better, more reliable scoring.

- **Evaluation Constraints**: We restricted to a maximum of 5 steps per plan, with optional early stopping using `[EOP]`.

## ⚡ Acknowledgements

- Our work is an extension of [**PlaSma: Symbolic Planning with Verifier-Guided Beam Search**](https://arxiv.org/abs/2305.19472).
- We use models and utilities from **Hugging Face**, **Qwen-VL**, and **OpenAI**.

## 🏆 Contributors

- Vatsal Joshi, M.S. University of Michigan ([link](https://github.com/jvatsal21))
- Omkar Yadav, M.S. University of Michigan ([link](https://github.com/omkar-yadav-12))

