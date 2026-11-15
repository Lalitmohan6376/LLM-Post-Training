# 🐍 Gemma-3-1B Python Code Explainer (Supervised Fine-Tuning)

🤗 **Live Model on Hugging Face:** [lalit63/gemma-3-1b-python-explainer](https://huggingface.co/lalit63/gemma-3-1b-python-explainer)

---

## 🎯 What Is This Project?

This project teaches an AI model how to read Python code and explain what it does in simple English.

* 🧠 **The Problem:** Base language models (like `google/gemma-3-1b-pt`) only know how to complete or write more code—they do not know how to answer questions or explain code clearly.
* 💡 **The Solution:** Using **Supervised Fine-Tuning (SFT)**, I trained the Gemma-3-1B model on **126,000+ Python function-explanation pairs** so that whenever you give it a Python function, it directly outputs a clear, human-readable summary.

---

## 🔄 How This Project Works (Step-by-Step Pipeline)

1. 📥 **Load Raw Data:** Downloaded the Python instruction dataset from Hugging Face (`Nan-Do/instructional_code-search-net-python`).
2. 🧹 **Clean & Filter Data:** Removed missing or empty rows using **Pandas** and **NumPy**, and filtered out code-generation tasks to keep only **Python explanation** examples.
3. 🏷️ **Format Prompts:** Structured every example into a clean `### Instruction:` and `### Response:` template.
4. ✂️ **Split Dataset:** Divided the clean data into **90% Training** and **10% Evaluation (Unseen Test Data)** using **Scikit-learn**.
5. 🏋️ **Fine-Tune Model:** Trained `google/gemma-3-1b-pt` on a Tesla T4 GPU using Hugging Face `SFTTrainer`.
6. 📊 **Evaluate & Benchmark:** Tested the trained model on **150 unseen Python functions** using **BERTScore**, **ROUGE-L**, and **BLEU**.
7. 🚀 **Deploy to Hub:** Pushed the final fine-tuned model and tokenizer to Hugging Face Hub for public use.

---

## 🛠️ Tools & Libraries Used

* 🐍 **Python 3** — Core programming language
* 🐼 **Pandas** — Data loading, cleaning, filtering, and text length analysis
* 🔢 **NumPy (Numerical Python)** — Numerical operations and array/metric handling
* 🧪 **Scikit-learn** — Splitting the dataset into train and test sets (`train_test_split`)
* 🤗 **Hugging Face `datasets`** — Loading and converting datasets for model training
* 🤗 **Hugging Face `transformers`** — Loading `google/gemma-3-1b-pt` tokenizer and model
* ⚡ **Hugging Face `trl` (`SFTTrainer`)** — Supervised Fine-Tuning of the language model
* 📏 **Hugging Face `evaluate`** — Calculating ROUGE-L, BLEU, and BERTScore metrics
* ☁️ **Hugging Face Hub & Google Colab** — Model hosting and T4 GPU training environment

---

## 📁 Project Files

* 📓 `01_Data_Clean.ipynb` — Dataset exploration, missing/empty value checks, filtering explanation tasks, and 90:10 train-test split.
* 📓 `02_sft_training.ipynb` — SFT configuration, model training on T4 GPU, before-vs-after testing, and uploading to Hugging Face.
* 📓 `03_Evalute.ipynb` — Testing the fine-tuned model on 150 unseen functions and running error analysis.
* 📓 `Full_SFT_Training.ipynb` — Complete end-to-end workflow in a single notebook.

---

## 📊 Dataset Summary

* 📦 **Dataset:** `Nan-Do/instructional_code-search-net-python`
* 🔢 **Total Raw Rows:** 418,545
* 🎯 **Filtered Python Explanation Rows:** 140,020
* 🏋️ **Training Examples (90%):** 126,018
* 🧪 **Evaluation Examples (10%):** 14,002

---

## ⚙️ Training Configuration

* 🧠 **Base Model:** `google/gemma-3-1b-pt`
* 📏 **Max Sequence Length:** 256 tokens
* 📉 **Learning Rate:** `2e-4`
* ⚡ **Optimizer:** `adafactor`
* 📦 **Batch Size:** 1 (Gradient Accumulation Steps: 8)
* 🖥️ **Precision & Hardware:** FP16 on NVIDIA Tesla T4 GPU

---

## 📈 Evaluation Results (150 Unseen Functions)

The fine-tuned model was evaluated on **150 unseen Python functions** from the test split:

| Evaluation Metric | Score | What It Measures |
| :--- | :---: | :--- |
| 🎯 **BERTScore (F1)** | **87.25%** | Checks if the **meaning** of the model's explanation matches the real answer |
| 📝 **ROUGE-L** | **30.42%** | Checks matching **sentence structure** and word sequences |
| 🔤 **BLEU** | **7.81** | Checks **exact word-for-word** overlap with the reference answer |

### 💡 Why Is BERTScore High (87.25%) While BLEU Is Lower (7.81)?
In code explanation, the same Python function can be explained correctly in many different ways using different English words. **BLEU** penalizes different wording, whereas **BERTScore** checks the actual meaning—proving that the model understands what the Python code is doing.

---

## ⚠️ Error Analysis & Known Limitations

* ✅ **Where It Works Best:** For short and medium Python functions, the model generates clean, accurate one-line summaries.
* ❌ **Where It Fails (Long Functions):** Because `max_length` was set to **256 tokens**, functions longer than 256 tokens get cut off before the `### Response:` tag. Without seeing `### Response:`, the model continues writing Python code instead of explaining it.
* 🔧 **Next Improvement:** Training and running inference with `max_length=512` or `1024` will allow the model to read full long functions without cutting off the prompt.

---

## 🚀 Quick Start (How to Run the Model)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "lalit63/gemma-3-1b-python-explainer"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = """### Instruction:
Explain what the following Python function does:
def is_even(num):
    return num % 2 == 0

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
