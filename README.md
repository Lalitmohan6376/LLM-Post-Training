# 🚀 LLM-Post-Training

🤗 **Hugging Face Profile:** [huggingface.co/lalit63](https://huggingface.co/lalit63)

---

## 🎯 What Is This Repository?

**LLM-Post-Training** is an all-in-one repository focused on taking pre-trained base Large Language Models (LLMs) and teaching them how to follow instructions, explain code, and align with human preferences.

* 🧠 **The Goal:** Base models only predict the next word—they do not naturally know how to answer questions or follow specific formats.
* 💡 **The Solution:** This repository covers end-to-end **Post-Training** pipelines, including **Full Supervised Fine-Tuning (SFT)**, **Direct Preference Optimization (DPO)**, and **Parameter-Efficient Fine-Tuning (LoRA / QLoRA)**.

---

## 🗂️ Repository Modules & Projects

### 1️⃣ Supervised Fine-Tuning (Full SFT)
* 📌 **Task:** Automated Python Code Explanation
* 🧠 **Base Model:** `google/gemma-3-1b-pt`
* 🤗 **Uploaded Model:** [lalit63/gemma-3-1b-python-explainer](https://huggingface.co/lalit63/gemma-3-1b-python-explainer)
* 📦 **Dataset:** `Nan-Do/instructional_code-search-net-python` (140,020 filtered Python explanation pairs)
* 📈 **Benchmark Scores (150 Unseen Functions):**
  * 🎯 **BERTScore (F1):** **87.25%**
  * 📝 **ROUGE-L:** **30.42%**
  * 🔤 **BLEU:** **7.81**

---

### 2️⃣ Direct Preference Optimization (DPO)
* 📌 **Task:** Preference Alignment (Choosing concise, accurate responses over noisy/rejected ones)
* 🧠 **Base Model:** `google/gemma-3-1b-pt`
* 🔄 **Status:** Active Experimentation (Direct DPO & Preference Dataset Alignment)

---

### 3️⃣ Parameter-Efficient Fine-Tuning (LoRA / QLoRA)
* 📌 **Task:** Memory-efficient adapter training on consumer GPUs (Tesla T4)
* ⚡ **Focus:** Low-Rank Adaptation (`peft`) for fast training with minimal VRAM usage
* 🔄 **Status:** Upcoming Module

---

## 🔄 End-to-End Post-Training Workflow

1. 📥 **Data Collection:** Load raw instruction or preference datasets from Hugging Face.
2. 🧹 **Data Cleaning & EDA:** Remove empty/missing rows, check token length distributions, and filter target tasks using **Pandas** and **NumPy**.
3. 🏷️ **Prompt Formatting:** Structure data into clean instruction-response or chosen-rejected templates.
4. ✂️ **Train / Test Split:** Split datasets into training and unseen evaluation sets using **Scikit-learn**.
5. 🏋️ **Model Training:** Run SFT, DPO, or LoRA training on Google Colab (T4 GPU) using Hugging Face `trl`.
6. 📊 **Quantitative Evaluation:** Benchmark trained models on unseen test samples using **BERTScore**, **ROUGE-L**, and **BLEU**, followed by error analysis.
7. 🚀 **Model Deployment:** Push trained weights and tokenizers to Hugging Face Hub.

---

## 🛠️ Tools & Libraries Used

* 🐍 **Python 3** — Core programming language
* 🐼 **Pandas** — Data loading, cleaning, filtering, and length analysis
* 🔢 **NumPy (Numerical Python)** — Numerical operations and metric calculations
* 🧪 **Scikit-learn** — Train/validation dataset splitting (`train_test_split`)
* 🤗 **Hugging Face `datasets`** — Dataset loading and preprocessing
* 🤗 **Hugging Face `transformers`** — Base model and tokenizer loading
* ⚡ **Hugging Face `trl`** — `SFTTrainer` and `DPOTrainer` for post-training
* 🧩 **Hugging Face `peft`** — LoRA / QLoRA adapter fine-tuning
* 📏 **Hugging Face `evaluate`** — Model benchmarking (BERTScore, ROUGE, SacreBLEU)
* ☁️ **Hugging Face Hub & Google Colab** — Model hosting and T4 GPU training

---

## 📁 Folder Structure

```text
LLM-Post-Training/
│
├── 01_SFT_Python_Explainer/       # Full Supervised Fine-Tuning (Data Prep, Training, Evaluation)
├── 02_DPO_Alignment/              # Direct Preference Optimization experiments
├── 03_LoRA_FineTuning/            # Parameter-Efficient Fine-Tuning (LoRA / QLoRA)
└── README.md                      # Main repository documentation
