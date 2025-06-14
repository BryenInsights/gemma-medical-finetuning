# Fine-tuning **Gemma-3 Instruct (1 B)** on *MTS-Dialog* with LoRA

An end-to-end, reproducible workflow that trains a 1 B-parameter Gemma model to
turn raw doctor-patient conversations into concise clinical notes.  
Everything happens in a single notebook / script:

* **LoRA rank 8**, **3 epochs** – fits on a single ≈8 GB T4.:contentReference[oaicite:0]{index=0}  
* **Mixed-precision (`bfloat16`)** for extra memory headroom.:contentReference[oaicite:1]{index=1}  
* **EarlyStopping on `val_loss`** to avoid over-training.  
* Post-training **ROUGE evaluation** and a small **Playground** widget for manual
  testing.:contentReference[oaicite:2]{index=2}

---

## 🏁 Quick start (Colab)
	1.	Open the notebook in Colab (File ▸ Open Notebook ▸ “Upload” and choose gemma_medical_finetuning.ipynb).
	2.	Set the runtime to GPU (Runtime ▸ Change runtime type ▸ GPU). A T4 or better (≈ 8 GB VRAM) is enough.
	3.	Install dependencies – simply run the first code cell:

!pip install -q -U tensorflow keras keras-nlp keras-hub rouge_score scipy tqdm ipywidgets


	4.	Load utils.py into Colab (needed by the notebook):

### ▶︎ Option A — clone the repo (preferred, keeps everything in sync)
!git clone https://github.com/your-handle/gemma-mts-dialog.git
%cd gemma-mts-dialog

### ▶︎ Option B — manual upload (if you don’t want to clone)
from google.colab import files
files.upload()   # then pick utils.py from your computer

The notebook expects utils.py to be in the current working directory; either option is fine.

	5.	Run all cells. The notebook pulls the MTS-Dialog CSVs automatically and starts training.
After ≈ 45 min a file named lora_rank8.weights.lora.h5 will appear — that’s your LoRA adapter, ready for inference.

---

## ⚠️ Local execution

This tutorial is written and tested for Google Colab.
Running it locally is untested and may require manual tweaks (CUDA setup, memory tweaks, custom dataset paths). If you do attempt it, please treat the process as experimental.

---

## Files

| File                                | Purpose                                                |
| ----------------------------------- | ------------------------------------------------------ |
| `gemma_medical_finetuning.py`       | Full training / evaluation / demo pipeline             |
| `utils.py`                          | Main functions used in the notebook                    |
| `lora_rank16.weights.lora.h5`       | LoRA adapter weights (created after training)          |
| `requirements.txt`                  | Exact Python packages & versions                       |
| `example_playground.ipynb`          | Minimal inference notebook (load adapters & summarise) |

---

## 💡 Memory & stability tips

* If the GPU crashes at **epoch 2**, disable `restore_best_weights` in the
  `EarlyStopping` callback or cap validation examples to `≤ 1024` tokens before
  calling `model.fit()`.
* Keep **batch\_size = 2** and use **Top-K = 5** sampling for deterministic eval
  runs.

---

## 📜 Dataset licence

*MTS-Dialog* © 2024 – MIT-licensed (see `LICENSE-DATA`).
Generated clinical notes are purely **research output**; they are **not** meant
for real-world medical use without additional safety checks.

---

## 🙏 Credits

* Google AI for releasing **Gemma-3** and the official LoRA tutorial.
* Original MTS-Dialog creators for the dataset.
* This repo authored & cleaned-up by **@your-handle** (May 2025).
