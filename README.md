# Fine-tuning **Gemma-3 Instruct (1 B)** on *MTS-Dialog* with LoRA

An end-to-end, reproducible workflow that trains a 1 B-parameter Gemma model to
turn raw doctor-patient conversations into concise clinical notes.  
Everything happens in a single notebook / script:

* **LoRA rank 16**, **3 epochs** – fits on a single ≈16 GB T4.
* **Mixed-precision (`bfloat16`)** for extra memory headroom. 
* **EarlyStopping on `val_loss`** to avoid over-training.  
* Post-training **ROUGE evaluation** and a small **Playground** for manual testing.

---

## 🏁 Quick start (Colab)
	1.	Open the notebook in Colab (File ▸ Open Notebook ▸ “Upload” and choose gemma_medical_finetuning.ipynb).
	2.	Set the runtime to GPU (Runtime ▸ Change runtime type ▸ GPU). A T4 or better (≈ 16 GB VRAM) is enough.
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
After ≈ 45 min a file named lora_rank_16.weights.lora.h5 will appear — that’s your LoRA adapter, ready for inference.

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
| `requirements.txt`                  | Exact Python packages & versions (if running locally)  |

---

## 📜 Dataset licence

*MTS-Dialog* © 2023 – released under **CC BY 4.0** (see `LICENSE-DATA`)
Generated clinical notes are purely **research output**; they are **not** meant
for real-world medical use without additional safety checks.

---

## 🙏 Credits

* Google AI for releasing **Gemma-3** and the official LoRA tutorial.
* Original MTS-Dialog creators for the dataset.
* This repo authored & cleaned-up by **@BryenInsights** (June 2025).
