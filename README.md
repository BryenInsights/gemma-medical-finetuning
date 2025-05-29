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

1. Open the Colab link (or upload the notebook).  
2. Run **`pip install -q -U keras-hub keras keras-nlp rouge_score scipy tqdm ipywidgets`**.:contentReference[oaicite:3]{index=3}  
3. Upload `MTS-Dialog-TrainingSet.csv` (and optionally
   `MTS-Dialog-ValidationSet.csv`) to `/content/`.:contentReference[oaicite:4]{index=4}  
4. **Run all**. After ±45 min on a T4 you’ll have `lora_rank8.weights.lora.h5`
   ready for inference.:contentReference[oaicite:5]{index=5}

---

## 🚀 Running locally

```bash
git clone
cd gemma-mts-dialog
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python finetune_gemma_medical_dataset.py
````

> **GPU:** any 10 GB+ CUDA card is fine; with a T4/RTX A2000 use the
> provided `mixed_bfloat16` policy.

---

## Files

| File                                | Purpose                                                |
| ----------------------------------- | ------------------------------------------------------ |
| `finetune_gemma_medical_dataset.py` | Full training / evaluation / demo pipeline             |
| `lora_rank8.weights.lora.h5`        | LoRA adapter weights (created after training)          |
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
