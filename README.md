# Whisper-Style ASR from Scratch

This repo contains an **end-to-end automatic speech recognition (ASR) system implemented in PyTorch**, inspired by OpenAI’s Whisper architecture.  
It combines a custom **FastSubword tokenizer**, a **Whisper-style encoder–decoder Transformer**, and a **mel-spectrogram frontend** to transcribe lecture-style audio.

This is an **educational, course-driven project**, not a production-ready ASR system.

---

## 🧱 Project Structure

Suggested layout:

- `notebooks/asr_whisper_from_scratch.ipynb`  
  Main Colab notebook with the full pipeline:
  - dataset loading (WAV + text transcripts)
  - tokenizer loading / usage
  - mel-spectrogram preprocessing
  - Whisper-style encoder–decoder model
  - training loop with mixed precision + early stopping
  - beam-search decoding and qualitative evaluation

- `FastSubwordTokenizer/`  
  Folder for the pretrained tokenizer file (e.g. `tokenizer.json`) used by `FastSubwordTokenizer.from_pretrained(...)`.

- `_datasets_/` (not committed to git)  
  Local dataset root, expected structure:
  - `dataset_root/wav/` – audio files (`.wav`)
  - `dataset_root/original_txt/` – matching text transcripts (`.txt`)

You can adapt folder names in the notebook/config as needed.

---

## ⚙️ Setup

Install the core dependencies (extend if needed):

```bash
pip install torch torchaudio numpy soundfile pandas matplotlib tqdm pymupdf
