# AI Music Generation Project

This repository implements and compares multiple symbolic music generation methods:
- Task 1: LSTM Autoencoder (AE)
- Task 2: LSTM Variational Autoencoder (VAE)
- Task 3: Transformer
- Task 4: RLHF-style fine-tuned Transformer
- Baselines: Random Generator and Markov Chain

---
## 0) Dataset Link: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2) Expected Data Layout

Place MIDI files in:
- `data/raw_midi/unknown/`

Current configuration uses a single genre label (`unknown`).

---

## 3) Quick Start (Training + Generation)

### Task 1: Autoencoder (5 samples)
```bash
python -m src.training.train_ae --epochs 10 --genre unknown
python -m src.generation.generate_music --model ae --num-samples 5 --seq-len 128 --genre unknown
```

### Task 2: VAE (8 samples + latent interpolation)
```bash
python -m src.training.train_vae --epochs 10
python -m src.generation.generate_music --model vae --num-samples 8 --seq-len 128 --genre unknown
python -m src.generation.latent_interpolation --steps 8 --seq-len 128
```

### Task 3: Transformer (10 samples)
```bash
python -m src.training.train_transformer --epochs 10
python -m src.generation.generate_music --model transformer --num-samples 10 --seq-len 128 --genre unknown
```

### Baselines (required comparison)
```bash
python -m src.generation.generate_music --model random --num-samples 5 --seq-len 128
python -m src.generation.generate_music --model markov --num-samples 5 --seq-len 128
```

### Task 4: RLHF-style fine-tuning (10 samples)
```bash
python -m src.training.train_rlhf --steps 100 --seq-len 128
python -m src.generation.generate_music --model rlhf --num-samples 10 --seq-len 128 --genre unknown
```

### Build evaluation artifacts
```bash
python -m src.evaluation.build_comparison_table
python -m src.evaluation.rlhf_before_after
```

---

## 4) Project File Guide (What each main file does)

### Root-level
- `README.md`: setup, run instructions, outputs, links, and contribution summary.
- `requirements.txt`: Python dependencies required for running the project.

### Configuration
- `src/config.py`: directory paths and shared training configuration (`TrainConfig`).

### Preprocessing
- `src/preprocessing/midi_parser.py`: loads MIDI files and converts MIDI into token/event sequences.
- `src/preprocessing/tokenizer.py`: token definitions, vocab, sequence conversion utilities.

### Model definitions
- `src/models/autoencoder.py`: LSTM autoencoder architecture.
- `src/models/vae.py`: LSTM variational autoencoder architecture and KL helper.
- `src/models/transformer.py`: transformer architecture used for autoregressive generation.

### Training scripts
- `src/training/common.py`: shared utilities (data loading, train/val split, plotting, metrics JSON).
- `src/training/train_ae.py`: trains Task 1 autoencoder and saves AE metrics/weights.
- `src/training/train_vae.py`: trains Task 2 VAE and saves total/recon/KL metrics and weights.
- `src/training/train_transformer.py`: trains Task 3 transformer and reports perplexity.
- `src/training/train_rlhf.py`: performs Task 4 RLHF-style policy-gradient fine-tuning.

### Generation
- `src/generation/generate_music.py`: generates MIDI from trained models and baseline methods.
- `src/generation/midi_export.py`: converts token sequences back to MIDI files.
- `src/generation/latent_interpolation.py`: latent interpolation generation utility for VAE.

### Evaluation
- `src/evaluation/metrics.py`: objective metrics over generated samples.
- `src/evaluation/build_comparison_table.py`: builds final `comparison_table.csv`.
- `src/evaluation/rlhf_before_after.py`: builds RLHF before/after comparison CSV/plot.
- `src/evaluation/build_baseline_plot.py` (if present): creates baseline comparison plot.

### Outputs
- `outputs/generated_midis/`: generated MIDI files for all tasks and baselines.
- `outputs/plots/`: all plots (`.png`), metrics (`.json`), and tables (`.csv`).
- `outputs/survey_results/`: human survey template/results files.

### Report
- `report/final_report.tex`: main LaTeX report source.
- `report/final_report.pdf`: compiled report PDF.

---

## 5) Expected Outputs

### Generated MIDI files (`outputs/generated_midis/`)
- Task 1: `task1_ae_genre-unknown_1.mid` ... `task1_ae_genre-unknown_5.mid`
- Task 2: `task2_vae_genre-unknown_1.mid` ... `task2_vae_genre-unknown_8.mid`
- Task 3: `task3_transformer_genre-unknown_1.mid` ... `task3_transformer_genre-unknown_10.mid`
- Task 4: `task4_rlhf_genre-unknown_1.mid` ... `task4_rlhf_genre-unknown_10.mid`
- Baselines: `baseline_random_*.mid`, `baseline_markov_*.mid`

### Plots and evaluation files (`outputs/plots/`)
- `task1_ae_reconstruction_loss.png`
- `task2_vae_total_loss.png`
- `task2_vae_reconstruction_loss.png`
- `task2_vae_kl_loss.png`
- `task3_transformer_perplexity.png`
- `task4_before_after.png`
- `task4_before_after.csv`
- `baseline_comparison.png`
- `comparison_table.csv`
- `*_generation_metrics.json`

---

## 6) Reproduction Commands (Artifact-only run)

```bash
python -m src.generation.generate_music --model ae --num-samples 5 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model vae --num-samples 8 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model transformer --num-samples 10 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model rlhf --num-samples 10 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model random --num-samples 5 --seq-len 128
python -m src.generation.generate_music --model markov --num-samples 5 --seq-len 128

python -m src.evaluation.build_comparison_table
python -m src.evaluation.rlhf_before_after
```

---

## 7) External Links

- Report PDF Drive link:  
  https://drive.google.com/file/d/15fojQBM1Qt8P19nPDtDpk-6my3DIaQhW/view?usp=sharing

- MIDI files Drive folder link (replace with your actual shared folder):  
  https://drive.google.com/drive/folders/1HRi3BxpQ1nbZinuAiZFgwNusStir2NE2?usp=drive_link

---

## 8) Group Member Contributions

- **Naheyan Tahmin (23101001)**  
  Task 3 (Transformer) and Task 4 (RLHF-style fine-tuning): implementation, experiments, and result analysis for these tasks.

- **Ragib Rawnak (23101346)**  
  Data preprocessing and dataset preparation, Task 1 (Autoencoder), Task 2 (VAE), and related generation/evaluation support.
