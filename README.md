## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Expected Data Layout

Place MIDI files in:

- Required: `data/raw_midi/unknown/`

## 3) Quick Start

#### Task 1 (Autoencoder, 5 samples)
```bash
python -m src.training.train_ae --epochs 10 --genre unknown
python -m src.generation.generate_music --model ae --num-samples 5 --seq-len 128
```

#### Task 2 (VAE, 8 samples + latent interpolation)
```bash
python -m src.training.train_vae --epochs 10
python -m src.generation.generate_music --model vae --num-samples 8 --seq-len 128
python -m src.generation.latent_interpolation --steps 8 --seq-len 128
```

#### Task 3 (Transformer, 10 long-sequence samples)
```bash
python -m src.training.train_transformer --epochs 10
python -m src.generation.generate_music --model transformer --num-samples 10 --seq-len 128 --genre unknown
```

#### Baselines (required comparison)
```bash
python -m src.generation.generate_music --model random --num-samples 5 --seq-len 128
python -m src.generation.generate_music --model markov --num-samples 5 --seq-len 128
```

#### Task 4 (RLHF, 10 fine-tuned samples)
```bash
python -m src.training.train_rlhf --steps 100 --seq-len 128
python -m src.generation.generate_music --model rlhf --num-samples 10 --seq-len 128 --genre unknown
```

#### Build Evaluation Tables
```bash
python -m src.evaluation.build_comparison_table
python -m src.evaluation.rlhf_before_after
```

## 4) Outputs

- Generated MIDI files (genre-tagged names): `outputs/generated_midis/`
  - Task 1: `task1_ae_genre-unknown_1.mid` ... `task1_ae_genre-unknown_5.mid`
  - Task 2: `task2_vae_genre-unknown_1.mid` ... `task2_vae_genre-unknown_8.mid`
  - Task 3: `task3_transformer_genre-unknown_1.mid` ... `task3_transformer_genre-unknown_10.mid`
  - Task 4: `task4_rlhf_genre-unknown_1.mid` ... `task4_rlhf_genre-unknown_10.mid`
  - Baselines: `baseline_random_*.mid`, `baseline_markov_*.mid`
- Required plots and evaluation artifacts: `outputs/plots/`
  - AE loss: `task1_ae_reconstruction_loss.png`
  - VAE metrics: `task2_vae_total_loss.png`, `task2_vae_reconstruction_loss.png`, `task2_vae_kl_loss.png`
  - Transformer perplexity: `task3_transformer_perplexity.png`
  - RLHF before/after: `task4_before_after.png` (and `task4_before_after.csv`)
  - Baseline comparison: `baseline_comparison.png`
  - Combined table: `comparison_table.csv`
  - Baseline metrics JSON: `random_generation_metrics.json`, `markov_generation_metrics.json`
- Human survey files: `outputs/survey_results/human_survey_template.csv`, `outputs/survey_results/human_survey_results.csv`
- Final compiled report PDF: `report/final_report.pdf`

## 5) Reproduction Commands (Artifacts)

Generate required MIDI files with exact counts:

```bash
python -m src.generation.generate_music --model ae --num-samples 5 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model vae --num-samples 8 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model transformer --num-samples 10 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model rlhf --num-samples 10 --seq-len 128 --genre unknown
python -m src.generation.generate_music --model random --num-samples 5 --seq-len 128
python -m src.generation.generate_music --model markov --num-samples 5 --seq-len 128
```

Build comparison artifacts:

```bash
python -m src.evaluation.build_comparison_table
python -m src.evaluation.rlhf_before_after
```

Compile report PDF:

```bash
cd report
pdflatex final_report.tex
```

## 6) Group Member Contributions

- **RAGIB RAWNAK (23101001)**: Preprocessing pipeline, dataset preparation, and implementation/experiments for **Task 1 (Autoencoder)** and **Task 2 (VAE)**.
- **Me**: Implementation/experiments for **Task 3 (Transformer)** and **Task 4 (RLHF-style fine-tuning)**, plus generation/evaluation artifacts for those tasks.
